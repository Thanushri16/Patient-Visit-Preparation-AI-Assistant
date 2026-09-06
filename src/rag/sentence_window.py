"""Sentence-window parsing and window building (Part C, step C1).

The Part A unit of retrieval is a 400-token chunk: large enough to dilute the
embedding, small enough to cut context. Sentence-window retrieval separates the
two jobs — embed one sentence so the match is precise, return its neighbourhood
so the model still has the context to answer from.

Ingestion runs once and serves every window size. Each sentence is stored with
the widest window the experiment matrix needs (±5) *and* with the sentences of
that window kept separately, so a narrower window is a slice at query time
rather than a re-ingest. Re-splitting the stored window text at query time would
be the obvious alternative and is worse: it depends on the splitter behaving
identically at query time as at ingestion time, and a drift there would silently
shift which sentences a citation covers.

Section-awareness is structural, not a rule enforced here. `documents.py` emits
one `Document` per section, and a node parser never splits across documents, so
a window cannot span two sections and a heading cannot merge into the sentence
after it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    from .documents import (
        EXCLUDED_FROM_EMBEDDING,
        EXCLUDED_FROM_PROMPT,
        LoadedDocument,
        to_llamaindex_documents,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from documents import (
        EXCLUDED_FROM_EMBEDDING,
        EXCLUDED_FROM_PROMPT,
        LoadedDocument,
        to_llamaindex_documents,
    )

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core.schema import BaseNode

# The widest window the matrix measures. Everything narrower is sliced from it.
MAX_WINDOW = 5

# Metadata keys this module adds on top of the citation keys in documents.py.
#
# Excluded from the embedding: the embedded text must be the sentence alone, or
# the precision the strategy exists for is lost. Excluded from the prompt too --
# the window is put in front of the model as the node's text at query time, not
# as a metadata blob repeated alongside it.
WINDOW_TEXT_KEY = "window_text"
# The parser copies the sentence into metadata under this key. The node's text
# is already that sentence, so leaving it embedded vectorises the sentence
# twice and dilutes it with its own duplicate.
ORIGINAL_SENTENCE_KEY = "original_sentence"
WINDOW_SENTENCES_KEY = "window_sentences"
WINDOW_CENTER_KEY = "window_center"
SENTENCE_INDEX_KEY = "sentence_index"

WINDOW_METADATA_KEYS = (
    WINDOW_TEXT_KEY,
    ORIGINAL_SENTENCE_KEY,
    WINDOW_SENTENCES_KEY,
    WINDOW_CENTER_KEY,
    SENTENCE_INDEX_KEY,
)


# Tokens that end in a period without ending a sentence. Splitting after them
# would cut "take 1 tablet by mouth twice a day, e.g. morning and evening" in
# half and leave a fragment as the embedded unit -- the exact failure sentence
# windows exist to avoid, since a fragment embeds poorly and cites badly.
_ABBREVIATIONS = (
    "dr", "drs", "mr", "mrs", "ms", "prof", "st", "no", "vs", "approx",
    "e.g", "i.e", "etc", "cf", "al", "fig", "inc", "ltd", "co",
    "a.m", "p.m", "u.s", "u.k", "mg", "ml", "mcg", "kg", "oz", "hr", "min",
)

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, deterministically and without NLTK.

    LlamaIndex's default splitter uses NLTK punkt, which needs a model download
    at first use. That puts a network fetch inside ingestion and makes sentence
    boundaries depend on a downloaded artefact rather than on this repository --
    two sentences differing between machines would silently change what a
    citation covers, and every stored window with it.

    Deliberately conservative: it under-splits rather than over-splits. A window
    that carries one sentence too many costs a few tokens; a fragment embedded
    as if it were a sentence is a retrieval miss.
    """

    if not text or not text.strip():
        return []

    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        candidate = text[start:match.start()].strip()
        if not candidate:
            continue
        # The token before the period, lowercased and stripped of punctuation.
        tail = candidate.rstrip("\"')]").rstrip(".")
        last = re.split(r"[\s]", tail)[-1].lower() if tail else ""
        if last in _ABBREVIATIONS:
            continue
        # A single letter or digit before the period is an initial or a list
        # marker ("J. Smith", "1. Fast for 8 hours"), not a sentence end.
        if len(last) <= 1 and last.isalnum():
            continue
        sentences.append(candidate)
        start = match.end()

    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def build_sentence_parser(window_size: int = MAX_WINDOW):
    """Return the configured SentenceWindowNodeParser.

    `window_size` is the number of sentences on *each* side, which is what
    LlamaIndex means by it and what section C.2 of the plan means by "±5".
    """

    from llama_index.core.node_parser import SentenceWindowNodeParser

    return SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key=WINDOW_TEXT_KEY,
        original_text_metadata_key=ORIGINAL_SENTENCE_KEY,
        sentence_splitter=split_sentences,
    )


@dataclass(frozen=True)
class SentenceWindow:
    """A sentence and the neighbourhood stored with it."""

    sentence: str
    sentences: tuple[str, ...]
    center: int

    def text(self, window_size: int) -> str:
        """Return the window narrowed to `window_size` sentences per side."""

        if window_size >= MAX_WINDOW:
            return " ".join(self.sentences).strip()
        low = max(0, self.center - window_size)
        high = min(len(self.sentences), self.center + window_size + 1)
        return " ".join(self.sentences[low:high]).strip()


def window_from_metadata(metadata: dict) -> SentenceWindow | None:
    """Read a stored window out of node metadata, or None if there is not one.

    Returns None rather than an empty window for a node that was never built by
    this module -- a chunk from the Part A table, say. The caller can then pass
    it through untouched instead of silently retrieving nothing for it.
    """

    raw = metadata.get(WINDOW_SENTENCES_KEY)
    if not raw:
        return None
    sentences = tuple(json.loads(raw) if isinstance(raw, str) else raw)
    if not sentences:
        return None
    return SentenceWindow(
        sentence=metadata.get(ORIGINAL_SENTENCE_KEY, ""),
        sentences=sentences,
        center=int(metadata.get(WINDOW_CENTER_KEY, 0)),
    )


def window_from_node(node) -> SentenceWindow:
    """Read back the window stored on a node at ingestion time."""

    window = window_from_metadata(node.metadata)
    if window is None:
        return SentenceWindow(
            sentence=node.get_content(), sentences=(node.get_content(),), center=0
        )
    if not window.sentence:
        window = SentenceWindow(
            sentence=node.get_content(),
            sentences=window.sentences,
            center=window.center,
        )
    return window


def sentence_nodes_from_loaded_document(
    document: LoadedDocument, window_size: int = MAX_WINDOW
) -> list["BaseNode"]:
    """Parse one document into sentence nodes carrying their windows.

    The node's own text stays the sentence, because that is what gets embedded.
    The window travels in metadata and is swapped in at query time.
    """

    parser = build_sentence_parser(window_size)
    nodes: list[BaseNode] = []

    for section_document in to_llamaindex_documents(document):
        section_nodes = parser.get_nodes_from_documents([section_document])
        # Position within the section, so a narrower window can be sliced and a
        # citation can say where in the section the sentence sits.
        sentences = [node.get_content() for node in section_nodes]

        for index, node in enumerate(section_nodes):
            low = max(0, index - window_size)
            high = min(len(sentences), index + window_size + 1)
            neighbourhood = sentences[low:high]

            node.metadata[SENTENCE_INDEX_KEY] = index
            node.metadata[WINDOW_CENTER_KEY] = index - low
            # Stored as JSON text rather than a list: the metadata column is
            # JSONB, but LlamaIndex flattens metadata for the embedding and the
            # prompt, and a list round-trips inconsistently across those paths.
            node.metadata[WINDOW_SENTENCES_KEY] = json.dumps(neighbourhood)
            node.metadata[WINDOW_TEXT_KEY] = " ".join(neighbourhood).strip()

            node.excluded_embed_metadata_keys = list(
                dict.fromkeys([*EXCLUDED_FROM_EMBEDDING, *WINDOW_METADATA_KEYS])
            )
            node.excluded_llm_metadata_keys = list(
                dict.fromkeys([*EXCLUDED_FROM_PROMPT, *WINDOW_METADATA_KEYS])
            )
            nodes.append(node)

    return [node for node in nodes if node.get_content().strip()]
