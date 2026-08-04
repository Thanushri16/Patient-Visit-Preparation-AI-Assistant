"""Chunking, via the LlamaIndex node parser.

The plan specifies 400-token chunks with 50 tokens of overlap, which is
`TokenTextSplitter` configured directly. Two properties are not settings on the
splitter and are worth stating, because both are load-bearing:

1. **A node never spans two sections.** That is not enforced here. It falls out
   of `documents.py` emitting one `Document` per section — a node parser splits
   within a document and never across documents, so the boundary is structural
   rather than a rule that could be forgotten.

2. **The heading travels with the node.** `title` and `section` are included in
   the embedded text via the metadata mode (see `EMBEDDED_METADATA_KEYS` in
   documents.py), so a node taken from the middle of a section still states what
   it is about. `TokenTextSplitter` subtracts the metadata length from the chunk
   budget itself, so the 400 covers text and heading together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken

try:
    from .config import EMBEDDING_ENCODING, MAX_MODEL_INPUT_TOKENS, SETTINGS
    from .documents import LoadedDocument, to_llamaindex_documents
except ImportError:  # pragma: no cover - allows running as a script
    from config import EMBEDDING_ENCODING, MAX_MODEL_INPUT_TOKENS, SETTINGS
    from documents import LoadedDocument, to_llamaindex_documents

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core.schema import BaseNode, Document


_ENCODING = None


def encoding() -> tiktoken.Encoding:
    """Return the tokenizer, loaded once."""

    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding(EMBEDDING_ENCODING)
    return _ENCODING


def count_tokens(text: str) -> int:
    return len(encoding().encode(text))


def build_node_parser(chunk_size: int | None = None, overlap: int | None = None):
    """Return the configured TokenTextSplitter.

    The splitter is given the same tokenizer the budget is measured with, so
    "400 tokens" means the same thing here as everywhere else in the pipeline.
    """

    from llama_index.core.node_parser import TokenTextSplitter

    size = chunk_size if chunk_size is not None else SETTINGS.chunk_size_tokens
    step = overlap if overlap is not None else SETTINGS.chunk_overlap_tokens

    # The model truncates input past its limit silently: text beyond it would
    # simply not be embedded, and nothing downstream would report it.
    if size > MAX_MODEL_INPUT_TOKENS:
        raise ValueError(
            f"chunk_size_tokens={size} exceeds the embedding model's "
            f"{MAX_MODEL_INPUT_TOKENS}-token input limit; text past the limit "
            "would be silently discarded."
        )

    return TokenTextSplitter(
        chunk_size=size,
        chunk_overlap=step,
        tokenizer=encoding().encode,
        include_metadata=True,
        include_prev_next_rel=False,
    )


def nodes_from_documents(
    documents: list["Document"],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list["BaseNode"]:
    """Split LlamaIndex documents into nodes."""

    parser = build_node_parser(chunk_size, overlap)
    return parser.get_nodes_from_documents(documents, show_progress=False)


def nodes_from_loaded_document(
    document: LoadedDocument,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list["BaseNode"]:
    """Convenience path: cleaned document straight to nodes."""

    return nodes_from_documents(
        to_llamaindex_documents(document), chunk_size, overlap
    )
