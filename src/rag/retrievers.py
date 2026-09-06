"""Retrieval, behind the protocol Part C swaps against.

Everything downstream of this module — the evidence check, the generation
prompt, citation binding — is written against `Retriever` and never against a
concrete strategy. That is what makes the Part C comparison honest: exchanging
`BasicChunkRetriever` for a sentence-window retriever is a construction-site
change, not an edit to any caller.

The protocol is deliberately thin. A retriever answers one question — what is
the evidence for this query — and declares which strategy produced it, because
the experiment matrix has to label its rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Protocol, Sequence, runtime_checkable

try:
    from .chunking import count_tokens
    from .config import SETTINGS
    from .embeddings import embed_query
    from .query import expand_query
    from .store import KnowledgeStore, RetrievedChunk
except ImportError:  # pragma: no cover - allows running as a script
    from chunking import count_tokens
    from config import SETTINGS
    from embeddings import embed_query
    from query import expand_query
    from store import KnowledgeStore, RetrievedChunk


# The plan names the retrieval result `RetrievedSource` (§6.1) and the store
# already returns `RetrievedChunk` (§5) with the same identity and citation
# fields plus `category`, `source_url` and `last_updated`. A second dataclass
# here would either drop `category` — which the A.4.2 category-consistency guard
# needs — or copy all ten fields for the sake of a different name. So the
# retrieval layer reuses the store's type under the plan's name.
#
# This is not the `RetrievedSource` of §6.2. That one is a pydantic model in
# state, a serializable projection of this that A7 adds when the RAG block joins
# `ConversationState`; state carries what a citation needs, not the whole node
# record. Two types with one name is a real cost, so the projection is A7's to
# name — the alias here is the retrieval layer's word for its own result.
RetrievedSource = RetrievedChunk


@dataclass(frozen=True)
class RetrievalFilters:
    """Metadata restrictions carried into the vector query.

    Frozen because filters are derived from the turn and read by the retriever,
    the evidence check and the metrics harness in turn; a mutable value passed
    through three consumers is a value nobody owns.
    """

    categories: tuple[str, ...] | None = None
    document_ids: tuple[str, ...] | None = None

    def is_empty(self) -> bool:
        return not self.categories and not self.document_ids


class RetrievedSources(list):
    """The retrieved evidence, with the latency that producing it cost.

    A list subclass rather than a wrapper object: the protocol returns a list
    because every caller iterates the evidence, and A.7 wants
    `retrieval_latency_ms` from the same call without a second return value or a
    stopwatch at each call site.
    """

    def __init__(
        self, sources: Sequence[RetrievedSource] = (), latency_ms: float = 0.0
    ) -> None:
        super().__init__(sources)
        self.latency_ms = latency_ms


@runtime_checkable
class Retriever(Protocol):
    """Returns the evidence for a query, whatever the strategy underneath."""

    strategy: str          # "basic" | "sentence_window"
    window_size: int | None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedSource]: ...


class BasicChunkRetriever:
    """Part A retrieval: embed the query, take the nearest fixed-size chunks.

    The store and the embedding model are injected rather than constructed here,
    matching how `client` and `visit_repository` reach the chain: a retriever
    built at import time would open a database connection in every test process
    that imports the module, and would fix one embedding profile for the whole
    experiment matrix.
    """

    strategy = "basic"
    window_size: int | None = None

    def __init__(self, store: KnowledgeStore, embed_model) -> None:
        self._store = store
        self._embed_model = embed_model

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedSource]:
        resolved_top_k = top_k if top_k is not None else SETTINGS.top_k
        active = filters or RetrievalFilters()

        # Measured around the embedding call as well as the query: the wait a
        # turn actually pays for retrieval includes the round trip to the
        # embedding API, and a number that excluded it would flatter the branch
        # against its 3-second timeout.
        started = time.perf_counter()
        embedding = embed_query(self._embed_model, expand_query(query))
        chunks = self._store.search(
            embedding,
            top_k=resolved_top_k,
            categories=list(active.categories) if active.categories else None,
            document_ids=list(active.document_ids) if active.document_ids else None,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # pgvector already returns nearest-first, but the ordering is a promise
        # of this interface rather than of the backend behind it, and the
        # evidence check reads `sources[0]` as the top match.
        ranked = sorted(chunks, key=lambda chunk: chunk.similarity, reverse=True)
        return RetrievedSources(ranked, latency_ms=elapsed_ms)


@dataclass(frozen=True)
class AssembledContext:
    """The sources that fit the context budget, and the count that did not."""

    sources: tuple[RetrievedSource, ...]
    total_tokens: int
    dropped: int


def assemble_context(
    sources: Sequence[RetrievedSource], max_tokens: int | None = None
) -> AssembledContext:
    """Take sources in rank order until the token budget is spent.

    Whole sources only. A chunk cut in half is unciteable — the answer would
    carry a marker pointing at a passage the reader cannot find in the document
    at that page, and a truncated passage can reverse the sense of the sentence
    it was cut from ("do not eat" / "do not eat before 8am"). Dropping the
    weakest matches instead costs the least: they are the ones least likely to
    be evidence, and `dropped` records what went so the metrics can see when the
    budget is the binding constraint.

    A single source larger than the whole budget is dropped like any other,
    which leaves an empty context and lets the A.5 evidence check refuse — the
    right outcome, rather than generating from a fragment.
    """

    budget = max_tokens if max_tokens is not None else SETTINGS.max_context_tokens
    kept: list[RetrievedSource] = []
    used = 0
    dropped = 0

    for source in sources:
        cost = count_tokens(source.text)
        if used + cost > budget:
            # Keep scanning rather than stopping: a shorter lower-ranked source
            # may still fit, and leaving budget unspent retrieves less evidence
            # than was paid for.
            dropped += 1
            continue
        kept.append(source)
        used += cost

    return AssembledContext(sources=tuple(kept), total_tokens=used, dropped=dropped)


# ---------------------------------------------------------------------------
# Part C: sentence-window retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Span:
    """One retrieved sentence, resolved to absolute positions in its section."""

    low: int              # first sentence index covered, inclusive
    high: int             # last sentence index covered, inclusive
    sentences: dict       # absolute index -> sentence text
    source: RetrievedSource


class SentenceWindowRetriever:
    """Part C retrieval: match on one sentence, return its neighbourhood.

    The window size is a query-time argument, not an ingestion decision. C1
    stores the widest window the matrix measures and the sentences behind it, so
    every arm of the experiment reads the same rows and the comparison cannot be
    contaminated by one strategy having been ingested from a different corpus
    revision than another.

    Overlapping windows are merged rather than emitted twice. At window 5 two
    adjacent hits share most of their text, and paying for it twice would make
    wide windows look expensive on context and strong on recall for the same
    artefact -- the duplicate would be counted once against the token budget for
    every hit that carried it, and the model would read the same passage
    repeatedly as if it were independent evidence.
    """

    strategy = "sentence_window"

    def __init__(
        self,
        store: KnowledgeStore,
        embed_model,
        window_size: int = 3,
    ) -> None:
        self._store = store
        self._embed_model = embed_model
        self.window_size = window_size

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedSource]:
        resolved_top_k = top_k if top_k is not None else SETTINGS.top_k
        active = filters or RetrievalFilters()

        started = time.perf_counter()
        embedding = embed_query(self._embed_model, expand_query(query))
        hits = self._store.search(
            embedding,
            top_k=resolved_top_k,
            categories=list(active.categories) if active.categories else None,
            document_ids=list(active.document_ids) if active.document_ids else None,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        ranked = sorted(hits, key=lambda hit: hit.similarity, reverse=True)
        merged = merge_windows(ranked, self.window_size)
        return RetrievedSources(merged, latency_ms=elapsed_ms)


def _span_for(source: RetrievedSource, window_size: int) -> _Span | None:
    """Resolve one hit to the absolute sentence range its window covers."""

    from_node = window_from_node_metadata(source.metadata)
    if from_node is None:
        return None

    window, index = from_node
    # sentences[j] sits at absolute index (index - center + j), so the stored
    # array's own bounds clamp the window at the section edges for free.
    offset = index - window.center
    low = max(0, window.center - window_size)
    high = min(len(window.sentences) - 1, window.center + window_size)
    return _Span(
        low=offset + low,
        high=offset + high,
        sentences={offset + j: window.sentences[j] for j in range(low, high + 1)},
        source=source,
    )


def merge_windows(
    sources: Sequence[RetrievedSource], window_size: int
) -> list[RetrievedSource]:
    """Expand each hit to its window, then merge overlaps within a section.

    Rank order is preserved by the strongest member: a merged span takes the
    position and similarity of its best hit, because the evidence check reads
    `sources[0]` as the top match and a merge must not demote it.
    """

    spans: list[_Span] = []
    passthrough: list[RetrievedSource] = []
    for source in sources:
        span = _span_for(source, window_size)
        # A hit with no stored window is not a sentence node. Returning it
        # unchanged keeps the retriever usable against a mixed table rather
        # than dropping evidence for a reason the caller cannot see.
        (spans if span is not None else passthrough).append(
            span if span is not None else source
        )

    groups: dict[tuple[str, str], list[_Span]] = {}
    for span in spans:
        key = (span.source.document_id, span.source.section or "")
        groups.setdefault(key, []).append(span)

    merged: list[RetrievedSource] = []
    for group in groups.values():
        for cluster in _cluster(group):
            merged.append(_combine(cluster))

    merged.extend(passthrough)
    return sorted(merged, key=lambda source: source.similarity, reverse=True)


def _cluster(spans: list[_Span]) -> list[list[_Span]]:
    """Group spans that overlap or touch, scanning left to right.

    The running extent is tracked as it grows, not taken from the first member:
    three spans can chain A-B-C where A and C do not touch each other directly,
    and comparing only against A would split one continuous passage in two.
    """

    clusters: list[list[_Span]] = []
    current: list[_Span] = []
    reach = 0

    for span in sorted(spans, key=lambda s: s.low):
        # Adjacency counts as overlap: windows ending at 4 and starting at 5 are
        # contiguous prose, and emitting them separately would show the model a
        # seam that is not in the document.
        if current and span.low <= reach + 1:
            current.append(span)
            reach = max(reach, span.high)
        else:
            if current:
                clusters.append(current)
            current = [span]
            reach = span.high

    if current:
        clusters.append(current)
    return clusters


def _combine(cluster: list[_Span]) -> RetrievedSource:
    """Fuse a cluster into one source covering the union of its sentences."""

    best = max(cluster, key=lambda span: span.source.similarity).source
    sentences: dict[int, str] = {}
    for span in cluster:
        sentences.update(span.sentences)

    text = " ".join(sentences[index] for index in sorted(sentences)).strip()
    return replace(best, text=text)


def window_from_node_metadata(metadata: dict):
    """Read a stored window out of raw node metadata, or None if absent."""

    try:
        from .sentence_window import SENTENCE_INDEX_KEY, window_from_metadata
    except ImportError:  # pragma: no cover - allows running as a script
        from sentence_window import SENTENCE_INDEX_KEY, window_from_metadata

    window = window_from_metadata(metadata)
    if window is None:
        return None
    return window, int(metadata.get(SENTENCE_INDEX_KEY, 0))
