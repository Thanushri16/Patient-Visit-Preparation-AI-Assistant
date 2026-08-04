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
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

try:
    from .chunking import count_tokens
    from .config import SETTINGS
    from .embeddings import embed_query
    from .store import KnowledgeStore, RetrievedChunk
except ImportError:  # pragma: no cover - allows running as a script
    from chunking import count_tokens
    from config import SETTINGS
    from embeddings import embed_query
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
        embedding = embed_query(self._embed_model, query)
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
