"""Embedding model, via LlamaIndex.

`OpenAIEmbedding` handles batching and retries, so there is no hand-rolled
version of either here. What this module does own is the choice of model and the
guarantee that the dimension it produces matches the dimension the vector table
was built with — a mismatch there fails on the first insert with an error that
says nothing about the cause.

Unit tests use `MockEmbedding` from LlamaIndex rather than a bespoke fake, so
the offline suite stays offline and unpaid.
"""

from __future__ import annotations

try:
    from .config import EMBEDDING, EmbeddingProfile, SETTINGS, env_value
except ImportError:  # pragma: no cover - allows running as a script
    from config import EMBEDDING, EmbeddingProfile, SETTINGS, env_value


def build_embed_model(profile: EmbeddingProfile | None = None):
    """Return the LlamaIndex embedding model for the configured profile."""

    resolved = profile or EMBEDDING
    if resolved.backend != "openai":
        raise RuntimeError(f"Unknown embedding backend {resolved.backend!r}")

    from llama_index.embeddings.openai import OpenAIEmbedding

    key = env_value("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAIEmbedding(
        model=resolved.model,
        dimensions=resolved.dimensions,
        embed_batch_size=SETTINGS.embed_batch_size,
        api_key=key,
    )


def build_mock_embed_model(profile: EmbeddingProfile | None = None):
    """Return a deterministic embedding model of the right dimension, for tests."""

    from llama_index.core.embeddings import MockEmbedding

    resolved = profile or EMBEDDING
    return MockEmbedding(embed_dim=resolved.dimensions)


def embed_nodes(embed_model, nodes) -> None:
    """Attach embeddings to nodes in place, using the embedding metadata view.

    `MetadataMode.EMBED` is what puts the document title and section heading into
    the vectorised text while leaving citation bookkeeping out of it. Calling
    `node.text` here instead would silently drop the heading from every vector,
    and nothing downstream would report the loss.
    """

    from llama_index.core.schema import MetadataMode

    texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
    vectors = embed_model.get_text_embedding_batch(texts, show_progress=False)
    for node, vector in zip(nodes, vectors):
        node.embedding = vector


# Query embeddings are cached across a process. Retrieval spends ~235ms on the
# embedding round trip against ~10ms in pgvector, so a repeat question is 20x
# cheaper from the cache — and the benchmark asks each question more than once
# across a tuning sweep, where it turns a paid API call into a dict lookup.
_QUERY_CACHE: dict[tuple[str, str], list[float]] = {}


def embed_query(embed_model, question: str) -> list[float]:
    """Embed a search query.

    Separate from `embed_nodes` because the two are not always the same
    operation: asymmetric retrieval models want an instruction prefix on the
    query and none on the passage. OpenAI's are symmetric, so
    `query_instruction` is empty — but the seam is where it needs to be if that
    ever changes, and `get_query_embedding` is the call LlamaIndex routes
    through for it.
    """

    key = (EMBEDDING.model, question.strip())
    cached = _QUERY_CACHE.get(key)
    if cached is not None:
        return cached
    vector = embed_model.get_query_embedding(EMBEDDING.query_instruction + question)
    _QUERY_CACHE[key] = vector
    return vector
