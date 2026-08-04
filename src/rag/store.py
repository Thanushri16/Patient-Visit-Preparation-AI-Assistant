"""Knowledge store, via the LlamaIndex PGVectorStore.

LlamaIndex owns the physical layout: it creates the `vector` extension, the
`data_knowledge_chunk` table, the HNSW index, and the JSONB metadata column, and
it does the similarity query. This module is the project's vocabulary on top of
that — documents, fingerprints, retrieved chunks with citation metadata — so that
nothing above it needs to know about `TextNode` or `VectorStoreQuery`.

There is no document-version table. The corpus is 12 static PDFs, re-ingest is a
full replacement of a document's nodes, and Part A citations live in a session
response rather than durable storage. Versioning belongs with the FR-10 admin
portal that manages documents, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    from .config import (
        EMBEDDING_DIMENSIONS,
        EMBEDDING_MODEL,
        VECTOR_TABLE_NAME,
        require_database_url,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from config import (
        EMBEDDING_DIMENSIONS,
        EMBEDDING_MODEL,
        VECTOR_TABLE_NAME,
        require_database_url,
    )


@dataclass(frozen=True)
class RetrievedChunk:
    """A node returned by similarity search, with its citation metadata."""

    node_id: str
    document_id: str
    title: str
    category: str
    section: str | None
    page_number: int | None
    source_url: str | None
    last_updated: str | None
    text: str
    similarity: float


class KnowledgeStore:
    """Project-facing gateway to the LlamaIndex vector store."""

    def __init__(
        self,
        database_url: str | None = None,
        table_name: str = VECTOR_TABLE_NAME,
        embed_dim: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        self._url = database_url or require_database_url()
        self._table_name = table_name
        self._embed_dim = embed_dim
        self._store = None

    @property
    def table(self) -> str:
        """The physical table name. PGVectorStore prefixes what it is given."""

        return f"data_{self._table_name}"

    def vector_store(self):
        """Return the PGVectorStore, creating the schema on first use."""

        if self._store is None:
            from llama_index.vector_stores.postgres import PGVectorStore

            # Pass the URL as components, not as `connection_string`.
            # from_params() builds `async_connection_string` from host/port/user
            # unconditionally, so supplying only the connection string yields
            # "postgresql+asyncpg://None:None@None:None/None" and the store fails
            # to construct. Handing it the parts lets it build both the psycopg2
            # and asyncpg URLs correctly.
            parts = _split_url(self._url)
            self._store = PGVectorStore.from_params(
                host=parts["host"],
                port=parts["port"],
                database=parts["database"],
                user=parts["user"],
                password=parts["password"],
                table_name=self._table_name,
                embed_dim=self._embed_dim,
                # JSONB so metadata stays queryable; the ingest fingerprint check
                # and the category filter in the evidence guards both need it.
                use_jsonb=True,
                perform_setup=True,
                hnsw_kwargs={
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 64,
                    "hnsw_ef_search": 40,
                    "hnsw_dist_method": "vector_cosine_ops",
                },
                indexed_metadata_keys={
                    ("corpus_document_id", "text"),
                    ("category", "text"),
                },
            )
        return self._store

    # -- schema and health ---------------------------------------------------

    def healthcheck(self) -> dict[str, object]:
        """Confirm the store is reachable and sized for the configured model."""

        import sqlalchemy

        store = self.vector_store()
        engine = sqlalchemy.create_engine(self._url)
        try:
            with engine.connect() as connection:
                version = connection.execute(
                    sqlalchemy.text(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                    )
                ).scalar()
                dimension = connection.execute(
                    sqlalchemy.text(
                        """
                        SELECT a.atttypmod
                        FROM pg_attribute a
                        JOIN pg_class c ON c.oid = a.attrelid
                        WHERE c.relname = :table AND a.attname = 'embedding'
                        """
                    ),
                    {"table": self.table},
                ).scalar()
        finally:
            engine.dispose()

        return {
            "pgvector_version": version,
            "table": self.table,
            "stored_dimension": dimension,
            "expected_dimension": self._embed_dim,
            "embedding_model": EMBEDDING_MODEL,
            "store": type(store).__name__,
        }

    def dimension_mismatch(self) -> str | None:
        """Describe a stored/configured dimension mismatch, if any.

        Switching embedding model changes the vector width, and pgvector rejects
        the first insert with a type error that says nothing about the cause.
        Checking up front turns that into an instruction.
        """

        health = self.healthcheck()
        stored = health["stored_dimension"]
        if stored in (None, -1) or stored == self._embed_dim:
            return None
        return (
            f"The knowledge store holds {stored}-dimension vectors but the "
            f"configured model produces {self._embed_dim}. Changing embedding "
            "model requires a rebuild:\n"
            "  docker compose down -v && docker compose up -d\n"
            "  uv run python -m src.rag.ingest ingest"
        )

    # -- ingestion ----------------------------------------------------------

    def stored_fingerprint(self, document_id: str) -> str | None:
        """Return the fingerprint of a stored document, or None if absent."""

        nodes = self.vector_store().get_nodes(
            filters=_document_filter(document_id)
        )
        for node in nodes:
            fingerprint = (node.metadata or {}).get("content_fingerprint")
            if fingerprint:
                return str(fingerprint)
        return None

    def delete_document(self, document_id: str) -> None:
        """Remove every node belonging to a document."""

        self.vector_store().delete_nodes(filters=_document_filter(document_id))

    def replace_document(self, document_id: str, nodes: Sequence) -> int:
        """Replace a document's nodes with the ones supplied.

        Delete-then-add rather than upsert: chunk boundaries move when the
        cleaning or chunking code changes, so the new nodes are not a
        one-for-one replacement of the old and leftovers would be retrievable
        evidence that no longer exists in the source.
        """

        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"{node.node_id}: node has no embedding")
            if len(node.embedding) != self._embed_dim:
                raise ValueError(
                    f"{node.node_id}: embedding has {len(node.embedding)} "
                    f"dimensions, expected {self._embed_dim}"
                )

        self.delete_document(document_id)
        self.vector_store().add(list(nodes))
        return len(nodes)

    # -- retrieval ----------------------------------------------------------

    def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        categories: Sequence[str] | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the nearest nodes, with similarity reported so larger is better."""

        from llama_index.core.vector_stores.types import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
            VectorStoreQuery,
        )

        filters: list[MetadataFilter] = []
        if categories:
            filters.append(
                MetadataFilter(
                    key="category", value=list(categories), operator=FilterOperator.IN
                )
            )
        if document_ids:
            filters.append(
                MetadataFilter(
                    key=CORPUS_DOCUMENT_KEY,
                    value=list(document_ids),
                    operator=FilterOperator.IN,
                )
            )

        result = self.vector_store().query(
            VectorStoreQuery(
                query_embedding=list(embedding),
                similarity_top_k=top_k,
                filters=MetadataFilters(filters=filters) if filters else None,
            )
        )

        chunks: list[RetrievedChunk] = []
        for index, node in enumerate(result.nodes or []):
            metadata = node.metadata or {}
            similarity = (
                result.similarities[index]
                if result.similarities and index < len(result.similarities)
                else 0.0
            )
            chunks.append(
                RetrievedChunk(
                    node_id=node.node_id,
                    document_id=str(metadata.get("corpus_document_id", "")),
                    title=str(metadata.get("title", "")),
                    category=str(metadata.get("category", "")),
                    section=metadata.get("section") or None,
                    page_number=metadata.get("page_number"),
                    source_url=metadata.get("source_url") or None,
                    last_updated=metadata.get("last_updated") or None,
                    text=node.get_content(),
                    similarity=float(similarity),
                )
            )
        return chunks

    # -- reporting ----------------------------------------------------------

    def corpus_status(self) -> list[dict]:
        """Per-document node counts, for `ingest status`."""

        import sqlalchemy

        self.vector_store()  # ensure the table exists before querying it
        engine = sqlalchemy.create_engine(self._url)
        try:
            with engine.connect() as connection:
                rows = connection.execute(
                    sqlalchemy.text(
                        f"""
                        SELECT
                            metadata_::jsonb ->> 'corpus_document_id' AS document_id,
                            metadata_::jsonb ->> 'title'       AS title,
                            metadata_::jsonb ->> 'category'    AS category,
                            COUNT(*)                           AS nodes
                        FROM {self.table}
                        GROUP BY 1, 2, 3
                        ORDER BY 1
                        """
                    )
                ).mappings().all()
        finally:
            engine.dispose()
        return [dict(row) for row in rows]


def _split_url(url: str) -> dict[str, str]:
    """Split a postgres URL into the parts PGVectorStore.from_params wants."""

    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    if not parsed.hostname or not parsed.path.lstrip("/"):
        raise ValueError(
            f"DATABASE_URL is not a usable postgres URL: {url!r}. Expected "
            "postgresql://user:password@host:port/database"
        )
    return {
        "host": parsed.hostname,
        "port": str(parsed.port or 5432),
        "database": parsed.path.lstrip("/"),
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
    }


# "document_id", "doc_id" and "ref_doc_id" are reserved by LlamaIndex and are
# overwritten on write, so the corpus key has to be named around them.
CORPUS_DOCUMENT_KEY = "corpus_document_id"


def _document_filter(document_id: str):
    from llama_index.core.vector_stores.types import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )

    return MetadataFilters(
        filters=[
            MetadataFilter(
                key=CORPUS_DOCUMENT_KEY, value=document_id, operator=FilterOperator.EQ
            )
        ]
    )
