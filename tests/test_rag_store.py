"""Integration tests for the LlamaIndex PGVectorStore knowledge store.

These need a running database and are skipped without one, so the offline suite
stays green on a machine with no Docker:

    docker compose up -d
    uv run python -m unittest tests.test_rag_store -v

The table is created by PGVectorStore on first use; there is no migration step.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import EMBEDDING_DIMENSIONS, database_url  # noqa: E402
from rag.store import KnowledgeStore  # noqa: E402

TEST_TABLE = "knowledge_chunk_test"


def _store_or_none():
    if database_url() is None:
        return None
    try:
        store = KnowledgeStore(table_name=TEST_TABLE)
        store.healthcheck()
        return store
    except Exception:  # noqa: BLE001 - absence of a database is the normal case
        return None


STORE = _store_or_none()


def vector(seed: float) -> list[float]:
    """A vector that differs by seed, for predictable ranking."""

    body = [0.0] * EMBEDDING_DIMENSIONS
    body[0] = 1.0
    body[1] = seed
    return body


def node(node_id: str, text: str, seed: float, document_id="mri", category="imaging"):
    from llama_index.core.schema import TextNode

    item = TextNode(
        id_=node_id,
        text=text,
        metadata={
            "corpus_document_id": document_id,
            "title": f"Title for {document_id}",
            "section": "A section",
            "category": category,
            "page_number": 2,
            "source_url": f"https://example.invalid/{document_id}",
            "last_updated": "2024-07-15",
            "content_fingerprint": f"hash-{document_id}:v1",
        },
    )
    item.embedding = vector(seed)
    return item


@unittest.skipIf(STORE is None, "no DATABASE_URL, or the store is unreachable")
class KnowledgeStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = STORE
        for document_id in ("mri", "ct-scans", "lab"):
            self.store.delete_document(document_id)

    def test_the_table_is_created_at_the_configured_dimension(self):
        health = self.store.healthcheck()

        self.assertIsNotNone(health["pgvector_version"])
        self.assertEqual(health["table"], f"data_{TEST_TABLE}")
        self.assertEqual(health["stored_dimension"], EMBEDDING_DIMENSIONS)
        self.assertIsNone(self.store.dimension_mismatch())

    def test_nodes_round_trip_with_citation_metadata(self):
        self.store.replace_document("mri", [node("n1", "MRI uses magnets.", 0.0)])

        results = self.store.search(vector(0.0), top_k=1)

        self.assertEqual(len(results), 1)
        found = results[0]
        self.assertEqual(found.document_id, "mri")
        self.assertEqual(found.title, "Title for mri")
        self.assertEqual(found.section, "A section")
        self.assertEqual(found.page_number, 2)
        self.assertEqual(found.last_updated, "2024-07-15")
        self.assertIn("MRI uses magnets.", found.text)

    def test_search_ranks_by_similarity_descending(self):
        self.store.replace_document(
            "mri",
            [
                node("n1", "closest", 0.0),
                node("n2", "middle", 0.5),
                node("n3", "furthest", 4.0),
            ],
        )

        results = self.store.search(vector(0.0), top_k=3)

        self.assertEqual(len(results), 3)
        similarities = [r.similarity for r in results]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
        self.assertIn("closest", results[0].text)

    def test_replace_document_removes_the_previous_nodes(self):
        self.store.replace_document("mri", [node("n1", "old text", 0.0)])
        self.store.replace_document("mri", [node("n2", "new text", 0.0)])

        results = self.store.search(vector(0.0), top_k=5)

        texts = " ".join(r.text for r in results)
        self.assertIn("new text", texts)
        self.assertNotIn("old text", texts)

    def test_the_stored_fingerprint_is_readable(self):
        self.store.replace_document("mri", [node("n1", "text", 0.0)])

        self.assertEqual(self.store.stored_fingerprint("mri"), "hash-mri:v1")
        self.assertIsNone(self.store.stored_fingerprint("absent-document"))

    def test_the_category_filter_restricts_results(self):
        self.store.replace_document("mri", [node("n1", "imaging text", 0.0)])
        self.store.replace_document(
            "lab", [node("n2", "lab text", 0.1, document_id="lab", category="lab_test")]
        )

        results = self.store.search(vector(0.0), top_k=5, categories=["lab_test"])

        self.assertEqual([r.document_id for r in results], ["lab"])

    def test_the_document_filter_restricts_results(self):
        self.store.replace_document("mri", [node("n1", "imaging text", 0.0)])
        self.store.replace_document(
            "ct-scans", [node("n2", "ct text", 0.1, document_id="ct-scans")]
        )

        results = self.store.search(vector(0.0), top_k=5, document_ids=["ct-scans"])

        self.assertEqual([r.document_id for r in results], ["ct-scans"])

    def test_an_unembedded_node_is_rejected_before_writing(self):
        from llama_index.core.schema import TextNode

        bare = TextNode(id_="bare", text="text", metadata={"corpus_document_id": "mri"})

        with self.assertRaises(ValueError):
            self.store.replace_document("mri", [bare])

    def test_a_wrong_dimension_is_rejected_before_writing(self):
        wrong = node("n1", "text", 0.0)
        wrong.embedding = [0.1, 0.2, 0.3]

        with self.assertRaises(ValueError) as raised:
            self.store.replace_document("mri", [wrong])

        self.assertIn("3", str(raised.exception))

    def test_corpus_status_counts_nodes_per_document(self):
        self.store.replace_document(
            "mri", [node("n1", "one", 0.0), node("n2", "two", 0.1)]
        )

        rows = {row["document_id"]: row for row in self.store.corpus_status()}

        self.assertEqual(rows["mri"]["nodes"], 2)
        self.assertEqual(rows["mri"]["category"], "imaging")


if __name__ == "__main__":
    unittest.main()
