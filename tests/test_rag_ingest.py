"""Unit tests for ingestion planning and the embedding step.

No API calls and no database. The embedding model is LlamaIndex's MockEmbedding
and the store is a stub, so this runs in CI exactly like the rest of the suite.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import EMBEDDING, EMBEDDING_DIMENSIONS, PIPELINE_VERSION  # noqa: E402
from rag.chunking import nodes_from_loaded_document  # noqa: E402
from rag.documents import content_hash, load_document, load_manifest  # noqa: E402
from rag.embeddings import build_mock_embed_model, embed_nodes, embed_query  # noqa: E402
from rag.ingest import build_plan, ingest  # noqa: E402


class FakeStore:
    """Enough of KnowledgeStore for the planner and the ingest loop."""

    def __init__(self, fingerprints=None):
        self.fingerprints = fingerprints or {}
        self.saved: dict[str, list] = {}
        self.deleted: list[str] = []

    def stored_fingerprint(self, document_id):
        return self.fingerprints.get(document_id)

    def delete_document(self, document_id):
        self.deleted.append(document_id)

    def replace_document(self, document_id, nodes):
        self.deleted.append(document_id)
        self.saved[document_id] = list(nodes)
        return len(nodes)


def fingerprint_of(document_id):
    manifest = next(m for m in load_manifest()[0] if m.document_id == document_id)
    return f"{content_hash(manifest.path)}:v{PIPELINE_VERSION}"


class EmbeddingTests(unittest.TestCase):
    def test_nodes_are_embedded_at_the_configured_dimension(self):
        manifest = load_manifest()[0][0]
        nodes = nodes_from_loaded_document(load_document(manifest))

        embed_nodes(build_mock_embed_model(), nodes)

        self.assertTrue(nodes)
        for node in nodes:
            self.assertEqual(len(node.embedding), EMBEDDING_DIMENSIONS)

    def test_a_query_embedding_has_the_same_dimension(self):
        vector = embed_query(build_mock_embed_model(), "do I need to fast?")

        self.assertEqual(len(vector), EMBEDDING_DIMENSIONS)

    def test_the_configured_profile_is_openai(self):
        self.assertEqual(EMBEDDING.backend, "openai")
        self.assertEqual(EMBEDDING.model, "text-embedding-3-small")
        self.assertEqual(EMBEDDING.dimensions, 1536)


class PlanTests(unittest.TestCase):
    def test_an_empty_store_plans_to_ingest_everything(self):
        plans = build_plan(FakeStore())

        self.assertEqual(len(plans), 11)
        self.assertTrue(all(plan.action == "ingest" for plan in plans))
        self.assertTrue(all(plan.nodes > 0 for plan in plans))

    def test_no_store_still_reports_what_would_be_built(self):
        plans = build_plan(None)

        self.assertEqual(len(plans), 11)
        # Sensitive to metadata length by design: the splitter budgets against
        # the longer of the embed and LLM metadata views, so adding a metadata
        # key legitimately moves this number. It is pinned as a regression
        # signal, not because 88 is significant.
        self.assertEqual(sum(plan.nodes for plan in plans), 88)

    def test_an_unchanged_document_is_skipped(self):
        store = FakeStore({"mri": fingerprint_of("mri")})

        plans = {plan.manifest.document_id: plan for plan in build_plan(store)}

        self.assertEqual(plans["mri"].action, "skip")
        self.assertEqual(plans["mri"].reason, "unchanged")

    def test_a_pipeline_version_bump_forces_a_reingest(self):
        """Cleaning and chunking are code. Changing them must re-ingest."""

        stale = fingerprint_of("mri").replace(f":v{PIPELINE_VERSION}", ":v0")
        store = FakeStore({"mri": stale})

        plans = {plan.manifest.document_id: plan for plan in build_plan(store)}

        self.assertEqual(plans["mri"].action, "reingest")
        self.assertEqual(plans["mri"].reason, "pipeline version changed")

    def test_a_changed_pdf_forces_a_reingest(self):
        store = FakeStore({"mri": f"deadbeef:v{PIPELINE_VERSION}"})

        plans = {plan.manifest.document_id: plan for plan in build_plan(store)}

        self.assertEqual(plans["mri"].action, "reingest")
        self.assertEqual(plans["mri"].reason, "source PDF changed")

    def test_force_reingests_an_unchanged_document(self):
        store = FakeStore({"mri": fingerprint_of("mri")})

        plans = {
            plan.manifest.document_id: plan for plan in build_plan(store, force=True)
        }

        self.assertEqual(plans["mri"].action, "reingest")
        self.assertEqual(plans["mri"].reason, "forced")


class IngestTests(unittest.TestCase):
    def test_ingest_stores_every_document_with_embedded_nodes(self):
        store = FakeStore()

        ingest(store, build_mock_embed_model())

        self.assertEqual(len(store.saved), 11)
        self.assertEqual(sum(len(nodes) for nodes in store.saved.values()), 88)
        for document_id, nodes in store.saved.items():
            with self.subTest(document_id):
                for node in nodes:
                    self.assertEqual(len(node.embedding), EMBEDDING_DIMENSIONS)

    def test_node_ids_are_unique_across_the_corpus(self):
        store = FakeStore()

        ingest(store, build_mock_embed_model())

        ids = [node.node_id for nodes in store.saved.values() for node in nodes]
        self.assertEqual(len(ids), len(set(ids)))

    def test_citation_metadata_reaches_the_store(self):
        store = FakeStore()

        ingest(store, build_mock_embed_model())

        for document_id, nodes in store.saved.items():
            with self.subTest(document_id):
                for node in nodes:
                    self.assertEqual(node.metadata["corpus_document_id"], document_id)
                    self.assertTrue(node.metadata["title"])
                    self.assertTrue(node.metadata["source_url"])
                    self.assertTrue(
                        node.metadata["content_fingerprint"].endswith(
                            f":v{PIPELINE_VERSION}"
                        )
                    )

    def test_a_second_run_stores_nothing(self):
        store = FakeStore()
        ingest(store, build_mock_embed_model())
        store.fingerprints = {
            document_id: nodes[0].metadata["content_fingerprint"]
            for document_id, nodes in store.saved.items()
        }
        before = dict(store.saved)

        plans = ingest(store, build_mock_embed_model())

        self.assertEqual(store.saved, before)
        self.assertTrue(all(plan.action == "skip" for plan in plans))

    def test_reingesting_replaces_rather_than_appends(self):
        """Chunk boundaries move when the pipeline changes, so leftovers would
        be retrievable evidence that no longer exists in the source."""

        store = FakeStore({"mri": f"deadbeef:v{PIPELINE_VERSION}"})

        ingest(store, build_mock_embed_model())

        self.assertIn("mri", store.deleted)


if __name__ == "__main__":
    unittest.main()
