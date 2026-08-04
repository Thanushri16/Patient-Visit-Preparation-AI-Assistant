"""Unit tests for the retriever protocol and Basic RAG retrieval.

The unit tests use a fake store and LlamaIndex's MockEmbedding, so they need
neither a database nor an API key. The integration class at the bottom seeds a
real pgvector table and is skipped without one, the same way
tests/test_rag_store.py is:

    docker compose up -d
    uv run python -m unittest tests.test_rag_retrievers -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking import count_tokens  # noqa: E402
from rag.config import EMBEDDING_DIMENSIONS, SETTINGS, database_url  # noqa: E402
from rag.embeddings import build_mock_embed_model  # noqa: E402
from rag.retrievers import (  # noqa: E402
    BasicChunkRetriever,
    RetrievalFilters,
    Retriever,
    RetrievedSource,
    assemble_context,
)
from rag.store import KnowledgeStore  # noqa: E402

TEST_TABLE = "knowledge_chunk_retriever_test"


def source(node_id: str, text: str, similarity: float, document_id="mri",
           category="imaging") -> RetrievedSource:
    return RetrievedSource(
        node_id=node_id,
        document_id=document_id,
        title=f"Title for {document_id}",
        category=category,
        section="A section",
        page_number=2,
        source_url=None,
        last_updated=None,
        text=text,
        similarity=similarity,
    )


class FakeStore:
    """Records what the retriever asked for, and returns what it is told to."""

    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []

    def search(self, embedding, top_k, categories=None, document_ids=None):
        self.calls.append(
            {
                "embedding": embedding,
                "top_k": top_k,
                "categories": categories,
                "document_ids": document_ids,
            }
        )
        return self.results[:top_k]


def retriever_over(results=()):
    store = FakeStore(results)
    return BasicChunkRetriever(store, build_mock_embed_model()), store


class RetrieverProtocolTests(unittest.TestCase):
    def test_the_basic_retriever_satisfies_the_protocol(self):
        instance, _ = retriever_over()

        self.assertIsInstance(instance, Retriever)

    def test_the_strategy_is_labelled_for_the_experiment_matrix(self):
        instance, _ = retriever_over()

        self.assertEqual(instance.strategy, "basic")
        self.assertIsNone(instance.window_size)


class BasicChunkRetrieverTests(unittest.TestCase):
    def test_results_come_back_ordered_by_similarity_descending(self):
        instance, _ = retriever_over(
            [
                source("n1", "middle", 0.5),
                source("n2", "best", 0.9),
                source("n3", "worst", 0.1),
            ]
        )

        results = instance.retrieve("how long do I fast?")

        self.assertEqual([r.node_id for r in results], ["n2", "n1", "n3"])

    def test_top_k_defaults_to_the_configured_value(self):
        instance, store = retriever_over()

        instance.retrieve("a question")

        self.assertEqual(store.calls[0]["top_k"], SETTINGS.top_k)

    def test_an_explicit_top_k_is_honoured(self):
        instance, store = retriever_over(
            [source(f"n{index}", "text", 0.9 - index / 10) for index in range(6)]
        )

        results = instance.retrieve("a question", top_k=2)

        self.assertEqual(store.calls[0]["top_k"], 2)
        self.assertEqual(len(results), 2)

    def test_filters_reach_the_store(self):
        instance, store = retriever_over()

        instance.retrieve(
            "a question",
            filters=RetrievalFilters(
                categories=("billing",), document_ids=("insurance-guide",)
            ),
        )

        self.assertEqual(store.calls[0]["categories"], ["billing"])
        self.assertEqual(store.calls[0]["document_ids"], ["insurance-guide"])

    def test_absent_filters_leave_the_query_unrestricted(self):
        instance, store = retriever_over()

        instance.retrieve("a question")

        self.assertIsNone(store.calls[0]["categories"])
        self.assertIsNone(store.calls[0]["document_ids"])

    def test_the_query_is_embedded_at_the_configured_dimension(self):
        instance, store = retriever_over()

        instance.retrieve("a question")

        self.assertEqual(len(store.calls[0]["embedding"]), EMBEDDING_DIMENSIONS)

    def test_an_empty_store_returns_an_empty_result_set(self):
        instance, _ = retriever_over()

        results = instance.retrieve("nothing matches this")

        self.assertEqual(list(results), [])
        self.assertGreaterEqual(results.latency_ms, 0.0)

    def test_retrieval_latency_is_recorded(self):
        instance, _ = retriever_over([source("n1", "text", 0.9)])

        results = instance.retrieve("a question")

        self.assertGreater(results.latency_ms, 0.0)
        self.assertLess(results.latency_ms, 10_000.0)


class RetrievalFiltersTests(unittest.TestCase):
    def test_filters_are_frozen(self):
        filters = RetrievalFilters(categories=("billing",))

        with self.assertRaises(Exception):
            filters.categories = ("imaging",)

    def test_an_unset_filter_reports_itself_empty(self):
        self.assertTrue(RetrievalFilters().is_empty())
        self.assertFalse(RetrievalFilters(document_ids=("mri",)).is_empty())


class ContextAssemblyTests(unittest.TestCase):
    def test_everything_within_budget_is_kept(self):
        sources = [source("n1", "short one", 0.9), source("n2", "short two", 0.8)]

        assembled = assemble_context(sources, max_tokens=100)

        self.assertEqual(len(assembled.sources), 2)
        self.assertEqual(assembled.dropped, 0)
        self.assertEqual(
            assembled.total_tokens,
            count_tokens("short one") + count_tokens("short two"),
        )

    def test_a_source_that_does_not_fit_is_dropped_whole(self):
        first = "word " * 40
        second = "other " * 40
        budget = count_tokens(first) + 5

        assembled = assemble_context(
            [source("n1", first, 0.9), source("n2", second, 0.8)], max_tokens=budget
        )

        self.assertEqual([s.node_id for s in assembled.sources], ["n1"])
        self.assertEqual(assembled.dropped, 1)
        # The kept source is byte-for-byte the retrieved text: a half chunk
        # would be unciteable.
        self.assertEqual(assembled.sources[0].text, first)
        self.assertLessEqual(assembled.total_tokens, budget)

    def test_a_source_larger_than_the_whole_budget_leaves_nothing(self):
        assembled = assemble_context([source("n1", "word " * 200, 0.9)], max_tokens=10)

        self.assertEqual(assembled.sources, ())
        self.assertEqual(assembled.total_tokens, 0)
        self.assertEqual(assembled.dropped, 1)

    def test_a_smaller_lower_ranked_source_still_fits(self):
        big = "word " * 60
        small = "tiny"
        budget = count_tokens(big) + count_tokens("skipped " * 60) - 1

        assembled = assemble_context(
            [
                source("n1", big, 0.9),
                source("n2", "skipped " * 60, 0.8),
                source("n3", small, 0.7),
            ],
            max_tokens=budget,
        )

        self.assertEqual([s.node_id for s in assembled.sources], ["n1", "n3"])
        self.assertEqual(assembled.dropped, 1)

    def test_the_budget_defaults_to_the_configured_maximum(self):
        oversized = [source(f"n{index}", "word " * 300, 0.9) for index in range(20)]

        assembled = assemble_context(oversized)

        self.assertLessEqual(assembled.total_tokens, SETTINGS.max_context_tokens)
        self.assertGreater(assembled.dropped, 0)

    def test_no_sources_assembles_an_empty_context(self):
        assembled = assemble_context([])

        self.assertEqual(assembled.sources, ())
        self.assertEqual(assembled.total_tokens, 0)
        self.assertEqual(assembled.dropped, 0)


# -- integration ------------------------------------------------------------


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


class StubEmbedModel:
    """Returns a fixed query vector, so ranking is the database's doing.

    The point of the integration class is that pgvector orders the rows, not
    that OpenAI embeds them; a real embedding call would spend money proving
    something the unit tests already cover.
    """

    def __init__(self, vector):
        self.vector = vector

    def get_query_embedding(self, text: str) -> list[float]:
        return list(self.vector)


def vector(seed: float) -> list[float]:
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
class BasicChunkRetrieverIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.store = STORE
        for document_id in ("mri", "lab"):
            self.store.delete_document(document_id)
        self.store.replace_document(
            "mri",
            [
                node("n1", "closest imaging text", 0.0),
                node("n2", "middle imaging text", 0.5),
                node("n3", "furthest imaging text", 4.0),
            ],
        )
        self.store.replace_document(
            "lab",
            [node("n4", "lab text", 0.1, document_id="lab", category="lab_test")],
        )
        self.retriever = BasicChunkRetriever(self.store, StubEmbedModel(vector(0.0)))

    def tearDown(self):
        for document_id in ("mri", "lab"):
            self.store.delete_document(document_id)

    def test_real_ranking_is_nearest_first(self):
        results = self.retriever.retrieve("imaging", top_k=4)

        self.assertIn("closest", results[0].text)
        similarities = [r.similarity for r in results]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
        self.assertGreater(results.latency_ms, 0.0)

    def test_top_k_limits_the_rows_returned(self):
        results = self.retriever.retrieve("imaging", top_k=2)

        self.assertEqual(len(results), 2)

    def test_a_category_filter_restricts_real_retrieval(self):
        results = self.retriever.retrieve(
            "anything", filters=RetrievalFilters(categories=("lab_test",))
        )

        self.assertEqual([r.document_id for r in results], ["lab"])

    def test_a_document_filter_restricts_real_retrieval(self):
        results = self.retriever.retrieve(
            "anything", filters=RetrievalFilters(document_ids=("lab",))
        )

        self.assertEqual([r.document_id for r in results], ["lab"])

    def test_a_filter_matching_nothing_retrieves_nothing(self):
        results = self.retriever.retrieve(
            "anything", filters=RetrievalFilters(document_ids=("no-such-document",))
        )

        self.assertEqual(list(results), [])

    def test_real_results_assemble_within_the_token_budget(self):
        results = self.retriever.retrieve("imaging", top_k=4)

        assembled = assemble_context(results, max_tokens=count_tokens(results[0].text))

        self.assertEqual(len(assembled.sources), 1)
        self.assertEqual(assembled.dropped, len(results) - 1)


if __name__ == "__main__":
    unittest.main()
