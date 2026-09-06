"""Unit tests for sentence-window retrieval: expansion, merging, ranking (C2).

Offline and unpaid: the store and the embedding model are fakes, so the
behaviour under test is the window arithmetic rather than pgvector's ordering.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.retrievers import (  # noqa: E402
    RetrievalFilters,
    SentenceWindowRetriever,
    merge_windows,
)
from rag.sentence_window import (  # noqa: E402
    ORIGINAL_SENTENCE_KEY,
    SENTENCE_INDEX_KEY,
    WINDOW_CENTER_KEY,
    WINDOW_SENTENCES_KEY,
)
from rag.store import RetrievedChunk  # noqa: E402

# A section of ten sentences. Windows are cut from this, so a merged span can be
# checked against the source text rather than against another computation.
SECTION = [f"S{n}." for n in range(10)]


def hit(index, similarity, window=5, document_id="doc", section="Prep"):
    """Build a retrieved sentence exactly as ingestion would have stored it."""

    low = max(0, index - window)
    high = min(len(SECTION), index + window + 1)
    neighbourhood = SECTION[low:high]
    return RetrievedChunk(
        node_id=f"{document_id}-{section}-{index}",
        document_id=document_id,
        title="Title",
        category="lab_test",
        section=section,
        page_number=1,
        source_url=None,
        last_updated=None,
        text=SECTION[index],
        similarity=similarity,
        metadata={
            SENTENCE_INDEX_KEY: index,
            WINDOW_CENTER_KEY: index - low,
            WINDOW_SENTENCES_KEY: json.dumps(neighbourhood),
            ORIGINAL_SENTENCE_KEY: SECTION[index],
            "corpus_document_id": document_id,
            "section": section,
        },
    )


class ExpansionTests(unittest.TestCase):
    def test_a_hit_expands_to_its_window(self):
        [source] = merge_windows([hit(5, 0.9)], window_size=1)

        self.assertEqual(source.text, "S4. S5. S6.")

    def test_the_window_size_selects_how_much_context_comes_back(self):
        widths = {}
        for size in (0, 1, 2, 3, 5):
            [source] = merge_windows([hit(5, 0.9)], window_size=size)
            widths[size] = source.text

        self.assertEqual(widths[0], "S5.")
        self.assertEqual(widths[1], "S4. S5. S6.")
        self.assertEqual(widths[2], "S3. S4. S5. S6. S7.")
        self.assertEqual(len(widths[5].split()), 10)

    def test_the_window_is_clamped_at_a_section_edge(self):
        [first] = merge_windows([hit(0, 0.9)], window_size=3)
        [last] = merge_windows([hit(9, 0.9)], window_size=3)

        self.assertEqual(first.text, "S0. S1. S2. S3.")
        self.assertEqual(last.text, "S6. S7. S8. S9.")

    def test_citation_metadata_survives_expansion(self):
        [source] = merge_windows([hit(5, 0.9)], window_size=2)

        self.assertEqual(source.document_id, "doc")
        self.assertEqual(source.section, "Prep")
        self.assertEqual(source.page_number, 1)
        self.assertEqual(source.title, "Title")


class MergeTests(unittest.TestCase):
    def test_overlapping_windows_become_one_source(self):
        """The reason merging exists: no passage should be paid for twice."""

        merged = merge_windows([hit(4, 0.9), hit(5, 0.8)], window_size=2)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "S2. S3. S4. S5. S6. S7.")

    def test_a_sentence_appears_once_in_a_merged_span(self):
        [source] = merge_windows([hit(4, 0.9), hit(5, 0.8)], window_size=3)

        for sentence in source.text.split():
            self.assertEqual(source.text.split().count(sentence), 1)

    def test_touching_windows_merge_because_the_prose_is_continuous(self):
        # Window 0 at 3 and 4 covers [3,3] and [4,4]: adjacent, not overlapping.
        merged = merge_windows([hit(3, 0.9), hit(4, 0.8)], window_size=0)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "S3. S4.")

    def test_distant_windows_stay_separate(self):
        merged = merge_windows([hit(0, 0.9), hit(9, 0.8)], window_size=1)

        self.assertEqual(len(merged), 2)

    def test_a_chain_of_spans_merges_even_when_the_ends_do_not_touch(self):
        """A-B-C where A and C are far apart but B bridges them."""

        merged = merge_windows(
            [hit(0, 0.9), hit(3, 0.8), hit(6, 0.7)], window_size=2
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].text.split()), 9)

    def test_windows_in_different_sections_never_merge(self):
        merged = merge_windows(
            [hit(5, 0.9, section="Prep"), hit(5, 0.8, section="Risks")],
            window_size=2,
        )

        self.assertEqual(len(merged), 2)

    def test_windows_in_different_documents_never_merge(self):
        merged = merge_windows(
            [hit(5, 0.9, document_id="a"), hit(5, 0.8, document_id="b")],
            window_size=2,
        )

        self.assertEqual(len(merged), 2)

    def test_a_merged_span_keeps_the_best_similarity(self):
        """The evidence check reads sources[0]; a merge must not demote it."""

        [source] = merge_windows([hit(4, 0.42), hit(5, 0.91)], window_size=2)

        self.assertAlmostEqual(source.similarity, 0.91)

    def test_results_come_back_in_rank_order(self):
        merged = merge_windows(
            [hit(0, 0.30), hit(9, 0.95), hit(5, 0.60)], window_size=0
        )

        self.assertEqual(
            [round(s.similarity, 2) for s in merged], [0.95, 0.60, 0.30]
        )

    def test_a_node_without_a_stored_window_passes_through_unchanged(self):
        """A chunk from the Part A table must not be silently dropped."""

        chunk = RetrievedChunk(
            node_id="plain",
            document_id="doc",
            title="Title",
            category="lab_test",
            section="Prep",
            page_number=1,
            source_url=None,
            last_updated=None,
            text="A whole 400-token chunk.",
            similarity=0.5,
        )

        merged = merge_windows([chunk, hit(5, 0.9)], window_size=2)

        self.assertEqual(len(merged), 2)
        self.assertIn("A whole 400-token chunk.", [s.text for s in merged])

    def test_no_sources_yields_no_sources(self):
        self.assertEqual(merge_windows([], window_size=3), [])


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, embedding, top_k, categories=None, document_ids=None):
        self.calls.append(
            {"top_k": top_k, "categories": categories, "document_ids": document_ids}
        )
        return list(self.hits)


class FakeEmbedModel:
    def get_query_embedding(self, text):
        return [0.1, 0.2, 0.3]


class RetrieverTests(unittest.TestCase):
    def retriever(self, hits, window_size=2):
        self.store = FakeStore(hits)
        return SentenceWindowRetriever(self.store, FakeEmbedModel(), window_size)

    def test_the_strategy_is_declared_for_the_experiment_matrix(self):
        retriever = self.retriever([])

        self.assertEqual(retriever.strategy, "sentence_window")
        self.assertEqual(retriever.window_size, 2)

    def test_retrieval_expands_and_merges(self):
        sources = self.retriever([hit(4, 0.9), hit(5, 0.8)]).retrieve("q")

        self.assertEqual(len(sources), 1)

    def test_latency_is_reported_with_the_sources(self):
        sources = self.retriever([hit(5, 0.9)]).retrieve("q")

        self.assertGreaterEqual(sources.latency_ms, 0.0)

    def test_filters_reach_the_store(self):
        retriever = self.retriever([hit(5, 0.9)])

        retriever.retrieve(
            "q",
            top_k=4,
            filters=RetrievalFilters(categories=("lab_test",), document_ids=("doc",)),
        )

        self.assertEqual(self.store.calls[0]["top_k"], 4)
        self.assertEqual(self.store.calls[0]["categories"], ["lab_test"])
        self.assertEqual(self.store.calls[0]["document_ids"], ["doc"])

    def test_window_size_changes_the_context_returned(self):
        narrow = self.retriever([hit(5, 0.9)], window_size=1).retrieve("q")
        wide = self.retriever([hit(5, 0.9)], window_size=4).retrieve("q")

        self.assertLess(len(narrow[0].text), len(wide[0].text))


if __name__ == "__main__":
    unittest.main()
