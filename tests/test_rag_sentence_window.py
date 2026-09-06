"""Unit tests for sentence splitting and window building (Part C, C1).

Offline and unpaid: no database, no embedding model. Documents are built by hand
from the same `LoadedDocument` shape ingestion uses.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core.schema import MetadataMode  # noqa: E402

from rag.documents import LoadedDocument, ManifestDocument, Section  # noqa: E402
from rag.sentence_window import (  # noqa: E402
    MAX_WINDOW,
    SENTENCE_INDEX_KEY,
    WINDOW_SENTENCES_KEY,
    WINDOW_TEXT_KEY,
    SentenceWindow,
    sentence_nodes_from_loaded_document,
    split_sentences,
    window_from_node,
)


class SplitSentencesTests(unittest.TestCase):
    def test_plain_sentences_split_on_terminal_punctuation(self):
        self.assertEqual(
            split_sentences("Fast for 8 hours. Bring your card."),
            ["Fast for 8 hours.", "Bring your card."],
        )

    def test_empty_input_yields_nothing(self):
        for text in ("", "   ", "\n"):
            self.assertEqual(split_sentences(text), [])

    def test_a_title_abbreviation_does_not_end_a_sentence(self):
        """'Ask Dr. Chen' must not become 'Ask Dr.' — a fragment embeds badly."""

        self.assertEqual(
            split_sentences("Ask Dr. Chen about it. He will call."),
            ["Ask Dr. Chen about it.", "He will call."],
        )

    def test_a_latin_abbreviation_does_not_end_a_sentence(self):
        parts = split_sentences("Take it twice, e.g. morning and night. Do not skip.")

        self.assertEqual(len(parts), 2)
        self.assertIn("e.g. morning and night", parts[0])

    def test_a_numbered_list_marker_does_not_end_a_sentence(self):
        parts = split_sentences("Steps: 1. Fast overnight. 2. Arrive early.")

        # The digit before the period is a list marker, so it does not split
        # there; the sentence after it still does.
        self.assertTrue(all(part.strip() for part in parts))
        self.assertTrue(any("Fast overnight" in part for part in parts))

    def test_text_without_terminal_punctuation_is_one_sentence(self):
        self.assertEqual(
            split_sentences("You need no special preparation"),
            ["You need no special preparation"],
        )

    def test_question_and_exclamation_marks_end_sentences(self):
        self.assertEqual(len(split_sentences("Do I fast? Yes! Bring water.")), 3)

    def test_no_text_is_lost(self):
        """Under-splitting is acceptable; dropping content is not."""

        text = "First one. Second one? Third one! Dr. Smith agrees."
        joined = " ".join(split_sentences(text))

        for word in ("First", "Second", "Third", "Smith"):
            self.assertIn(word, joined)


def loaded(sentences, heading="How do I prepare?"):
    manifest = ManifestDocument(
        document_id="doc",
        file="doc.pdf",
        title="Preparing for a Test",
        category="lab_test",
        page_shape="single_column",
        source_url="https://example.invalid/doc",
        last_updated=None,
        page_count=1,
        sections=(heading,),
    )
    return LoadedDocument(
        manifest=manifest,
        sections=[Section(heading=heading, page_number=1, text=" ".join(sentences))],
        content_hash="hash",
    )


class WindowBuildingTests(unittest.TestCase):
    SENTENCES = [f"Sentence number {n} is here." for n in range(12)]

    def nodes(self):
        return sentence_nodes_from_loaded_document(loaded(self.SENTENCES))

    def test_one_node_per_sentence(self):
        self.assertEqual(len(self.nodes()), len(self.SENTENCES))

    def test_the_embedded_unit_is_the_sentence_not_the_window(self):
        """The whole point of the strategy: embed narrow, return wide."""

        node = self.nodes()[6]
        embedded = node.get_content(metadata_mode=MetadataMode.EMBED)

        self.assertIn("Sentence number 6", embedded)
        self.assertNotIn("Sentence number 1 ", embedded)
        # The heading still travels with the node, as it does for chunks.
        self.assertIn("Preparing for a Test", embedded)

    def test_the_window_is_not_embedded(self):
        node = self.nodes()[6]
        embedded = node.get_content(metadata_mode=MetadataMode.EMBED)

        self.assertNotIn(WINDOW_TEXT_KEY, embedded)
        self.assertNotIn(WINDOW_SENTENCES_KEY, embedded)

    def test_the_sentence_is_not_embedded_twice(self):
        """`original_sentence` metadata duplicates the node text if left in."""

        node = self.nodes()[6]
        embedded = node.get_content(metadata_mode=MetadataMode.EMBED)

        self.assertEqual(embedded.count("Sentence number 6 is here."), 1)

    def test_the_stored_window_centres_on_its_own_sentence(self):
        for index, node in enumerate(self.nodes()):
            with self.subTest(index=index):
                window = window_from_node(node)
                self.assertEqual(
                    window.sentences[window.center].strip(),
                    node.get_content(metadata_mode=MetadataMode.NONE).strip(),
                )

    def test_sentence_index_is_recorded_in_order(self):
        indices = [n.metadata[SENTENCE_INDEX_KEY] for n in self.nodes()]

        self.assertEqual(indices, list(range(len(self.SENTENCES))))

    def test_a_narrower_window_is_a_slice_of_the_stored_one(self):
        window = window_from_node(self.nodes()[6])

        widths = [len(window.text(size).split()) for size in (1, 2, 3, 5)]

        self.assertEqual(widths, sorted(widths))
        self.assertLess(widths[0], widths[-1])

    def test_window_one_holds_exactly_three_sentences_mid_document(self):
        window = window_from_node(self.nodes()[6])

        self.assertEqual(window.text(1).count("Sentence number"), 3)

    def test_the_window_is_clamped_at_the_start(self):
        window = window_from_node(self.nodes()[0])

        self.assertEqual(window.center, 0)
        self.assertEqual(window.text(1).count("Sentence number"), 2)

    def test_the_window_is_clamped_at_the_end(self):
        window = window_from_node(self.nodes()[-1])

        self.assertEqual(window.text(1).count("Sentence number"), 2)

    def test_a_window_never_crosses_a_section(self):
        """Structural, via one Document per section — but worth pinning."""

        document = LoadedDocument(
            manifest=loaded([]).manifest,
            sections=[
                Section(heading="First", page_number=1, text="Alpha one. Alpha two."),
                Section(heading="Second", page_number=2, text="Beta one. Beta two."),
            ],
            content_hash="hash",
        )

        for node in sentence_nodes_from_loaded_document(document):
            with self.subTest(section=node.metadata["section"]):
                window = window_from_node(node)
                blob = " ".join(window.sentences)
                self.assertFalse("Alpha" in blob and "Beta" in blob)

    def test_stored_windows_are_json_so_they_survive_the_metadata_column(self):
        node = self.nodes()[3]

        self.assertIsInstance(node.metadata[WINDOW_SENTENCES_KEY], str)
        self.assertIsInstance(json.loads(node.metadata[WINDOW_SENTENCES_KEY]), list)

    def test_citation_metadata_survives_onto_every_sentence(self):
        for node in self.nodes():
            self.assertEqual(node.metadata["corpus_document_id"], "doc")
            self.assertEqual(node.metadata["title"], "Preparing for a Test")
            self.assertEqual(node.metadata["page_number"], 1)


class SentenceWindowSliceTests(unittest.TestCase):
    def window(self):
        return SentenceWindow(
            sentence="c",
            sentences=("a", "b", "c", "d", "e"),
            center=2,
        )

    def test_a_window_at_or_above_the_stored_width_returns_everything(self):
        self.assertEqual(self.window().text(MAX_WINDOW), "a b c d e")
        self.assertEqual(self.window().text(MAX_WINDOW + 3), "a b c d e")

    def test_a_window_of_one_returns_the_immediate_neighbours(self):
        self.assertEqual(self.window().text(1), "b c d")

    def test_a_window_of_zero_returns_the_sentence_alone(self):
        self.assertEqual(self.window().text(0), "c")


if __name__ == "__main__":
    unittest.main()
