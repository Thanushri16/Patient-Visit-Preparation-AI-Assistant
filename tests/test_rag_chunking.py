"""Unit tests for the LlamaIndex node parser pipeline."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llama_index.core.schema import MetadataMode  # noqa: E402

from rag.chunking import (  # noqa: E402
    build_node_parser,
    count_tokens,
    nodes_from_documents,
    nodes_from_loaded_document,
)
from rag.documents import (  # noqa: E402
    LoadedDocument,
    Section,
    load_document,
    load_manifest,
    strip_echoes,
    to_llamaindex_documents,
)


def section(heading, text, page=1):
    return Section(heading=heading, page_number=page, text=text)


def loaded(sections, manifest=None):
    return LoadedDocument(
        manifest=manifest or load_manifest()[0][0],
        sections=sections,
        content_hash="x",
    )


def embed_text(node):
    return node.get_content(metadata_mode=MetadataMode.EMBED)


class EchoStrippingTests(unittest.TestCase):
    def test_a_line_repeating_the_title_is_dropped(self):
        self.assertEqual(strip_echoes("Colonoscopy", "Colonoscopy", None), "")

    def test_real_content_beside_an_echo_survives(self):
        kept = strip_echoes("CT Scans\nAlso called: CAT scan", "CT Scans", None)

        self.assertEqual(kept, "Also called: CAT scan")

    def test_a_short_complete_answer_is_kept(self):
        # This is the whole answer to benchmark RAG-024. A minimum-size rule
        # would have discarded it.
        text = "You don't need any special preparations for a hearing test."

        kept = strip_echoes(text, "Hearing Tests for Adults", "Will I need to prepare?")

        self.assertEqual(kept, text)


class DocumentConversionTests(unittest.TestCase):
    def test_one_document_per_section(self):
        documents = to_llamaindex_documents(
            loaded([section("One", "First body."), section("Two", "Second body.")])
        )

        self.assertEqual(len(documents), 2)

    def test_a_section_that_is_only_an_echo_produces_no_document(self):
        manifest = load_manifest()[0][0]
        documents = to_llamaindex_documents(
            loaded([section(None, manifest.title)], manifest)
        )

        self.assertEqual(documents, [])

    def test_citation_metadata_is_carried(self):
        manifest = load_manifest()[0][0]
        document = to_llamaindex_documents(
            loaded([section("A heading", "Some body text.", page=3)], manifest)
        )[0]

        self.assertEqual(document.metadata["corpus_document_id"], manifest.document_id)
        self.assertEqual(document.metadata["section"], "A heading")
        self.assertEqual(document.metadata["page_number"], 3)
        self.assertEqual(document.metadata["source_url"], manifest.source_url)

    def test_the_review_date_is_carried_for_citations(self):
        """A citation to health guidance is worth little without a date."""

        manifest = next(
            m for m in load_manifest()[0] if m.document_id == "mri"
        )
        document = to_llamaindex_documents(
            loaded([section("Risks", "MRI does not use ionizing radiation.")], manifest)
        )[0]

        self.assertEqual(document.metadata["last_updated"], "2024-07-15")
        self.assertNotIn(
            "2024-07-15", document.get_content(metadata_mode=MetadataMode.EMBED)
        )

    def test_the_heading_is_embedded_but_bookkeeping_is_not(self):
        document = to_llamaindex_documents(
            loaded([section("How do I prepare?", "Some body text.")])
        )[0]

        embedded = document.get_content(metadata_mode=MetadataMode.EMBED)
        self.assertIn("How do I prepare?", embedded)
        self.assertIn("Some body text.", embedded)
        self.assertNotIn("content_fingerprint", embedded)
        self.assertNotIn("source_url", embedded)


class NodeParserTests(unittest.TestCase):
    def test_a_chunk_size_over_the_model_limit_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            build_node_parser(chunk_size=99_999)

        self.assertIn("silently discarded", str(raised.exception))

    def test_a_short_section_becomes_one_node(self):
        nodes = nodes_from_loaded_document(
            loaded([section("Are there risks?", "There is no risk to this test.")])
        )

        self.assertEqual(len(nodes), 1)
        self.assertIn("There is no risk", nodes[0].text)

    def test_a_long_section_splits(self):
        body = "\n".join(f"Sentence {n} about preparing." for n in range(200))

        nodes = nodes_from_documents(
            to_llamaindex_documents(loaded([section("How do I prepare?", body)])),
            chunk_size=200,
            overlap=30,
        )

        self.assertGreater(len(nodes), 1)
        for node in nodes:
            self.assertLessEqual(count_tokens(embed_text(node)), 200)

    def test_a_node_never_spans_two_sections(self):
        nodes = nodes_from_loaded_document(
            loaded([section("One", "First body."), section("Two", "Second body.")])
        )

        self.assertEqual(len(nodes), 2)
        self.assertNotIn("Second body", nodes[0].text)
        self.assertNotIn("First body", nodes[1].text)

    def test_every_node_keeps_its_section_metadata(self):
        nodes = nodes_from_loaded_document(
            loaded([section("One", "First body."), section("Two", "Second body.")])
        )

        self.assertEqual(
            [node.metadata["section"] for node in nodes], ["One", "Two"]
        )


class CorpusTests(unittest.TestCase):
    def test_the_whole_corpus_parses_within_the_token_budget(self):
        """Both metadata views must fit: one is embedded, one is prompted."""

        for manifest in load_manifest()[0]:
            with self.subTest(manifest.document_id):
                nodes = nodes_from_loaded_document(load_document(manifest))
                self.assertTrue(nodes)
                for node in nodes:
                    self.assertLessEqual(count_tokens(embed_text(node)), 400)
                    self.assertLessEqual(
                        count_tokens(node.get_content(metadata_mode=MetadataMode.LLM)),
                        400,
                    )

    def test_the_colonoscopy_prep_answer_lands_in_one_node(self):
        """A fact split across a node boundary is a retrieval failure waiting."""

        manifest = next(
            m for m in load_manifest()[0] if m.document_id == "colonoscopy"
        )
        nodes = nodes_from_loaded_document(load_document(manifest))

        matches = [node for node in nodes if "red or purple" in node.text]
        self.assertTrue(matches)
        self.assertIn("clear liquid diet", matches[0].text)

    def test_no_node_is_a_bare_title(self):
        """Contentless nodes are near-miss magnets: strong match, no answer."""

        for manifest in load_manifest()[0]:
            with self.subTest(manifest.document_id):
                for node in nodes_from_loaded_document(load_document(manifest)):
                    self.assertNotEqual(node.text.strip(), manifest.title)


if __name__ == "__main__":
    unittest.main()
