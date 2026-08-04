"""Unit tests for corpus loading, cleaning and sectioning.

These run against the real PDFs in clinical_docs/ and make no API calls. The
corpus is committed, so its extraction behaviour is as testable as the code.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import MANIFEST_PATH  # noqa: E402
from rag.documents import (  # noqa: E402
    CONTENT_END_MARKERS,
    ManifestError,
    clean_pages,
    load_document,
    load_manifest,
    normalise_typography,
    split_sections,
)


class ManifestTests(unittest.TestCase):
    def test_manifest_matches_the_files_on_disk(self):
        documents, raw = load_manifest()

        self.assertEqual(len(documents), 11)
        self.assertEqual(len(raw["documents"]), 13)

    def test_the_az_index_is_excluded_from_indexing(self):
        documents, raw = load_manifest()

        indexed = {document.document_id for document in documents}
        self.assertNotIn("diagnostic-tests-index", indexed)

        entry = next(
            item for item in raw["documents"]
            if item["document_id"] == "diagnostic-tests-index"
        )
        self.assertFalse(entry["indexed"])
        self.assertIn("no prose", entry["exclusion_reason"])

    def test_the_manifest_declares_no_inert_cleaning_rules(self):
        """The rules are regexes in this module, not data in the manifest.

        An earlier manifest duplicated them, which read as configuration and was
        never loaded: editing it changed nothing. Re-adding them would restore a
        file that silently disagrees with the pipeline.
        """

        _, raw = load_manifest()

        for name, shape in raw["page_shapes"].items():
            with self.subTest(name):
                self.assertNotIn("strip", shape)
                self.assertNotIn("content_ends_at", shape)
                self.assertNotIn("keep_after_cut", shape)

    def test_every_declared_page_shape_has_cleaning_rules_in_code(self):
        _, raw = load_manifest()

        for name in raw["page_shapes"]:
            if name == "index":  # never ingested, so it needs no rules
                continue
            with self.subTest(name):
                self.assertIn(name, CONTENT_END_MARKERS)

    def test_a_page_shape_with_no_rules_in_code_is_rejected(self):
        broken = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        broken["page_shapes"]["invented_shape"] = {"description": "no rules exist"}
        for entry in broken["documents"]:
            if entry.get("indexed"):
                entry["page_shape"] = "invented_shape"
                break

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            path.write_text(yaml.safe_dump(broken), encoding="utf-8")

            with self.assertRaises(ManifestError) as raised:
                load_manifest(path)

        self.assertIn("CONTENT_END_MARKERS", str(raised.exception))

    def test_every_indexed_document_carries_citation_metadata(self):
        documents, _ = load_manifest()

        for document in documents:
            with self.subTest(document.document_id):
                self.assertTrue(document.title)
                self.assertTrue(document.category)
                self.assertTrue(document.source_url)
                self.assertIsNotNone(document.last_updated)
                self.assertTrue(document.path.exists())


class TypographyTests(unittest.TestCase):
    def test_ligatures_are_expanded(self):
        # These PDFs render fl and fi as single glyphs. Left alone, "flexible"
        # never matches its own heading and "specific" tokenises into nonsense.
        self.assertEqual(normalise_typography("ﬂexible"), "flexible")
        self.assertEqual(normalise_typography("speciﬁc"), "specific")

    def test_typographic_punctuation_becomes_ascii(self):
        self.assertEqual(normalise_typography("don’t"), "don't")
        self.assertEqual(normalise_typography("a — b"), "a - b")


class CleaningTests(unittest.TestCase):
    def test_health_topic_link_farm_is_cut(self):
        pages = [
            "How do you prepare for a CT scan?\nYou may be asked not to eat.",
            "Start Here\nComputed Tomography (National Institute)\nCT Scan (Mayo)",
        ]

        cleaned = clean_pages(pages, "health_topic")

        text = " ".join(line for _, line in cleaned)
        self.assertIn("not to eat", text)
        self.assertNotIn("Mayo", text)

    def test_references_end_a_medical_test_page(self):
        pages = ["What is it?\nA lab test.\nReferences\n1. Accu Reference Medical Lab"]

        cleaned = clean_pages(pages, "medical_test")

        text = " ".join(line for _, line in cleaned)
        self.assertIn("A lab test.", text)
        self.assertNotIn("Accu Reference", text)

    def test_boilerplate_and_inline_links_are_removed(self):
        pages = [
            "8/1/26, 1:46 PM How to Prepare for a Lab Test\n"
            "Home → Medical Tests → How to Prepare\n"
            "URL of this page: https://medlineplus.gov/lab-tests/\n"
            "Fasting [https://medlineplus.gov/fasting.html] improves accuracy."
        ]

        cleaned = clean_pages(pages, "medical_test")

        text = " ".join(line for _, line in cleaned)
        self.assertEqual(text, "Fasting improves accuracy.")

    def test_page_numbers_are_preserved_for_citation(self):
        cleaned = clean_pages(["first page text", "second page text"], "medical_test")

        self.assertEqual(cleaned[0][0], 1)
        self.assertEqual(cleaned[-1][0], 2)


class SectioningTests(unittest.TestCase):
    def test_declared_sections_are_matched(self):
        lines = [(1, "What is it used for?"), (1, "To check for allergies.")]

        sections, warnings = split_sections(lines, ("What is it used for?",))

        self.assertEqual(warnings, [])
        self.assertEqual(sections[0].heading, "What is it used for?")
        self.assertEqual(sections[0].text, "To check for allergies.")

    def test_a_heading_wrapped_across_lines_is_matched(self):
        # Every colonoscopy heading names three procedures and wraps in the PDF.
        lines = [
            (2, "How do you prepare for a colonoscopy, virtual colonoscopy, or flexible"),
            (2, "sigmoidoscopy?"),
            (2, "Follow the bowel prep instructions."),
        ]
        declared = (
            "How do you prepare for a colonoscopy, virtual colonoscopy, or flexible sigmoidoscopy?",
        )

        sections, warnings = split_sections(lines, declared)

        self.assertEqual(warnings, [])
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].text, "Follow the bowel prep instructions.")

    def test_a_missing_declared_section_is_reported(self):
        lines = [(1, "Some text with no headings at all.")]

        _, warnings = split_sections(lines, ("What is it used for?",))

        self.assertEqual(len(warnings), 1)
        self.assertIn("What is it used for?", warnings[0])


class CorpusTests(unittest.TestCase):
    """The whole corpus must load cleanly. A warning here is a real defect."""

    def test_every_document_loads_without_warnings(self):
        documents, _ = load_manifest()

        for manifest in documents:
            with self.subTest(manifest.document_id):
                loaded = load_document(manifest)
                self.assertEqual(loaded.warnings, [])
                self.assertTrue(loaded.sections)
                self.assertNotIn("ﬂ", loaded.text)
                self.assertNotIn("ﬁ", loaded.text)

    def test_known_facts_survive_extraction(self):
        """Spot-check the facts the benchmark asserts, at the source."""

        expected = {
            "mri": "4 to 6 hours",
            "ct-scans": "higher than a",
            "colonoscopy": "red or purple",
            "colorectal-cancer-screening": "every 10 years",
            "skin-cancer-screening": "nail polish",
            "hearing-tests-adults": "any special preparations",
            "how-to-prepare-lab-test": "Cholesterol Levels Test",
            "rapid-tests": "20 minutes or less",
        }
        documents = {d.document_id: d for d in load_manifest()[0]}

        for document_id, fragment in expected.items():
            with self.subTest(document_id):
                loaded = load_document(documents[document_id])
                self.assertIn(fragment, loaded.text)

    def test_no_fasting_duration_is_stated_for_a_cholesterol_test(self):
        """Benchmark RAG-044 depends on this gap being real."""

        documents = {d.document_id: d for d in load_manifest()[0]}
        text = load_document(documents["how-to-prepare-lab-test"]).text

        self.assertIn("length of time you need to fast can vary", text)
        self.assertNotIn("8 to 12 hours", text)


if __name__ == "__main__":
    unittest.main()
