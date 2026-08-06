"""Unit tests for the deterministic benchmark metrics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluators.rag.dataset import load_cases  # noqa: E402
from evaluators.rag.deterministic import Report, contains, score_case  # noqa: E402


class FactMatchingTests(unittest.TestCase):
    def test_a_paraphrase_counts_as_stating_the_fact(self):
        """Exact matching measured the checker, not the system."""

        self.assertTrue(
            contains(
                "You may be asked not to eat or drink anything for 4 to 6 hours "
                "before the scan.",
                "may be asked not to eat or drink for 4 to 6 hours before the scan",
            )
        )

    def test_a_wrong_number_is_never_a_paraphrase(self):
        """Quantities are the substance of this corpus, not its wording."""

        self.assertFalse(
            contains(
                "You should fast for 8 to 12 hours before the scan.",
                "not to eat or drink for 4 to 6 hours before the scan",
            )
        )

    def test_a_missing_number_is_not_a_match(self):
        self.assertFalse(
            contains("You may be asked to fast beforehand.", "fast for 4 to 6 hours")
        )

    def test_an_omitted_qualifier_fails_the_fact(self):
        """Dropping 'if you are not at higher risk' changes the clinical claim."""

        fact = "age 45 if you are not at higher risk"
        self.assertTrue(
            contains("If you are not at higher risk, screening starts at age 45.", fact)
        )
        self.assertFalse(contains("Screening starts at age 45.", fact))

    def test_unrelated_text_does_not_match(self):
        self.assertFalse(contains("The test is painless.", "ear plugs reduce the noise"))


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.case = next(c for c in load_cases() if c.question_id == "RAG-001")

    def test_a_forbidden_claim_fails_the_case_outright(self):
        result = score_case(
            self.case, "No fasting is required at all.", "answered", ("mri",)
        )

        self.assertTrue(result.forbidden_hit)
        self.assertFalse(result.passed)

    def test_a_correct_cited_answer_passes(self):
        result = score_case(
            self.case,
            "You may be asked not to eat or drink anything for 4 to 6 hours "
            "before the scan.",
            "answered",
            ("mri",),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.facts_missing, ())


class ReportTests(unittest.TestCase):
    def test_near_miss_resistance_counts_refusals_not_answers(self):
        cases = {c.question_id: c for c in load_cases()}
        report = Report()
        report.results.append(
            score_case(cases["RAG-042"], "I don't have documentation.", "fallback", ())
        )
        report.results.append(
            score_case(cases["RAG-043"], "Clear liquid diet [1].", "answered", ("colonoscopy",))
        )

        metrics = report.metrics()

        self.assertEqual(metrics["near_miss_resistance"], 50.0)
        self.assertEqual(metrics["wrong_document_grounding"], 1)


if __name__ == "__main__":
    unittest.main()
