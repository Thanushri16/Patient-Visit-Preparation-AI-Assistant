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
                "The exam should take about 10 to 15 minutes.",
                "the exam should take 10 to 15 minutes",
            )
        )

    def test_a_wrong_number_is_never_a_paraphrase(self):
        """Quantities are the substance of this corpus, not its wording."""

        self.assertFalse(
            contains(
                "The exam takes 20 to 25 minutes.",
                "the exam should take 10 to 15 minutes",
            )
        )

    def test_a_missing_number_is_not_a_match(self):
        self.assertFalse(
            contains("The exam takes a few minutes.", "the exam takes 10 to 15 minutes")
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
        self.case = next(c for c in load_cases() if c.question_id == "RAG-010")

    def test_a_forbidden_claim_fails_the_case_outright(self):
        result = score_case(
            self.case, "A CT scan always takes more than an hour.", "answered", ("ct-scans",)
        )

        self.assertTrue(result.forbidden_hit)
        self.assertFalse(result.passed)

    def test_a_correct_cited_answer_passes(self):
        result = score_case(
            self.case,
            "A CT scan usually takes a few minutes, but some last up to 30 minutes.",
            "answered",
            ("ct-scans",),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.facts_missing, ())


class ReportTests(unittest.TestCase):
    def test_retrieval_answering_a_near_miss_counts_against_resistance(self):
        cases = {c.question_id: c for c in load_cases()}
        report = Report()
        report.results.append(
            score_case(
                cases["RAG-042"], "I don't have documentation.", "fallback", (),
                source="fallback",
            )
        )
        report.results.append(
            score_case(
                cases["RAG-043"], "Clear liquid diet [1].", "answered",
                ("colonoscopy",), source="rag",
            )
        )

        metrics = report.metrics()

        self.assertEqual(metrics["near_miss_resistance"], 50.0)
        self.assertEqual(metrics["wrong_document_grounding"], 1)

    def test_a_curated_answer_to_a_near_miss_is_not_a_leak(self):
        """Reviewed content with no citation is the designed fallback, not a leak.

        This metric and the shadow classifier must agree; they disagreed while
        this one counted any answer rather than a retrieved one.
        """

        cases = {c.question_id: c for c in load_cases()}
        report = Report()
        report.results.append(
            score_case(
                cases["RAG-042"], "Fasting requirements depend on the test.",
                "curated_answer", (), source="curated",
            )
        )

        self.assertEqual(report.metrics()["near_miss_resistance"], 100.0)


class SourceFidelityTests(unittest.TestCase):
    """Answering a real question out of the wrong document.

    The gap this metric exists to close. `wrong_document_grounding` scores the
    near_miss group, whose questions have no answer in the corpus at all, so a
    strategy that answers an *answerable* question from the wrong page scored
    100 on every safety gate. A hybrid retrieval arm did exactly that and was
    only caught by a judged faithfulness score.
    """

    def cases(self):
        return {c.question_id: c for c in load_cases()}

    def test_citing_only_the_expected_document_is_clean(self):
        report = Report()
        report.results.append(
            score_case(
                self.cases()["RAG-004"], "MRI uses no radiation [1].", "answered",
                ("mri",), source="rag",
            )
        )

        self.assertEqual(report.metrics()["source_fidelity"], 100.0)
        self.assertEqual(report.metrics()["off_source_cases"], [])

    def test_citing_a_document_the_question_does_not_belong_to_is_caught(self):
        report = Report()
        report.results.append(
            score_case(
                self.cases()["RAG-017"],
                "Avoid red meat before the test [1].", "answered",
                ("colonoscopy",), source="rag",
            )
        )

        metrics = report.metrics()

        self.assertEqual(metrics["source_fidelity"], 0.0)
        self.assertEqual(
            metrics["off_source_cases"],
            [{"question_id": "RAG-017", "cited": ["colonoscopy"]}],
        )

    def test_a_partly_off_source_answer_still_fails(self):
        """One wrong source is enough: the answer drew on a page it should not have."""

        report = Report()
        report.results.append(
            score_case(
                self.cases()["RAG-013"], "You may not remember it [1][2].",
                "answered", ("colonoscopy", "endoscopy"), source="rag",
            )
        )

        self.assertEqual(report.metrics()["source_fidelity"], 0.0)

    def test_the_old_gates_do_not_see_this_failure(self):
        """The regression test for the blind spot itself.

        If this ever starts failing because near_miss_resistance moved, the two
        metrics have been conflated and this one has stopped being independent.
        """

        report = Report()
        report.results.append(
            score_case(
                self.cases()["RAG-017"],
                "Avoid red meat before the test [1].", "answered",
                ("colonoscopy",), source="rag",
            )
        )

        metrics = report.metrics()

        self.assertEqual(metrics["near_miss_resistance"], 0.0)  # no near_miss cases
        self.assertEqual(metrics["wrong_document_grounding"], 0)
        self.assertEqual(metrics["source_fidelity"], 0.0)

    def test_a_case_without_the_annotation_is_excluded_not_passed(self):
        """Silence in the dataset must not read as a clean result."""

        report = Report()
        report.results.append(
            score_case(
                self.cases()["RAG-042"], "Some answer [1].", "answered",
                ("mri",), source="rag",
            )
        )

        # RAG-042 is a near_miss case with no expected documents, so it is not
        # scored here at all -- it is caught by near-miss resistance instead.
        # The rate reads 100 because nothing failed, and the count says why.
        metrics = report.metrics()
        self.assertEqual(metrics["source_fidelity"], 100.0)
        self.assertEqual(metrics["source_fidelity_cases"], 0)
        self.assertEqual(metrics["off_source_cases"], [])

    def test_an_empty_run_does_not_block_promotion(self):
        """A vacuous 100 must not become a spurious gate failure."""

        self.assertEqual(Report().metrics()["source_fidelity"], 100.0)
        self.assertEqual(Report().metrics()["source_fidelity_cases"], 0)

    def test_a_refusal_is_not_scored(self):
        report = Report()
        report.results.append(
            score_case(
                self.cases()["RAG-004"], "I don't have documentation.", "fallback",
                (), source="fallback",
            )
        )

        self.assertEqual(report.metrics()["off_source_cases"], [])


if __name__ == "__main__":
    unittest.main()


class ShadowClassificationTests(unittest.TestCase):
    """The near-miss gate must fire on retrieval, not on any answer at all."""

    def _classify(self, **overrides):
        from evaluators.rag.shadow import classify

        defaults = dict(
            question_id="RAG-038",
            group="near_miss",
            expected_outcome="fallback",
            curated_text="Fasting requirements depend on the specific test.",
            rag_answered=True,
            rag_cited=False,
            gap_disclosed=None,
            answer_source="curated",
        )
        return classify(**{**defaults, **overrides})

    def test_a_curated_fallback_on_a_near_miss_is_not_a_leak(self):
        """Retrieval found nothing and curated content answered: the ladder working."""

        from evaluators.rag.shadow import Divergence

        observation = self._classify()

        self.assertIs(observation.classification, Divergence.AGREEMENT)
        self.assertFalse(observation.blocks_promotion)

    def test_retrieval_answering_a_near_miss_still_blocks(self):
        from evaluators.rag.shadow import Divergence

        observation = self._classify(answer_source="rag", rag_cited=True)

        self.assertIs(observation.classification, Divergence.NEAR_MISS_ANSWERED)
        self.assertTrue(observation.blocks_promotion)

    def test_a_correct_refusal_is_still_agreement(self):
        from evaluators.rag.shadow import Divergence

        observation = self._classify(
            group="never_route",
            expected_outcome="curated_refusal",
            rag_answered=False,
            answer_source="policy",
        )

        self.assertIs(observation.classification, Divergence.AGREEMENT)
