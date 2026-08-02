"""Tests for the combined prompt-chain regression suite."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluators.regression_suite import (  # noqa: E402
    run_prompt_chain_evaluation,
    run_regression_suite,
    write_suite_reports,
)


class PromptChainEvaluationTests(unittest.TestCase):
    """The deterministic half, which must pass without any model calls."""

    def test_all_baseline_scenarios_pass(self):
        report = run_prompt_chain_evaluation()

        self.assertEqual(report["summary"]["failed_scenarios"], 0)
        self.assertEqual(report["summary"]["pass_rate"], 1.0)
        self.assertGreaterEqual(report["summary"]["total_scenarios"], 7)

    def test_it_covers_retraction_and_historical_symptoms(self):
        names = {result["name"] for result in run_prompt_chain_evaluation()["results"]}

        self.assertIn("retraction_clears_field", names)
        self.assertIn("historical_symptom_not_escalated", names)


class SuiteCompositionTests(unittest.TestCase):
    def test_the_suite_runs_without_a_model_client(self):
        report = run_regression_suite(client=None)

        # No API key must still leave a useful, free evaluation to run.
        self.assertIn("prompt_chain", report)
        self.assertNotIn("prompt_injection", report)
        self.assertTrue(report["summary"]["prompt_chain_passed"])

    def test_the_suite_combines_all_three_evaluations(self):
        chain = {
            "summary": {"total_scenarios": 2, "passed_scenarios": 2, "failed_scenarios": 0},
            "results": [],
        }
        injection = {
            "summary": {
                "total_scenarios": 1,
                "average_overall_score": 9.5,
                "high_risk_scenarios": [],
            },
            "results": [],
        }
        intent = {
            "summary": {
                "accuracy": 0.92,
                "overall_precision": 0.9,
                "overall_recall": 0.91,
                "unknown_prediction_rate": 0.08,
            },
            "per_intent_metrics": [],
            "confusion_matrix": {},
            "misclassifications": [],
            "predictions": [],
        }

        with patch(
            "src.evaluators.regression_suite.run_prompt_chain_evaluation", return_value=chain
        ), patch(
            "src.evaluators.regression_suite.run_prompt_injection_evaluation",
            return_value=injection,
        ), patch(
            "src.evaluators.regression_suite.run_intent_classifier_evaluation",
            return_value=intent,
        ):
            report = run_regression_suite(client=object())

        summary = report["summary"]
        self.assertEqual(summary["prompt_chain"], chain["summary"])
        self.assertEqual(summary["prompt_injection"], injection["summary"])
        self.assertEqual(summary["intent_classifier"], intent["summary"])
        self.assertTrue(summary["prompt_chain_passed"])
        self.assertEqual(summary["intent_classifier_accuracy"], 0.92)
        self.assertEqual(summary["prompt_injection_average_score"], 9.5)


class ReportWritingTests(unittest.TestCase):
    def test_writing_produces_json_and_markdown(self):
        report = run_regression_suite(client=None)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            json_path = write_suite_reports(report, output_dir)

            self.assertTrue(json_path.exists())
            json.loads(json_path.read_text(encoding="utf-8"))
            # The suite report and the prompt-chain report it contains.
            self.assertTrue(list(output_dir.glob("regression_suite_report_*.md")))
            self.assertTrue(list(output_dir.glob("prompt_chain_report_*.json")))
