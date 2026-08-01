"""Tests for the combined iteration-2 evaluation pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluators.iteration_2_evaluation_pipeline import (
    run_iteration_2_evaluation_pipeline,
    write_reports,
)


class Iteration2EvaluationPipelineTests(unittest.TestCase):
    def test_pipeline_combines_all_iteration_two_reports(self) -> None:
        fake_prompt_chain = {
            "summary": {"total_scenarios": 2, "passed_scenarios": 2, "failed_scenarios": 0},
            "results": [],
        }
        fake_prompt_injection = {
            "summary": {
                "total_scenarios": 1,
                "average_overall_score": 9.5,
                "high_risk_scenarios": [],
            },
            "results": [],
        }
        fake_intent_classifier = {
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
            "src.evaluators.iteration_2_evaluation_pipeline.run_prompt_chain_evaluation",
            return_value=fake_prompt_chain,
        ), patch(
            "src.evaluators.iteration_2_evaluation_pipeline.run_prompt_injection_evaluation",
            return_value=fake_prompt_injection,
        ), patch(
            "src.evaluators.iteration_2_evaluation_pipeline.run_intent_classifier_evaluation",
            return_value=fake_intent_classifier,
        ):
            report = run_iteration_2_evaluation_pipeline(client=object())

        self.assertEqual(report["summary"]["prompt_chain"], fake_prompt_chain["summary"])
        self.assertEqual(report["summary"]["prompt_injection"], fake_prompt_injection["summary"])
        self.assertEqual(report["summary"]["intent_classifier"], fake_intent_classifier["summary"])
        self.assertTrue(report["summary"]["prompt_chain_passed"])
        self.assertEqual(report["summary"]["intent_classifier_accuracy"], 0.92)
        self.assertEqual(report["summary"]["prompt_injection_average_score"], 9.5)

    def test_write_reports_creates_json_and_markdown(self) -> None:
        report = {
            "generated_at": "2026-08-01T12:00:00+00:00",
            "summary": {
                "prompt_chain": {"passed_scenarios": 2, "total_scenarios": 2, "failed_scenarios": 0},
                "prompt_injection": {"total_scenarios": 1, "average_overall_score": 9.5, "high_risk_scenarios": []},
                "intent_classifier": {"accuracy": 0.92, "overall_precision": 0.9, "overall_recall": 0.91, "unknown_prediction_rate": 0.08},
                "prompt_chain_passed": True,
                "intent_classifier_accuracy": 0.92,
                "prompt_injection_average_score": 9.5,
            },
            "prompt_chain": {"summary": {"total_scenarios": 2, "passed_scenarios": 2, "failed_scenarios": 0}, "results": []},
            "prompt_injection": {"summary": {"total_scenarios": 1, "average_overall_score": 9.5, "high_risk_scenarios": []}, "results": []},
            "intent_classifier": {"summary": {"accuracy": 0.92, "overall_precision": 0.9, "overall_recall": 0.91, "unknown_prediction_rate": 0.08}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, markdown_path = write_reports(report, Path(tmpdir))

            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertIn("Iteration 2 Evaluation Report", markdown_path.read_text(encoding="utf-8"))
            self.assertTrue(list(Path(tmpdir).glob("prompt_chain_report_*.json")))
            self.assertTrue(list(Path(tmpdir).glob("prompt_chain_report_*.md")))


if __name__ == "__main__":
    unittest.main()
