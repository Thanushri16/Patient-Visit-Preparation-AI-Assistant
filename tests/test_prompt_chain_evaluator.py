"""Unit test for the deterministic prompt-chain evaluation report."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluators.prompt_chain_evaluator import run_prompt_chain_evaluation  # noqa: E402


class PromptChainEvaluatorTests(unittest.TestCase):
    def test_all_baseline_scenarios_pass(self):
        report = run_prompt_chain_evaluation()

        self.assertEqual(report["summary"]["total_scenarios"], 7)
        self.assertEqual(report["summary"]["failed_scenarios"], 0)
        self.assertEqual(report["summary"]["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
