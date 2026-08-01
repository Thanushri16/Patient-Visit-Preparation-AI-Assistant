"""Run the full iteration-2 evaluation suite for the typed prompt-chain chatbot.

This pipeline combines the deterministic prompt-chain regression checks with the
existing model-backed prompt-injection and intent-classifier evaluations so the
iteration-2 quality checks can be executed from one entry point.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from ..chatbot import load_api_key
    from .intent_classifier_evaluator import DEFAULT_EVAL_DATASET, run_intent_classifier_evaluation
    from .prompt_chain_evaluator import run_prompt_chain_evaluation, write_reports as write_prompt_chain_reports
    from .prompt_injection_evaluator import run_prompt_injection_evaluation
except ImportError:  # pragma: no cover - allows running as a script
    project_root = Path(__file__).resolve().parents[1]
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from chatbot import load_api_key
    from evaluators.intent_classifier_evaluator import DEFAULT_EVAL_DATASET, run_intent_classifier_evaluation
    from evaluators.prompt_chain_evaluator import run_prompt_chain_evaluation, write_reports as write_prompt_chain_reports
    from evaluators.prompt_injection_evaluator import run_prompt_injection_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_iteration_2_evaluation_pipeline(client: OpenAI) -> dict[str, Any]:
    """Run the prompt-chain, prompt-injection, and intent-classifier evaluations."""

    prompt_chain_report = run_prompt_chain_evaluation()
    prompt_injection_report = run_prompt_injection_evaluation(client)
    intent_classifier_report = run_intent_classifier_evaluation(DEFAULT_EVAL_DATASET, client=client)

    return {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "prompt_chain": prompt_chain_report["summary"],
            "prompt_injection": prompt_injection_report["summary"],
            "intent_classifier": intent_classifier_report["summary"],
            "prompt_chain_passed": prompt_chain_report["summary"]["failed_scenarios"] == 0,
            "intent_classifier_accuracy": intent_classifier_report["summary"]["accuracy"],
            "prompt_injection_average_score": prompt_injection_report["summary"]["average_overall_score"],
        },
        "prompt_chain": prompt_chain_report,
        "prompt_injection": prompt_injection_report,
        "intent_classifier": intent_classifier_report,
    }


def write_reports(report_data: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write a combined JSON and Markdown report for the iteration-2 evaluation suite."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"iteration_2_evaluation_report_{timestamp}.json"
    markdown_path = output_dir / f"iteration_2_evaluation_report_{timestamp}.md"
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    write_prompt_chain_reports(report_data["prompt_chain"], output_dir)

    summary = report_data["summary"]
    lines = [
        "# Iteration 2 Evaluation Report",
        "",
        f"- Generated at: {report_data['generated_at']}",
        f"- Prompt-chain scenarios passed: {summary['prompt_chain']['passed_scenarios']}/{summary['prompt_chain']['total_scenarios']}",
        f"- Intent-classifier accuracy: {summary['intent_classifier_accuracy']}",
        f"- Prompt-injection average score: {summary['prompt_injection_average_score']}",
        "",
        "## Prompt Chain",
        f"- Passed scenarios: {summary['prompt_chain']['passed_scenarios']}",
        f"- Failed scenarios: {summary['prompt_chain'].get('failed_scenarios', 0)}",
        "",
        "## Prompt Injection",
        f"- Total scenarios: {summary['prompt_injection']['total_scenarios']}",
        f"- Average overall score: {summary['prompt_injection']['average_overall_score']}",
        f"- High-risk scenarios: {', '.join(summary['prompt_injection']['high_risk_scenarios']) if summary['prompt_injection']['high_risk_scenarios'] else 'None'}",
        "",
        "## Intent Classifier",
        f"- Accuracy: {summary['intent_classifier']['accuracy']}",
        f"- Overall precision: {summary['intent_classifier']['overall_precision']}",
        f"- Overall recall: {summary['intent_classifier']['overall_recall']}",
        f"- Unknown prediction rate: {summary['intent_classifier']['unknown_prediction_rate']}",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full iteration-2 evaluation pipeline.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        print(f"Could not run iteration-2 evaluation pipeline: {exc}")
        return 1

    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    report_data = run_iteration_2_evaluation_pipeline(client)
    json_path, _ = write_reports(report_data, args.output_dir)
    print(f"Iteration 2 evaluation pipeline complete; report={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
