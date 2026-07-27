"""Evaluates the direct model-backed intent classifier against labeled examples."""

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

import sys

try:
    from ..chatbot import classify_intent, load_api_key
    from ..workflow_catalog import INTENT_LABELS
except ImportError:  # pragma: no cover - allows running as a script
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from chatbot import classify_intent, load_api_key
    from workflow_catalog import INTENT_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DEFAULT_EVAL_DATASET = [
    {"text": "Show me the menu.", "expected_intent": "show_menu"},
    {"text": "What can you do for me?", "expected_intent": "show_menu"},
    {"text": "Please show menu options.", "expected_intent": "show_menu"},
    {"text": "I need help with options.", "expected_intent": "show_menu"},
    {"text": "I want to prepare for my appointment.", "expected_intent": "appointment_preparation"},
    {"text": "Start appointment prep now.", "expected_intent": "appointment_preparation"},
    {"text": "Please prepare me for my visit.", "expected_intent": "appointment_preparation"},
    {"text": "I need to report new symptoms.", "expected_intent": "report_new_symptoms"},
    {"text": "I have symptoms to share.", "expected_intent": "report_new_symptoms"},
    {"text": "Symptom report.", "expected_intent": "report_new_symptoms"},
    {"text": "Describe my symptoms please.", "expected_intent": "report_new_symptoms"},
    {"text": "Please review my health notes.", "expected_intent": "review_health_notes"},
    {"text": "Can we review notes?", "expected_intent": "review_health_notes"},
    {"text": "Review my summary.", "expected_intent": "review_health_notes"},
    {"text": "I have a peanut allergy.", "expected_intent": "report_allergy"},
    {"text": "Report an allergy reaction.", "expected_intent": "report_allergy"},
    {"text": "I am allergic to penicillin.", "expected_intent": "report_allergy"},
    {"text": "I have a medication question.", "expected_intent": "medication_question"},
    {"text": "Can I ask about treatment?", "expected_intent": "medication_question"},
    {"text": "Medicine question.", "expected_intent": "medication_question"},
    {"text": "This is an emergency.", "expected_intent": "emergency_support"},
    {"text": "I need urgent help now.", "expected_intent": "emergency_support"},
    {"text": "Should I seek emergency care?", "expected_intent": "emergency_support"},
    {"text": "Show my appointment summary.", "expected_intent": "view_summary"},
    {"text": "View summary.", "expected_intent": "view_summary"},
    {"text": "Visit summary please.", "expected_intent": "view_summary"},
    {"text": "Hello there.", "expected_intent": "unknown"},
    {"text": "Thank you.", "expected_intent": "unknown"},
    {"text": "Can you tell me the weather?", "expected_intent": "unknown"},
    {"text": "", "expected_intent": "unknown"},
]


def _round(value: float) -> float:
    return round(value, 4)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _stable_unique(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def load_dataset(dataset_path: Path) -> list[dict[str, str]]:
    content = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError("Dataset JSON must be a list of objects.")

    records: list[dict[str, str]] = []
    for index, row in enumerate(content, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {index} must be an object.")
        text = row.get("text")
        expected_intent = row.get("expected_intent")
        if not isinstance(text, str):
            raise ValueError(f"Dataset row {index} field 'text' must be a string.")
        if not isinstance(expected_intent, str):
            raise ValueError(f"Dataset row {index} field 'expected_intent' must be a string.")
        records.append({"text": text, "expected_intent": expected_intent})
    return records


def run_intent_classifier_evaluation(
    dataset: list[dict[str, str]],
    client: OpenAI,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in dataset:
        result = classify_intent(item["text"], client)
        predicted_intent = str(result.get("intent", "unknown"))
        is_correct = predicted_intent == item["expected_intent"]
        rows.append(
            {
                "text": item["text"],
                "expected_intent": item["expected_intent"],
                "predicted_intent": predicted_intent,
                "confidence": float(result.get("confidence", 0.0)),
                "status": str(result.get("status", "unknown")),
                "correct": is_correct,
            }
        )

    labels = _stable_unique(
        list(INTENT_LABELS)
        + ["unknown"]
        + [row["expected_intent"] for row in rows]
        + [row["predicted_intent"] for row in rows]
    )

    confusion_matrix: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in labels} for expected in labels
    }
    for row in rows:
        confusion_matrix[row["expected_intent"]][row["predicted_intent"]] += 1

    correct_predictions = sum(1 for row in rows if row["correct"])
    total_samples = len(rows)
    accuracy = _safe_divide(correct_predictions, total_samples)

    per_intent_metrics = []
    macro_labels = [label for label in labels if sum(confusion_matrix[label].values()) > 0]

    for label in labels:
        support = sum(confusion_matrix[label].values())
        true_positives = confusion_matrix[label][label]
        false_positives = sum(
            confusion_matrix[other_label][label]
            for other_label in labels
            if other_label != label
        )
        false_negatives = sum(
            confusion_matrix[label][other_label]
            for other_label in labels
            if other_label != label
        )

        precision = _safe_divide(true_positives, true_positives + false_positives)
        recall = _safe_divide(true_positives, true_positives + false_negatives)
        per_intent_metrics.append(
            {
                "intent": label,
                "support": support,
                "precision": _round(precision),
                "recall": _round(recall),
            }
        )

    macro_precision = _safe_divide(
        sum(metric["precision"] for metric in per_intent_metrics if metric["support"] > 0),
        len(macro_labels),
    )
    macro_recall = _safe_divide(
        sum(metric["recall"] for metric in per_intent_metrics if metric["support"] > 0),
        len(macro_labels),
    )
    unknown_prediction_rate = _safe_divide(
        sum(1 for row in rows if row["predicted_intent"] == "unknown"),
        total_samples,
    )

    return {
        "summary": {
            "mode": "model",
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "accuracy": _round(accuracy),
            "overall_precision": _round(macro_precision),
            "overall_recall": _round(macro_recall),
            "unknown_prediction_rate": _round(unknown_prediction_rate),
        },
        "per_intent_metrics": per_intent_metrics,
        "confusion_matrix": confusion_matrix,
        "misclassifications": [row for row in rows if not row["correct"]],
        "predictions": rows,
    }


def write_reports(report_data: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_at = datetime.datetime.now().isoformat()
    report_data["generated_at"] = generated_at
    json_path = output_dir / f"intent_classifier_report_{timestamp}.json"
    markdown_path = output_dir / f"intent_classifier_report_{timestamp}.md"

    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    summary = report_data["summary"]
    lines = [
        "# Intent Classifier Evaluation Report",
        "",
        f"- Mode: {summary['mode']}",
        f"- Total samples: {summary['total_samples']}",
        f"- Correct predictions: {summary['correct_predictions']}",
        f"- Accuracy: {summary['accuracy']}",
        f"- Overall precision: {summary['overall_precision']}",
        f"- Overall recall: {summary['overall_recall']}",
        f"- Unknown prediction rate: {summary['unknown_prediction_rate']}",
        f"- Generated at: {generated_at}",
        "",
        "## Per-intent Metrics",
    ]

    for metric in report_data["per_intent_metrics"]:
        lines.append(
            f"- {metric['intent']}: support={metric['support']}, precision={metric['precision']}, "
            f"recall={metric['recall']}"
        )

    lines.append("")
    lines.append("## Misclassifications")
    if report_data["misclassifications"]:
        for row in report_data["misclassifications"]:
            lines.append(
                f"- expected={row['expected_intent']} predicted={row['predicted_intent']} "
                f"confidence={row['confidence']} status={row['status']} text={row['text']!r}"
            )
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Confusion Matrix")
    labels = list(report_data["confusion_matrix"].keys())
    header = "| expected \\ predicted | " + " | ".join(labels) + " |"
    separator = "|" + " --- |" * (len(labels) + 1)
    lines.append(header)
    lines.append(separator)
    for expected_label in labels:
        row_counts = [str(report_data["confusion_matrix"][expected_label][predicted]) for predicted in labels]
        lines.append(f"| {expected_label} | " + " | ".join(row_counts) + " |")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate intent classifier performance.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional JSON file containing [{\"text\": ..., \"expected_intent\": ...}].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help=f"Directory for generated reports (default: {REPORTS_DIR}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = DEFAULT_EVAL_DATASET
    if args.dataset is not None:
        try:
            dataset = load_dataset(args.dataset)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Could not load dataset: {exc}")
            return 1

    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        print(f"Could not run model-backed evaluation: {exc}")
        return 1
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    report_data = run_intent_classifier_evaluation(dataset, client=client)
    json_path = write_reports(report_data, args.output_dir)
    summary = report_data["summary"]
    print(
        "Intent classifier evaluation complete. "
        f"accuracy={summary['accuracy']}, precision={summary['overall_precision']}, "
        f"recall={summary['overall_recall']}, "
        f"report={json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
