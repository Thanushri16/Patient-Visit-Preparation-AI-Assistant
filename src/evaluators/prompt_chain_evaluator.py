"""Deterministic evaluation runner for core prompt-chain node behavior."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from ..extraction import validate_and_merge_extraction
    from ..models import (
        ConfirmationAction,
        ConversationPhase,
        ConversationState,
        FieldExtractionResult,
        VisitData,
        VisitDataPatch,
        WorkflowType,
    )
    from ..moderation import moderate_text
    from ..summary_workflow import build_summary_text, classify_confirmation
    from ..workflow_schemas import refresh_state_completeness
except ImportError:  # pragma: no cover - allows running as a script
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from extraction import validate_and_merge_extraction
    from models import (
        ConfirmationAction,
        ConversationPhase,
        ConversationState,
        FieldExtractionResult,
        VisitData,
        VisitDataPatch,
        WorkflowType,
    )
    from moderation import moderate_text
    from summary_workflow import build_summary_text, classify_confirmation
    from workflow_schemas import refresh_state_completeness


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def _state(workflow: WorkflowType, visit_data: VisitData | None = None) -> ConversationState:
    return ConversationState(
        session_id="evaluation-session",
        workflow=workflow,
        phase=ConversationPhase.COLLECTING,
        visit_data=visit_data or VisitData(),
    )


def run_prompt_chain_evaluation() -> dict[str, Any]:
    """Evaluate extraction handoffs, validation, branching, summary, and safety."""

    results: list[dict[str, Any]] = []

    symptom_state = _state(WorkflowType.REPORT_NEW_SYMPTOMS)
    symptom_state.visit_data = VisitData(
        patient_name="Dana",
        date_of_birth="1984-06-05",
        email="dana@example.com",
        phone="555-0100",
    )
    symptom_merge = validate_and_merge_extraction(
        symptom_state,
        FieldExtractionResult(
            updates=VisitDataPatch(
                chief_complaint="headache",
                symptom_duration="three days",
                symptom_severity=5,
            )
        ),
    )
    results.append(
        {
            "name": "valid_multi_field_extraction",
            "passed": not symptom_merge.errors and not symptom_merge.missing_fields,
        }
    )

    appointment_state = _state(WorkflowType.APPOINTMENT_PREPARATION)
    invalid_email = validate_and_merge_extraction(
        appointment_state,
        FieldExtractionResult(updates=VisitDataPatch(email="invalid")),
    )
    results.append(
        {
            "name": "invalid_email_rejection",
            "passed": "email" in invalid_email.errors and appointment_state.visit_data.email is None,
        }
    )

    correction_state = _state(
        WorkflowType.REPORT_NEW_SYMPTOMS,
        VisitData(
            patient_name="Dana",
            date_of_birth="1984-06-05",
            email="dana@example.com",
            phone="555-0100",
            chief_complaint="headache",
        ),
    )
    validate_and_merge_extraction(
        correction_state,
        FieldExtractionResult(
            corrections=VisitDataPatch(chief_complaint="migraine")
        ),
    )
    results.append(
        {
            "name": "explicit_correction",
            "passed": correction_state.visit_data.chief_complaint == "migraine",
        }
    )

    allergy_state = _state(
        WorkflowType.REPORT_ALLERGY,
        VisitData(
            patient_name="Dana",
            date_of_birth="1984-06-05",
            email="dana@example.com",
            phone="555-0100",
            allergies=[{"allergen": "penicillin"}],
        ),
    )
    refresh_state_completeness(allergy_state)
    results.append(
        {
            "name": "conditional_allergy_reaction",
            "passed": allergy_state.missing_fields == ["allergies.0.reaction"],
        }
    )

    summary = build_summary_text(VisitData(chief_complaint="headache"))
    results.append(
        {
            "name": "summary_faithfulness",
            "passed": "headache" in summary and "diagnos" not in summary.lower(),
        }
    )

    confirmation = classify_confirmation(None, "yes", summary)
    results.append(
        {
            "name": "confirmation_classification",
            "passed": confirmation.action is ConfirmationAction.CONFIRM,
        }
    )

    emergency = moderate_text("I have severe chest pain", stage="input")
    results.append(
        {
            "name": "emergency_escalation",
            "passed": emergency.action == "escalate",
        }
    )

    passed_count = sum(1 for result in results if result["passed"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_scenarios": len(results),
            "passed_scenarios": passed_count,
            "failed_scenarios": len(results) - passed_count,
            "pass_rate": round(passed_count / len(results), 4),
        },
        "results": results,
    }


def write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"prompt_chain_report_{timestamp}.json"
    markdown_path = output_dir / f"prompt_chain_report_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    pass_rate = summary.get("pass_rate")
    if pass_rate is None:
        total = summary["total_scenarios"]
        passed = summary["passed_scenarios"]
        pass_rate = round(passed / total, 4) if total else 0.0
    lines = [
        "# Prompt Chain Evaluation Report",
        "",
        f"- Total scenarios: {summary['total_scenarios']}",
        f"- Passed: {summary['passed_scenarios']}",
        f"- Failed: {summary['failed_scenarios']}",
        f"- Pass rate: {pass_rate}",
        "",
        "## Scenarios",
        "",
    ]
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"- {status}: {result['name']}")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic prompt-chain behavior.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()
    report = run_prompt_chain_evaluation()
    json_path, _ = write_reports(report, args.output_dir)
    print(f"Prompt-chain evaluation complete: {report['summary']}; report={json_path}")
    return 0 if report["summary"]["failed_scenarios"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
