"""Prompt builder for adaptive follow-up question generation."""

import json

try:
    from ..models import VisitData
    from ..workflow_schemas import WorkflowSchema
except ImportError:  # pragma: no cover
    from models import VisitData
    from workflow_schemas import WorkflowSchema

from .common import HEALTHCARE_ROLE_AND_SAFETY_PROMPT

FOLLOWUP_PROMPT_VERSION = "adaptive_followup_v1"


def build_followup_prompt(
    schema: WorkflowSchema,
    current_data: VisitData,
    missing_fields: list[str],
    requested_field: str | None = None,
) -> str:
    context = {
        "workflow": schema.workflow.value,
        "current_data": current_data.model_dump(mode="json"),
        "missing_fields": missing_fields,
        "requested_field": requested_field,
    }
    return f"""{HEALTHCARE_ROLE_AND_SAFETY_PROMPT}

Task: Choose exactly one next follow-up question for this patient intake turn.
Context: {json.dumps(context)}

Rules:
- Only choose from missing_fields, which is already in priority order. Take the
  first one unless current_data shows it is effectively answered, in which case
  take the next.
- Choose the gap a clinician would most want filled next given what is already
  in current_data. If the patient described where it hurts, ask when it started,
  not where. If they gave a vague intensity like "pretty bad", ask for a number
  from 0 to 10. If they gave a number with no unit, ask whether it means days,
  weeks, or months.
- Name the patient's own symptom in the question rather than saying "the
  symptom", and refer to each of them when several were reported.
- Closely related detail may be gathered in one sentence — when did it start and
  how often does it happen — but ask about only one field.
- Do not ask about identity or contact details while clinical detail is missing.
- Do not guess, diagnose, or suggest what the answer might be.

Return JSON only:
{{"field_path": "one missing field", "question": "one follow-up question"}}
""".strip()
