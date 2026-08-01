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
- Only choose from missing_fields.
- Prefer the most relevant missing or ambiguous top-level field after administrative fields are complete.
- Do not ask about identity/contact fields unless they are truly missing.
- Do not guess. Ask one concise question.
- Return a direct question that helps collect exactly one missing field.

Return JSON only:
{{"field_path": "one missing field", "question": "one follow-up question"}}
""".strip()
