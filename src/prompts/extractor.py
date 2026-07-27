"""Prompt builder for extracting structured visit-data updates and corrections."""

import json

try:
    from ..models import VisitData
    from ..workflow_schemas import WorkflowSchema
except ImportError:  # pragma: no cover
    from models import VisitData
    from workflow_schemas import WorkflowSchema

from .common import HEALTHCARE_ROLE_AND_SAFETY_PROMPT

EXTRACTOR_PROMPT_VERSION = "visit_field_extractor_v1"


def build_extractor_prompt(
    latest_message: str,
    schema: WorkflowSchema,
    current_data: VisitData,
) -> str:
    allowed_fields = schema.required_fields + schema.optional_fields
    context = {
        "workflow": schema.workflow.value,
        "allowed_fields": allowed_fields,
        "current_data": current_data.model_dump(mode="json"),
        "latest_message": latest_message,
    }
    return f"""{HEALTHCARE_ROLE_AND_SAFETY_PROMPT}

Task: Extract only information explicitly stated in the latest message.
Context: {json.dumps(context)}

Do not infer missing values. Do not return fields outside allowed_fields.
Put newly supplied values in updates and replacements of existing values in corrections.
Put ambiguous field names in uncertain_fields.
The structured response schema requires every patch field. Set fields that were not supplied or corrected to null.

Return JSON only:
{{"updates": {{"each patch field": "value or null"}}, "corrections": {{"each patch field": "value or null"}}, "uncertain_fields": []}}
""".strip()
