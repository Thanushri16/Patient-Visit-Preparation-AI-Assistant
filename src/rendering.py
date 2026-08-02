"""Render recorded visit data as language a patient would actually understand.

Every user-facing surface — the acknowledgement after a turn, the prose summary,
the answer to "what do you have for my insurance?" — goes through here. They used
to render values independently, and each one leaked storage detail in its own
way: the worst was insurance appearing as "bcbs, True, False", where the two
booleans were internal flags for whether the patient has cover and whether they
had their card to hand. Nobody can read that.

The rules are simple and apply everywhere: a boolean is stated as the thing it
means, a nested value is labelled with the name of the part it describes, and a
field that holds nothing says so plainly.
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel


# Booleans are never shown as True/False. Each is the pair of sentences it
# actually stands for, so the record reads as a statement about the patient.
BOOLEAN_PHRASES: dict[str, tuple[str, str]] = {
    "has_insurance": ("insurance on file", "no insurance"),
    "details_available": ("details to hand", "details not to hand"),
    "new_patient": ("first visit with this provider", "returning patient"),
}

# Sub-field names are scoped by their parent, because the same name means
# different things in different places: `provider_name` is the carrier inside
# insurance and the clinician at the top level.
NESTED_LABELS: dict[str, dict[str, str]] = {
    "insurance_info": {
        "provider_name": "carrier",
        "plan_type": "plan",
        "policy_number": "policy number",
        "member_id": "member ID",
        "group_number": "group number",
    },
    "address": {
        "street": "street",
        "city": "city",
        "state": "state",
        "postal_code": "postal code",
    },
}

# A flag is suppressed when the record already shows the opposite.
CONTRADICTED_BY: dict[str, tuple[str, ...]] = {
    "details_available": ("policy_number", "member_id", "group_number", "plan_type"),
}

EMPTY_LIST_PHRASES: dict[str, str] = {
    "allergies": "no known drug allergies (NKDA)",
    "current_medications": "no medications",
    "medical_conditions": "no medical conditions",
    "associated_symptoms": "no associated symptoms",
    "accessibility_needs": "no accessibility needs",
    "special_instructions": "no pre-visit instructions",
}


def _boolean_phrase(field_name: str, value: bool) -> str:
    affirmative, negative = BOOLEAN_PHRASES.get(
        field_name, (f"{field_name.replace('_', ' ')}: yes", f"{field_name.replace('_', ' ')}: no")
    )
    return affirmative if value else negative


def _render_medication(item: object) -> str:
    parts = [
        getattr(item, "name", None),
        getattr(item, "dosage", None),
        getattr(item, "frequency", None),
    ]
    rendered = " ".join(part for part in parts if part)
    purpose = getattr(item, "purpose", None)
    return f"{rendered} for {purpose}" if purpose else rendered


def _render_allergy(item: object) -> str:
    allergen = getattr(item, "allergen", "")
    reaction = getattr(item, "reaction", None)
    severity = getattr(item, "severity", None)
    detail = ", ".join(part for part in (reaction, severity) if part)
    return f"{allergen} ({detail})" if detail else str(allergen)


def _render_measurement(item: object) -> str:
    return f"{getattr(item, 'value', '')}{getattr(item, 'unit', '')}".strip()


def _render_nested(field_name: str, value: BaseModel) -> str:
    """Render a nested record as labelled parts, never as bare values."""

    labels = NESTED_LABELS.get(field_name, {})
    payload = value.model_dump(exclude_none=True)
    # A flag that the record itself contradicts is noise. Saying "details not to
    # hand" directly after listing the details is worse than saying nothing.
    contradicted = {
        key
        for key, contradicting in CONTRADICTED_BY.items()
        if payload.get(key) is False and any(payload.get(name) for name in contradicting)
    }
    # Recording one identifier under both its names is correct storage but reads
    # as two different numbers, so the duplicate is shown once.
    if payload.get("member_id") and payload.get("member_id") == payload.get("policy_number"):
        payload.pop("member_id")
    parts: list[str] = []
    for key, item in payload.items():
        if key in contradicted:
            continue
        if isinstance(item, bool):
            parts.append(_boolean_phrase(key, item))
            continue
        label = labels.get(key, key.replace("_", " "))
        parts.append(f"{label} {item}")
    return ", ".join(parts)


def render_value(field_name: str, value: object) -> str | None:
    """Render one field's value as a readable phrase, or ``None`` if unanswered.

    An empty list is an answer — the patient said "none" — and is rendered as
    that statement rather than as an empty space.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return _boolean_phrase(field_name, value)
    if isinstance(value, list):
        if not value:
            return EMPTY_LIST_PHRASES.get(field_name, f"no {field_name.replace('_', ' ')}")
        rendered: list[str] = []
        for item in value:
            if hasattr(item, "allergen"):
                rendered.append(_render_allergy(item))
            elif hasattr(item, "name"):
                rendered.append(_render_medication(item))
            elif isinstance(item, BaseModel):
                rendered.append(_render_nested(field_name, item))
            else:
                rendered.append(str(item))
        return ", ".join(part for part in rendered if part)
    if isinstance(value, BaseModel):
        if hasattr(value, "unit"):
            return _render_measurement(value)
        return _render_nested(field_name, value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    if field_name == "symptom_severity":
        return f"{value}/10"
    return str(value)
