"""Faithful summary rendering and typed confirmation classification workflow."""

import json
import re
from datetime import date
from enum import Enum

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

try:
    from .chatbot_content import APPOINTMENT_SUMMARY_FIELDS, APPOINTMENT_SUMMARY_HEADER, DEFAULT_MODEL
    from .models import (
        ConfirmationAction,
        ConfirmationResult,
        ConversationPhase,
        ConversationState,
        VisitData,
    )
    from .observability import emit_chain_event
    from .prompts.confirmation import build_confirmation_prompt
except ImportError:  # pragma: no cover - allows running as a script
    from chatbot_content import APPOINTMENT_SUMMARY_FIELDS, APPOINTMENT_SUMMARY_HEADER, DEFAULT_MODEL
    from models import (
        ConfirmationAction,
        ConfirmationResult,
        ConversationPhase,
        ConversationState,
        VisitData,
    )
    from observability import emit_chain_event
    from prompts.confirmation import build_confirmation_prompt


CONFIRM_PHRASES = {
    "yes",
    "yes correct",
    "correct",
    "confirm",
    "confirmed",
    "looks good",
    "looks correct",
    "that is correct",
    "that s correct",
    "yes everything looks correct",
    "everything looks correct",
    "yes everything is correct",
    "yes now it s correct",
    "now it s correct",
    "that s complete now",
    "that s everything",
    "all correct",
    "yes that s right",
    "yep",
    "yeah that s right",
}
CORRECTION_SIGNALS = (
    "actually",
    "change ",
    "correct ",
    "update ",
    "wrong",
    "not correct",
    "i never said",
    "i forgot",
    "forgot to mention",
    "should be",
    "instead of",
    "add ",
    "remove ",
    "start over",
    "redo",
)
MAX_CONFIRMATION_ATTEMPTS = 3


def _format_summary_value(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return ", ".join(
            f"{key.replace('_', ' ')}: {_format_summary_value(item)}"
            for key, item in value.items()
        )
    if isinstance(value, list):
        if not value:
            return "None reported"
        return "; ".join(_format_summary_value(item) for item in value)
    return str(value)


# Naming every unanswered field would bury the summary in noise, so the prose
# rendering only flags the handful a visit record is incomplete without. The
# JSON document still reports every gap for programmatic consumers.
KEY_SUMMARY_FIELDS = (
    "patient_name",
    "visit_reason",
    "provider_name",
    "appointment_date",
    "insurance_info",
    "chief_complaint",
    "current_medications",
    "allergies",
)


def build_summary_text(visit_data: VisitData) -> str:
    """Render only canonical VisitData values without asking a model to add prose."""

    lines = [APPOINTMENT_SUMMARY_HEADER, ""]
    provided: list[str] = []
    missing: list[str] = []
    for field_name, label in APPOINTMENT_SUMMARY_FIELDS:
        value = getattr(visit_data, field_name)
        if value is None:
            if field_name in KEY_SUMMARY_FIELDS:
                missing.append(label.lower())
            continue
        lines.append(f"- {label}: {_format_summary_value(value)}")
        provided.append(label)

    if not provided:
        lines.append("- No visit information has been collected yet.")
    if missing:
        lines.extend(["", f"Still to confirm: {', '.join(missing)}."])
    return "\n".join(lines)


def build_summary_document(visit_data: VisitData) -> dict:
    """Return the summary as a schema-complete document.

    Every field in the contract is present. A field the patient has not answered
    is explicitly ``null`` rather than omitted, so a consumer can distinguish
    "not provided" from "not part of the record".
    """

    document: dict[str, object] = {}
    for field_name, _ in APPOINTMENT_SUMMARY_FIELDS:
        value = getattr(visit_data, field_name, None)
        if value is None:
            document[field_name] = None
        elif isinstance(value, BaseModel):
            document[field_name] = value.model_dump(mode="json", exclude_none=True)
        elif isinstance(value, list):
            document[field_name] = [
                item.model_dump(mode="json", exclude_none=True)
                if isinstance(item, BaseModel)
                else item
                for item in value
            ]
        elif isinstance(value, date):
            document[field_name] = value.isoformat()
        else:
            document[field_name] = value
    return {
        "visit_summary": document,
        "missing_fields": [name for name, value in document.items() if value is None],
    }


def build_summary_json(visit_data: VisitData) -> str:
    """Serialise the summary document as the assistant's entire reply."""

    return json.dumps(build_summary_document(visit_data), indent=2, ensure_ascii=False)


def begin_summary_review(state: ConversationState, *, as_json: bool = False) -> str:
    """Store and display the current faithful summary, then await confirmation.

    A summary request renders JSON, which is what downstream consumers validate
    against the schema. Reaching the end of an intake workflow renders the same
    data as prose with a confirmation prompt, because that is a conversation.
    """

    phase_before = state.phase.value
    summary = build_summary_text(state.visit_data)
    state.summary_text = summary
    state.phase = ConversationPhase.AWAITING_CONFIRMATION
    state.confirmed = False
    state.requested_field = None
    emit_chain_event(
        state,
        "summary_renderer",
        success=True,
        phase_before=phase_before,
    )
    if as_json:
        return build_summary_json(state.visit_data)
    return summary + "\n\nIs this summary correct? Reply yes or tell me what to change."


def _normalize_confirmation_text(text: str) -> str:
    return re.sub(r"[^a-z0-9@.]+", " ", text.lower()).strip()


def classify_confirmation(
    client: OpenAI | None,
    latest_message: str,
    displayed_summary: str,
) -> ConfirmationResult:
    """Use deterministic phrases first, then typed model classification if needed."""

    normalized = _normalize_confirmation_text(latest_message)
    if normalized in CONFIRM_PHRASES:
        return ConfirmationResult(action=ConfirmationAction.CONFIRM)

    if any(signal in normalized for signal in CORRECTION_SIGNALS) or (
        normalized.startswith("no ") and len(normalized.split()) > 1
    ):
        return ConfirmationResult(
            action=ConfirmationAction.CORRECT,
            correction_text=latest_message,
        )

    if client is None:
        return ConfirmationResult(action=ConfirmationAction.UNCLEAR)

    prompt = build_confirmation_prompt(latest_message, displayed_summary)
    try:
        response = client.chat.completions.parse(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=ConfirmationResult,
            temperature=0.0,
        )
        parsed = response.choices[0].message.parsed
        if isinstance(parsed, ConfirmationResult):
            return parsed
        return ConfirmationResult.model_validate(parsed)
    except (OpenAIError, ValidationError, AttributeError, IndexError, TypeError):
        return ConfirmationResult(action=ConfirmationAction.UNCLEAR)


def confirmation_response(state: ConversationState, result: ConfirmationResult) -> str:
    """Apply non-correction confirmation transitions and return a safe response."""

    if result.action is ConfirmationAction.CONFIRM:
        state.confirmed = True
        state.phase = ConversationPhase.COMPLETED
        state.confirmation_attempt_count = 0
        return (
            "Your appointment summary is confirmed. "
            "It has not been submitted or sent anywhere."
        )
    state.confirmation_attempt_count += 1
    if state.confirmation_attempt_count >= MAX_CONFIRMATION_ATTEMPTS:
        return (
            "I still couldn't determine whether you confirmed the summary. "
            "It has not been finalized. Reply yes, describe a change, or type menu."
        )
    return "Please reply yes if the summary is correct, or tell me exactly what to change."
