"""Structured field extraction, deterministic validation, and safe state merging."""

import re
from datetime import date
from time import perf_counter

from openai import OpenAI, OpenAIError
from pydantic import Field, ValidationError

try:
    from .chatbot_content import DEFAULT_MODEL
    from .models import (
        ConversationState,
        DomainModel,
        FieldExtractionResult,
        VisitData,
    )
    from .observability import emit_chain_event
    from .prompts.extractor import EXTRACTOR_PROMPT_VERSION, build_extractor_prompt
    from .questions import select_next_question
    from .workflow_schemas import (
        get_workflow_schema,
        refresh_state_completeness,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from chatbot_content import DEFAULT_MODEL
    from models import ConversationState, DomainModel, FieldExtractionResult, VisitData
    from observability import emit_chain_event
    from prompts.extractor import EXTRACTOR_PROMPT_VERSION, build_extractor_prompt
    from questions import select_next_question
    from workflow_schemas import (
        get_workflow_schema,
        refresh_state_completeness,
    )


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
HEIGHT_UNITS = {"cm", "m", "in", "inch", "inches", "ft", "foot", "feet"}
WEIGHT_UNITS = {"kg", "g", "lb", "lbs", "pound", "pounds"}
MAX_EXTRACTION_RETRIES = 2
MAX_VALIDATION_ATTEMPTS = 3


class ExtractionError(RuntimeError):
    """Raised when the model does not return a usable structured extraction."""


class MergeResult(DomainModel):
    """Result of validating and merging one structured extraction."""

    accepted_fields: list[str] = Field(default_factory=list)
    rejected_fields: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)


class CollectionTurnResult(DomainModel):
    """User-facing result of processing one collection-phase message."""

    response: str
    merge_result: MergeResult


def extract_structured_fields(
    client: OpenAI | None,
    state: ConversationState,
    latest_message: str,
) -> FieldExtractionResult:
    """Call the extractor prompt and require a parsed Pydantic response."""

    if client is None:
        raise ExtractionError("The extraction service is unavailable.")
    if state.workflow is None:
        raise ExtractionError("A workflow must be selected before extraction.")

    schema = get_workflow_schema(state.workflow)
    prompt = build_extractor_prompt(latest_message, schema, state.visit_data)
    response = client.chat.completions.parse(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=FieldExtractionResult,
        temperature=0.0,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ExtractionError("The extractor did not return structured data.")
    if isinstance(parsed, FieldExtractionResult):
        return parsed
    try:
        return FieldExtractionResult.model_validate(parsed)
    except ValidationError as exc:
        raise ExtractionError("The extractor returned invalid structured data.") from exc


def _validate_field_value(field_name: str, value: object, today: date) -> str | None:
    if isinstance(value, str) and not value.strip():
        return "The value cannot be empty."

    if field_name == "email" and (
        not isinstance(value, str) or not EMAIL_PATTERN.fullmatch(value)
    ):
        return "Enter a valid email address."

    if field_name == "phone":
        digits = re.sub(r"\D", "", str(value))
        if not 7 <= len(digits) <= 15:
            return "Enter a phone number containing 7 to 15 digits."

    if field_name == "date_of_birth" and isinstance(value, date) and value >= today:
        return "Date of birth must be earlier than today."

    if field_name in {"height", "weight"}:
        unit = getattr(value, "unit", "").strip().lower()
        allowed_units = HEIGHT_UNITS if field_name == "height" else WEIGHT_UNITS
        if unit not in allowed_units:
            return f"Use a supported {field_name} unit."

    if field_name in {"medical_conditions", "emergency_symptoms", "notes"}:
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            return "Provide non-empty text values."

    return None


def _merge_nested_value(field_name: str, current_value: object, value: object) -> object:
    if field_name not in {"address", "insurance_info"}:
        return value
    if current_value is None or not isinstance(value, dict):
        return value
    return {**current_value.model_dump(exclude_none=True), **value}


def validate_and_merge_extraction(
    state: ConversationState,
    extraction: FieldExtractionResult,
    today: date | None = None,
) -> MergeResult:
    """Validate allowed fields and atomically merge each accepted value into state."""

    if state.workflow is None:
        raise ValueError("A workflow must be selected before merging extracted data.")

    validation_date = today or date.today()
    schema = get_workflow_schema(state.workflow)
    allowed_fields = set(schema.required_fields) | set(schema.optional_fields)
    updates = extraction.updates.model_dump(exclude_none=True)
    corrections = extraction.corrections.model_dump(exclude_none=True)
    proposed_fields = {**updates, **corrections}
    errors: dict[str, str] = {}
    accepted: dict[str, object] = {}

    for field_name in extraction.uncertain_fields:
        if field_name in allowed_fields:
            errors[field_name] = "Please clarify this value."
        else:
            errors[field_name] = "This field is not collected by the active workflow."

    current_payload = state.visit_data.model_dump()
    for field_name, value in proposed_fields.items():
        if field_name not in allowed_fields:
            errors[field_name] = "This field is not collected by the active workflow."
            continue

        current_value = getattr(state.visit_data, field_name)
        value = _merge_nested_value(field_name, current_value, value)
        if field_name in updates and current_value is not None and field_name not in corrections:
            if current_payload[field_name] == value:
                continue
            errors[field_name] = "Use an explicit correction to replace the existing value."
            continue

        if field_error := _validate_field_value(field_name, value, validation_date):
            errors[field_name] = field_error
            continue

        candidate_payload = {**current_payload, **accepted, field_name: value}
        try:
            VisitData.model_validate(candidate_payload)
        except ValidationError:
            errors[field_name] = "The value does not match the required format."
            continue
        accepted[field_name] = value

    if accepted:
        state.visit_data = VisitData.model_validate({**current_payload, **accepted})

    state.validation_errors = errors
    if errors:
        state.validation_attempt_count += 1
    else:
        state.validation_attempt_count = 0
    missing_fields = refresh_state_completeness(state)

    return MergeResult(
        accepted_fields=list(accepted),
        rejected_fields=list(errors),
        errors=errors,
        missing_fields=missing_fields,
    )


def _build_collection_response(state: ConversationState, result: MergeResult) -> str:
    if result.errors:
        field_name = result.rejected_fields[0]
        label = field_name.replace("_", " ")
        question_selection = select_next_question(state)
        clarification = (
            question_selection.question
            if question_selection and question_selection.field_path == field_name
            else "Please provide that information again."
        )
        if state.validation_attempt_count >= MAX_VALIDATION_ATTEMPTS:
            return (
                f"I still couldn't validate {label}: {result.errors[field_name]} "
                f"{clarification} You can also type menu or restart if you cannot provide it."
            )
        return f"I couldn't record {label}: {result.errors[field_name]} {clarification}"

    question_selection = select_next_question(state)
    if question_selection:
        prefix = "Thanks, I've recorded that. " if result.accepted_fields else ""
        return prefix + question_selection.question

    if not result.missing_fields:
        state.requested_field = None
        return "Thanks, I have collected all required information for this workflow."

    return "Please provide the next requested detail."


def process_collection_turn(
    client: OpenAI | None,
    state: ConversationState,
    latest_message: str,
) -> CollectionTurnResult:
    """Extract, validate, merge, and create the next deterministic response."""

    extraction_started = perf_counter()
    try:
        extraction = extract_structured_fields(client, state, latest_message)
    except (
        ExtractionError,
        OpenAIError,
        ValidationError,
        AttributeError,
        IndexError,
        TypeError,
    ) as exc:
        state.extraction_retry_count += 1
        emit_chain_event(
            state,
            "field_extractor",
            success=False,
            latency_ms=(perf_counter() - extraction_started) * 1_000,
            prompt_version=EXTRACTOR_PROMPT_VERSION,
            retry_count=state.extraction_retry_count,
            error_category=type(exc).__name__,
        )
        missing_fields = refresh_state_completeness(state)
        question_selection = select_next_question(state)
        direct_question = (
            f" {question_selection.question}" if question_selection is not None else ""
        )
        if state.extraction_retry_count >= MAX_EXTRACTION_RETRIES:
            response = (
                "I still couldn't reliably capture that information."
                + direct_question
                + " You can also type menu or restart."
            )
        else:
            response = (
                "I couldn't reliably capture that information. "
                "Please restate it in a short, direct sentence."
                + direct_question
            )
        return CollectionTurnResult(
            response=response,
            merge_result=MergeResult(missing_fields=missing_fields),
        )

    state.extraction_retry_count = 0
    emit_chain_event(
        state,
        "field_extractor",
        success=True,
        latency_ms=(perf_counter() - extraction_started) * 1_000,
        prompt_version=EXTRACTOR_PROMPT_VERSION,
    )
    validation_started = perf_counter()
    merge_result = validate_and_merge_extraction(state, extraction)
    emit_chain_event(
        state,
        "validation_merge",
        success=not merge_result.errors,
        latency_ms=(perf_counter() - validation_started) * 1_000,
        retry_count=state.validation_attempt_count,
        error_category="validation_error" if merge_result.errors else None,
        metadata={
            "accepted_field_count": len(merge_result.accepted_fields),
            "rejected_field_count": len(merge_result.rejected_fields),
            "missing_field_count": len(merge_result.missing_fields),
        },
    )
    return CollectionTurnResult(
        response=_build_collection_response(state, merge_result),
        merge_result=merge_result,
    )
