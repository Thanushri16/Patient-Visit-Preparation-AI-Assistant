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
        VisitDataPatch,
    )
    from .observability import emit_chain_event
    from .rendering import render_value
    from .prompts.extractor import EXTRACTOR_PROMPT_VERSION, build_extractor_prompt
    from .questions import select_next_question
    from .workflow_schemas import (
        get_workflow_schema,
        refresh_state_completeness,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from chatbot_content import DEFAULT_MODEL
    from models import (
        ConversationState,
        DomainModel,
        FieldExtractionResult,
        VisitData,
        VisitDataPatch,
    )
    from observability import emit_chain_event
    from rendering import render_value
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
    # Fields whose previous value this turn replaced, so the reply can confirm
    # the change rather than silently overwriting the record.
    corrected_fields: list[str] = Field(default_factory=list)
    # Fields the patient retracted this turn.
    cleared_fields: list[str] = Field(default_factory=list)
    # Fields the extractor proposed that the active workflow does not collect.
    # Recorded for diagnosis; never shown to the user.
    ignored_fields: list[str] = Field(default_factory=list)


class CollectionTurnResult(DomainModel):
    """User-facing result of processing one collection-phase message."""

    response: str
    merge_result: MergeResult


PLACEHOLDER_VALUES = {
    "unknown",
    "not sure",
    "unsure",
    "n/a",
    "na",
    "none specified",
    "not specified",
    "not provided",
    "tbd",
    "unspecified",
}

# A message opening with one of these is adding information, not answering the
# question that was just asked. Passing the pending field as context for "also
# nausea" made the extractor file nausea as the symptom's *location*.
ADDITIVE_OPENING = re.compile(
    r"^\s*(and|also|plus|oh|as well|another|additionally|too\b|i also|"
    r"one more|by the way)\b",
    flags=re.IGNORECASE,
)

# An empty list from the extractor is ambiguous: it can mean "the patient said
# none" or "this message mentioned nothing". Recording the first when the second
# was true wipes real clinical data — a garbled spelling correction once erased
# a patient's whole medication and allergy list. An explicit none is therefore
# only accepted when the message actually contains a denial.
DENIAL_PATTERN = re.compile(
    r"\b(no|none|nothing|not?)\b[^.?!]{0,30}\b(allerg|medication|meds|drugs|"
    r"condition|taking|take)\b"
    r"|\bnkda\b|\bno known\b|\bdon'?t (take|have)\b|\bnot taking\b"
    r"|\bnone\b",
    flags=re.IGNORECASE,
)

NOT_KNOWING_PATTERN = re.compile(
    r"\b(not sure|unsure|don'?t know|do not know|no idea|can'?t remember|"
    r"cannot remember|haven'?t decided|not certain)\b",
    flags=re.IGNORECASE,
)


def drop_unsupported_placeholders(
    extraction: FieldExtractionResult,
    latest_message: str,
) -> FieldExtractionResult:
    """Remove placeholder values the user never actually expressed.

    Asked to record "unknown" when a patient says they do not know something,
    the extractor will sometimes volunteer "unknown" for fields the message
    never mentioned — which writes fabricated data into the record. A
    placeholder is kept only when the message really does express not knowing.
    """

    if NOT_KNOWING_PATTERN.search(latest_message):
        return extraction

    payload = extraction.fields.model_dump(exclude_none=True)
    kept = {
        field_name: value
        for field_name, value in payload.items()
        if not (isinstance(value, str) and value.strip().lower() in PLACEHOLDER_VALUES)
    }
    return FieldExtractionResult(
        fields=VisitDataPatch.model_validate(kept),
        uncertain_fields=extraction.uncertain_fields,
        cleared_fields=extraction.cleared_fields,
        removed_items=extraction.removed_items,
    )


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
    # An additive message is not an answer to the pending question, so the
    # pending field is withheld rather than offered as somewhere to put it.
    requested_field = (
        None if ADDITIVE_OPENING.match(latest_message) else state.requested_field
    )
    prompt = build_extractor_prompt(
        latest_message,
        schema,
        state.visit_data,
        requested_field,
    )
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

    if field_name in {"symptom_duration", "symptom_onset"}:
        if isinstance(value, (int, float)):
            return "Please specify whether that means hours, days, weeks, months, or years."
        if isinstance(value, str):
            normalized = value.strip().lower()
            # "about 3" carries a number but no unit, which is unusable — 3 hours
            # and 3 months are very different clinical pictures.
            has_number = bool(re.search(r"\d", normalized))
            has_unit = bool(
                re.search(
                    r"\b(hour|hr|day|week|wk|month|mo|year|yr)s?\b|"
                    r"\b(today|yesterday|overnight|since)\b",
                    normalized,
                )
            )
            if has_number and not has_unit:
                return "Please specify whether that means hours, days, weeks, months, or years."

    if field_name in {"height", "weight"}:
        unit = getattr(value, "unit", "").strip().lower()
        allowed_units = HEIGHT_UNITS if field_name == "height" else WEIGHT_UNITS
        if unit not in allowed_units:
            return f"Use a supported {field_name} unit."

    if field_name == "current_medications" and isinstance(value, list):
        for item in value:
            dosage = item.get("dosage") if isinstance(item, dict) else getattr(item, "dosage", None)
            # "500 of metformin" is not a dose — 500 mg and 500 mcg differ
            # thousandfold, so a bare number has to be clarified.
            if (
                isinstance(dosage, str)
                and re.search(r"\d", dosage)
                and not re.search(r"[a-z]", dosage, flags=re.IGNORECASE)
            ):
                return "Please include the unit, such as mg, mcg, g, ml, or IU."

    if field_name in {"medical_conditions", "emergency_symptoms", "notes"}:
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            return "Provide non-empty text values."

    return None


# Free-text fields that describe a growing set rather than a single answer. A
# patient listing symptoms over several turns is adding to the picture, not
# replacing it, so "also nausea" must not discard the headache.
ACCUMULATING_TEXT_FIELDS = {"chief_complaint"}

# Wording that means the patient is replacing what they said, not adding to it.
# "Actually it's a migraine" must overwrite the headache; "also nausea" must not.
REPLACEMENT_SIGNAL = re.compile(
    r"\b(actually|instead|rather than|not |wrong|mistake|correction|"
    r"i meant|change|correct)\b",
    flags=re.IGNORECASE,
)


def _merge_accumulating_text(current_value: object, value: object) -> object:
    """Combine a newly-mentioned item with what is already recorded."""

    if not isinstance(current_value, str) or not isinstance(value, str):
        return value
    existing = [part.strip() for part in re.split(r",| and ", current_value) if part.strip()]
    additions = [part.strip() for part in re.split(r",| and ", value) if part.strip()]
    lowered = {part.lower() for part in existing}
    # Anything the model returned that is already recorded is a repeat, and a
    # value that restates the whole list is a rewrite rather than an addition.
    novel = [part for part in additions if part.lower() not in lowered]
    if not novel:
        return current_value
    return ", ".join(existing + novel)


LIST_ITEM_KEY = {
    "current_medications": "name",
    "allergies": "allergen",
}


def _merge_list_value(field_name: str, current_value: object, value: object) -> object:
    """Accumulate list entries across turns instead of replacing them.

    "I take lisinopril" followed by "Also metformin" has to end with both
    medications recorded. Entries are matched on their identifying key so a later
    turn supplying a dose fills in the existing entry rather than duplicating it.
    An explicitly empty list is a deliberate "none", and replaces what is there.
    """

    if not isinstance(current_value, list) or not isinstance(value, list) or not value:
        return value

    key = LIST_ITEM_KEY.get(field_name)
    if key is None:
        merged = list(current_value)
        merged.extend(item for item in value if item not in merged)
        return merged

    def identity(item: object) -> str:
        if isinstance(item, dict):
            return str(item.get(key, "")).strip().lower()
        return str(getattr(item, key, "")).strip().lower()

    merged: list[dict] = [
        item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
        for item in current_value
    ]
    index_by_identity = {identity(item): position for position, item in enumerate(merged)}
    for item in value:
        payload = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
        position = index_by_identity.get(identity(item))
        if position is None:
            index_by_identity[identity(item)] = len(merged)
            merged.append(payload)
        else:
            merged[position] = {**merged[position], **payload}
    return merged


def _without_item(current_value: object, target: str) -> object:
    """Drop one named entry, leaving the rest of the field intact.

    Handles both shapes a multi-entry field takes: a comma-separated complaint
    string and a list of records keyed by name or allergen.
    """

    needle = target.strip().lower()
    if isinstance(current_value, str):
        kept = [
            part.strip()
            for part in re.split(r",| and ", current_value)
            if part.strip() and needle not in part.strip().lower()
        ]
        return ", ".join(kept) if kept else None
    if isinstance(current_value, list):
        kept = [
            item
            for item in current_value
            if needle
            not in str(
                (item.get("name") or item.get("allergen") or item)
                if isinstance(item, dict)
                else item
            ).lower()
        ]
        return kept if len(kept) != len(current_value) else current_value
    return current_value


def _coerce_to_field_type(field_name: str, value: object) -> object:
    """Convert a proposed value into the form state stores it in.

    Returns the value untouched when it cannot be converted, so validation
    downstream still reports the problem rather than this silently swallowing it.
    """

    try:
        return getattr(VisitData.model_validate({field_name: value}), field_name)
    except ValidationError:
        return value


def _without_nones(value: object) -> object:
    """Strip unanswered keys so two shapes of the same value compare equal."""

    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        return {key: _without_nones(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_nones(item) for item in value]
    return value


def _merge_nested_value(
    field_name: str,
    current_value: object,
    value: object,
    latest_message: str = "",
) -> object:
    """Combine a proposed value with what state already holds for that field."""

    # Accumulate only on positive evidence that the turn is additive: a message
    # to read, with no wording that signals a replacement. Without the message
    # there is nothing to judge by, so the later value simply wins.
    if (
        field_name in ACCUMULATING_TEXT_FIELDS
        and current_value
        and latest_message.strip()
        and not REPLACEMENT_SIGNAL.search(latest_message)
    ):
        return _merge_accumulating_text(current_value, value)
    if field_name in LIST_ITEM_KEY or isinstance(current_value, list):
        return _merge_list_value(field_name, current_value, value)
    if field_name not in {"address", "insurance_info"}:
        return value
    if current_value is None or not isinstance(value, dict):
        return value
    return {**current_value.model_dump(exclude_none=True), **value}


def validate_and_merge_extraction(
    state: ConversationState,
    extraction: FieldExtractionResult,
    today: date | None = None,
    latest_message: str = "",
) -> MergeResult:
    """Validate allowed fields and atomically merge each accepted value into state."""

    if state.workflow is None:
        raise ValueError("A workflow must be selected before merging extracted data.")

    validation_date = today or date.today()
    # A workflow governs what the assistant *asks for*, not what a patient is
    # allowed to tell it. There is one visit record, so anything the patient
    # volunteers about it is kept — someone reporting leg pain who mentions
    # their insurance should not have it thrown away for being off-topic.
    allowed_fields = set(VisitData.model_fields)
    proposed_fields = extraction.fields.model_dump(exclude_none=True)
    errors: dict[str, str] = {}
    accepted: dict[str, object] = {}
    corrected: list[str] = []

    # An uncertain field is a gap to ask about, not a failed write.
    uncertain = [name for name in extraction.uncertain_fields if name in allowed_fields]
    # A field outside the active workflow is dropped in silence. Which fields a
    # workflow happens to collect is an internal concern, and asking a patient
    # who just reported an allergy to restate their "symptoms" because the
    # extractor also proposed a chief complaint is nonsense to them.
    ignored = [
        name
        for name in {*proposed_fields, *extraction.uncertain_fields}
        if name not in allowed_fields
    ]

    current_payload = state.visit_data.model_dump()
    # Retractions are applied first so a replacement supplied in the same
    # message lands on a cleared field rather than merging into the old value.
    cleared: list[str] = []
    for field_name in extraction.cleared_fields:
        if field_name in allowed_fields and current_payload.get(field_name) is not None:
            current_payload[field_name] = None
            cleared.append(field_name)
    for entry in extraction.removed_items:
        field_name, _, target = entry.partition(":")
        field_name, target = field_name.strip(), target.strip()
        if not target or field_name not in allowed_fields:
            continue
        remaining = _without_item(current_payload.get(field_name), target)
        if remaining != current_payload.get(field_name):
            current_payload[field_name] = remaining
            cleared.append(field_name)

    if cleared:
        state.visit_data = VisitData.model_validate(current_payload)

    for field_name, value in proposed_fields.items():
        if field_name not in allowed_fields:
            continue

        current_value = getattr(state.visit_data, field_name)

        # Guard the ambiguous empty list before anything else can act on it.
        if value == [] and current_value not in (None, []):
            if not DENIAL_PATTERN.search(latest_message):
                continue

        value = _merge_nested_value(field_name, current_value, value, latest_message)
        # Compare like with like. The patch carries a date of birth as the text
        # the patient wrote while state holds a parsed date, so an unchanged
        # value looked different every turn and was reported as "updated".
        value = _coerce_to_field_type(field_name, value)
        # Compare with unanswered keys stripped from both sides. A merged nested
        # value drops its ``None`` entries while the stored dump keeps them, so a
        # direct comparison never matched and every turn reported having
        # "updated" insurance the user had not mentioned.
        if current_value is not None and _without_nones(current_payload[field_name]) == _without_nones(value):
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
        # The latest statement wins. A patient who says "actually it's Dr. Jones"
        # is correcting the record whether or not the extractor labelled it as a
        # correction, so the overwrite is applied and reported back for confirmation.
        if current_value is not None:
            corrected.append(field_name)
        accepted[field_name] = value

    if accepted:
        state.visit_data = VisitData.model_validate({**current_payload, **accepted})

    state.validation_errors = errors
    if errors:
        state.validation_attempt_count += 1
    else:
        state.validation_attempt_count = 0
    # Whether a second symptom is filed as the complaint or as an associated
    # symptom is a modelling detail; the record should name every symptom in
    # one place either way.
    associated = state.visit_data.associated_symptoms
    if associated:
        combined = _merge_accumulating_text(
            state.visit_data.chief_complaint or "", ", ".join(associated)
        )
        if combined and combined != state.visit_data.chief_complaint:
            state.visit_data.chief_complaint = combined
            if "chief_complaint" not in accepted:
                accepted["chief_complaint"] = combined

    missing_fields = refresh_state_completeness(state)
    # A field the patient tried to answer but answered vaguely — "about 3",
    # "pretty bad" — is the most useful thing to ask about next, ahead of
    # fields they have not touched at all.
    if uncertain:
        prioritized = [name for name in uncertain if name in missing_fields]
        missing_fields = prioritized + [
            name for name in missing_fields if name not in prioritized
        ]
        state.missing_fields = missing_fields

    return MergeResult(
        accepted_fields=list(accepted),
        rejected_fields=list(errors),
        errors=errors,
        missing_fields=missing_fields,
        corrected_fields=corrected,
        cleared_fields=cleared,
        ignored_fields=sorted(ignored),
    )


FIELD_LABELS = {
    "visit_reason": "reason for the visit",
    "provider_name": "provider",
    "appointment_date": "appointment date",
    "appointment_time": "appointment time",
    "visit_type": "visit type",
    "referral_source": "referral",
    "chief_complaint": "symptoms",
    "symptom_location": "location",
    "symptom_onset": "onset",
    "symptom_duration": "duration",
    "symptom_severity": "severity",
    "symptom_pattern": "pattern",
    "current_medications": "medications",
    "medical_conditions": "medical conditions",
    "insurance_info": "insurance",
    "accessibility_needs": "accessibility needs",
    "special_instructions": "pre-visit instructions",
    "documents_status": "documents",
    "fasting_status": "fasting instructions",
    "transportation_needs": "transportation",
    "patient_name": "name",
    "date_of_birth": "date of birth",
}


def describe_field_value(field_name: str, value: object) -> str:
    """Render one recorded field as a short phrase for the acknowledgement line."""

    label = FIELD_LABELS.get(field_name, field_name.replace("_", " "))
    rendered = render_value(field_name, value)
    if rendered is None:
        return label
    # An explicit "none" already reads as a full statement, so labelling it
    # again would produce "allergies: no known drug allergies".
    if isinstance(value, list) and not value:
        return rendered
    if field_name == "symptom_severity":
        return f"severity {rendered}"
    return f"{label}: {rendered}"


def build_acknowledgement(state: ConversationState, result: MergeResult) -> str:
    """Summarise what this turn recorded, so the user can see it was understood."""

    if not result.accepted_fields:
        return ""
    described = [
        describe_field_value(field_name, getattr(state.visit_data, field_name))
        for field_name in result.accepted_fields
    ]
    corrected = [
        FIELD_LABELS.get(field_name, field_name.replace("_", " "))
        for field_name in result.corrected_fields
    ]
    lead = (
        f"I've updated your {', '.join(corrected)}."
        if corrected
        else "Thanks — I've recorded that."
    )
    return f"{lead} So far I have {'; '.join(described)}."


def _build_collection_response(
    state: ConversationState,
    result: MergeResult,
    client: OpenAI | None,
) -> str:
    """Compose the turn's reply: what was captured, then the single next question."""

    if result.errors:
        field_name = result.rejected_fields[0]
        label = FIELD_LABELS.get(field_name, field_name.replace("_", " "))
        question_selection = select_next_question(state, client)
        clarification = (
            question_selection.question
            if question_selection and question_selection.field_path == field_name
            else "Could you give me that detail again?"
        )
        if state.validation_attempt_count >= MAX_VALIDATION_ATTEMPTS:
            return (
                f"I still couldn't record the {label}: {result.errors[field_name]} "
                f"{clarification} You can also type menu or restart if you cannot provide it."
            )
        return f"I need one clarification on the {label}. {result.errors[field_name]} {clarification}"

    acknowledgement = build_acknowledgement(state, result)
    question_selection = select_next_question(state, client)
    if question_selection:
        return f"{acknowledgement} {question_selection.question}".strip()

    state.requested_field = None
    return (acknowledgement or "Thanks — I've recorded that.").strip()


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
        question_selection = select_next_question(state, client)
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

    extraction = drop_unsupported_placeholders(extraction, latest_message)
    state.extraction_retry_count = 0
    emit_chain_event(
        state,
        "field_extractor",
        success=True,
        latency_ms=(perf_counter() - extraction_started) * 1_000,
        prompt_version=EXTRACTOR_PROMPT_VERSION,
    )
    validation_started = perf_counter()
    merge_result = validate_and_merge_extraction(state, extraction, latest_message=latest_message)
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
        response=_build_collection_response(state, merge_result, client),
        merge_result=merge_result,
    )
