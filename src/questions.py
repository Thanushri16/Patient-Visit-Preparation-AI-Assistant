"""Question selection for deterministic admin fields and adaptive intake follow-ups."""

from openai import OpenAI, OpenAIError
from pydantic import Field, ValidationError

try:
    from .chatbot_content import DEFAULT_MODEL
    from .models import ConversationState, DomainModel
    from .prompts.followup import build_followup_prompt
    from .workflow_schemas import SHARED_IDENTITY_CONTACT_FIELDS, get_question_for_field, get_workflow_schema
except ImportError:  # pragma: no cover - allows running as a script
    from chatbot_content import DEFAULT_MODEL
    from models import ConversationState, DomainModel
    from prompts.followup import build_followup_prompt
    from workflow_schemas import SHARED_IDENTITY_CONTACT_FIELDS, get_question_for_field, get_workflow_schema


class QuestionSelection(DomainModel):
    """Application-selected field and question for the next collection turn."""

    field_path: str
    question: str
    reason: str


class AdaptiveQuestionResult(DomainModel):
    """Typed output returned by the adaptive follow-up question generator."""

    field_path: str = Field(min_length=1)
    question: str = Field(min_length=1)


NESTED_FIELD_QUESTIONS = {
    "address.street": "What is the street portion of your address?",
    "address.city": "What city is the address in?",
    "address.state": "What state is the address in?",
    "address.postal_code": "What is the postal or ZIP code?",
    "insurance_info.provider_name": "What is the name of your insurance provider?",
    "insurance_info.policy_number": (
        "What is your plan type, member ID, or group number for that insurance?"
    ),
}


def _medication_detail_question(state: ConversationState, field_path: str) -> str:
    """Ask for a named medication's missing dose or frequency."""

    _, raw_index, detail = field_path.split(".")
    try:
        medication = state.visit_data.current_medications[int(raw_index)]
    except (AttributeError, IndexError, TypeError, ValueError):
        return "What dosage and how often do you take it?"

    name = medication.name
    # An unnamed or vaguely described medication needs its name before anything else.
    if not name or name.lower() in VAGUE_MEDICATION_NAMES:
        return "Do you know the name of the medication?"
    if detail == "dosage":
        return f"What dosage of {name} do you take?"
    return f"How often do you typically take {name}?"


VAGUE_MEDICATION_NAMES = {
    "unknown",
    "unspecified",
    "the little blue pill",
    "little blue pill",
    "blue pill",
    "a pill",
}

def _is_administrative_field(field_path: str) -> bool:
    return field_path in SHARED_IDENTITY_CONTACT_FIELDS


def _is_nested_structured_field(field_path: str) -> bool:
    return "." in field_path


def _is_adaptive_field(field_path: str) -> bool:
    return not _is_administrative_field(field_path) and not _is_nested_structured_field(field_path)


# The generator phrases the question and breaks ties between the next couple of
# gaps; it does not get to reorder the collection sequence. Handing it the whole
# missing list let it skip past "where does it hurt" to "when did it start".
ADAPTIVE_CANDIDATE_LIMIT = 2


def _adaptive_field_candidates(state: ConversationState) -> list[str]:
    return [
        field_name
        for field_name in state.missing_fields
        if _is_adaptive_field(field_name)
    ][:ADAPTIVE_CANDIDATE_LIMIT]


def _build_adaptive_question(
    state: ConversationState,
    client: OpenAI | None,
) -> QuestionSelection | None:
    candidates = _adaptive_field_candidates(state)
    if not candidates:
        return None

    if client is None:
        chosen_field = candidates[0]
        return QuestionSelection(
            field_path=chosen_field,
            question=get_question_for_field(state.workflow, chosen_field)
            or "Please provide the next symptom detail.",
            reason="adaptive",
        )

    prompt = build_followup_prompt(
        schema=get_workflow_schema(state.workflow),
        current_data=state.visit_data,
        missing_fields=candidates,
        requested_field=state.requested_field,
    )
    try:
        response = client.chat.completions.parse(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=AdaptiveQuestionResult,
            temperature=0.0,
        )
        parsed = response.choices[0].message.parsed
        if isinstance(parsed, AdaptiveQuestionResult):
            result = parsed
        else:
            result = AdaptiveQuestionResult.model_validate(parsed)
        if result.field_path not in candidates:
            raise ValueError("Adaptive question selected an invalid field.")
        return QuestionSelection(
            field_path=result.field_path,
            question=result.question,
            reason="adaptive",
        )
    except (OpenAIError, ValidationError, AttributeError, IndexError, TypeError, ValueError):
        chosen_field = candidates[0]
        return QuestionSelection(
            field_path=chosen_field,
            question=get_question_for_field(state.workflow, chosen_field)
            or "Please provide the next symptom detail.",
            reason="adaptive",
        )


def _allergy_reaction_question(state: ConversationState, field_path: str) -> str:
    try:
        allergy_index = int(field_path.split(".")[1])
        allergy = state.visit_data.allergies[allergy_index]
    except (AttributeError, IndexError, TypeError, ValueError):
        return "What reaction do you experience with that allergy?"
    return (
        f"What happens when you take {allergy.allergen}? "
        "Also tell me if there are other allergies to add."
    )


def select_next_question(
    state: ConversationState,
    client: OpenAI | None = None,
) -> QuestionSelection | None:
    """Select exactly one question from application-owned missing-field state."""

    if state.workflow is None or not state.missing_fields:
        state.requested_field = None
        return None

    field_path = state.missing_fields[0]
    # Nested detail (a medication's dose, an allergy's reaction) has a precise
    # deterministic question, so only genuinely open top-level gaps are handed
    # to the adaptive generator.
    if _is_adaptive_field(field_path):
        adaptive_selection = _build_adaptive_question(state, client)
        if adaptive_selection is not None:
            state.requested_field = adaptive_selection.field_path
            return adaptive_selection

    if field_path.startswith("allergies.") and field_path.endswith(".reaction"):
        question = _allergy_reaction_question(state, field_path)
        reason = "conditional"
    elif field_path.startswith("current_medications."):
        question = _medication_detail_question(state, field_path)
        reason = "conditional"
    elif field_path in NESTED_FIELD_QUESTIONS:
        question = NESTED_FIELD_QUESTIONS[field_path]
        reason = "conditional"
    else:
        question = get_question_for_field(state.workflow, field_path)
        reason = "required"

    if question is None:
        question = "Please provide the remaining information."

    state.requested_field = field_path
    return QuestionSelection(field_path=field_path, question=question, reason=reason)
