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
    "insurance_info.policy_number": "What is your insurance policy number?",
}

def _is_administrative_field(field_path: str) -> bool:
    return field_path in SHARED_IDENTITY_CONTACT_FIELDS


def _is_nested_structured_field(field_path: str) -> bool:
    return "." in field_path


def _is_adaptive_field(field_path: str) -> bool:
    return not _is_administrative_field(field_path) and not _is_nested_structured_field(field_path)


def _adaptive_field_candidates(state: ConversationState) -> list[str]:
    return [
        field_name
        for field_name in state.missing_fields
        if _is_adaptive_field(field_name)
    ]


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
    return f"What reaction do you experience with {allergy.allergen}?"


def select_next_question(
    state: ConversationState,
    client: OpenAI | None = None,
) -> QuestionSelection | None:
    """Select exactly one question from application-owned missing-field state."""

    if state.workflow is None or not state.missing_fields:
        state.requested_field = None
        return None

    field_path = state.missing_fields[0]
    if not _is_administrative_field(field_path):
        adaptive_selection = _build_adaptive_question(state, client)
        if adaptive_selection is not None:
            state.requested_field = adaptive_selection.field_path
            return adaptive_selection

    if field_path.startswith("allergies.") and field_path.endswith(".reaction"):
        question = _allergy_reaction_question(state, field_path)
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
