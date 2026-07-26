"""Deterministic next-question selection for required and conditional fields."""

try:
    from .models import ConversationState, DomainModel
    from .workflow_schemas import get_question_for_field
except ImportError:  # pragma: no cover - allows running as a script
    from models import ConversationState, DomainModel
    from workflow_schemas import get_question_for_field


class QuestionSelection(DomainModel):
    """Application-selected field and question for the next collection turn."""

    field_path: str
    question: str
    reason: str


NESTED_FIELD_QUESTIONS = {
    "address.street": "What is the street portion of your address?",
    "address.city": "What city is the address in?",
    "address.state": "What state is the address in?",
    "address.postal_code": "What is the postal or ZIP code?",
    "insurance_info.provider_name": "What is the name of your insurance provider?",
    "insurance_info.policy_number": "What is your insurance policy number?",
}


def _allergy_reaction_question(state: ConversationState, field_path: str) -> str:
    try:
        allergy_index = int(field_path.split(".")[1])
        allergy = state.visit_data.allergies[allergy_index]
    except (AttributeError, IndexError, TypeError, ValueError):
        return "What reaction do you experience with that allergy?"
    return f"What reaction do you experience with {allergy.allergen}?"


def select_next_question(state: ConversationState) -> QuestionSelection | None:
    """Select exactly one question from application-owned missing-field state."""

    if state.workflow is None or not state.missing_fields:
        state.requested_field = None
        return None

    field_path = state.missing_fields[0]
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
