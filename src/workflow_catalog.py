"""Single source of truth for workflow menu and intent-classification metadata."""

from dataclasses import dataclass

try:
    from .models import WorkflowType
except ImportError:  # pragma: no cover - allows running as a script
    from models import WorkflowType


SHOW_MENU_INTENT = "show_menu"
SHOW_MENU_COMMANDS = (
    "menu",
    "help",
    "options",
    "show menu",
    "show me the menu",
    "what can you help with",
    "what can you do",
)
SHOW_MENU_EXAMPLE = "Show me the menu."
INTENT_CONFIDENCE_THRESHOLD = 0.2


@dataclass(frozen=True)
class WorkflowDefinition:
    """Navigation and classification metadata for one workflow."""

    workflow: WorkflowType
    menu_option: str
    menu_label: str
    start_response: str | None
    few_shot_example: str


WORKFLOW_DEFINITIONS = (
    WorkflowDefinition(
        workflow=WorkflowType.APPOINTMENT_PREPARATION,
        menu_option="1",
        menu_label="Appointment Preparation",
        start_response="Let’s start preparing for your appointment. What brings you in today?",
        few_shot_example="I want to prepare for my appointment.",
    ),
    WorkflowDefinition(
        workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
        menu_option="2",
        menu_label="Report New Symptoms",
        start_response=(
            "I can help you describe your symptoms. What symptoms are you experiencing today?"
        ),
        few_shot_example="I need to report new symptoms.",
    ),
    WorkflowDefinition(
        workflow=WorkflowType.REVIEW_HEALTH_NOTES,
        menu_option="3",
        menu_label="Review My Health Notes",
        start_response=(
            "I can help review your health notes. First, I need to confirm your name, date of birth, email, and phone."
        ),
        few_shot_example="Please review my health notes.",
    ),
    WorkflowDefinition(
        workflow=WorkflowType.REPORT_ALLERGY,
        menu_option="4",
        menu_label="Report an Allergy or Reaction",
        start_response=(
            "I can help capture allergy or reaction information. Please tell me what "
            "you are allergic to and whether you have had any recent reactions."
        ),
        few_shot_example="I have an allergy to penicillin.",
    ),
    WorkflowDefinition(
        workflow=WorkflowType.MEDICATION_QUESTION,
        menu_option="5",
        menu_label="Ask About Medication or Treatment",
        start_response=(
            "I can help with general medication or treatment questions. I can gather "
            "information about your current medications and discuss general education, "
            "but I cannot diagnose or prescribe treatment."
        ),
        few_shot_example="I want to ask about my medication.",
    ),
    WorkflowDefinition(
        workflow=WorkflowType.EMERGENCY_SUPPORT,
        menu_option="6",
        menu_label="Emergency Support",
        start_response=None,
        few_shot_example="This feels like an emergency.",
    ),
    WorkflowDefinition(
        workflow=WorkflowType.VIEW_SUMMARY,
        menu_option="7",
        menu_label="View Appointment Summary",
        start_response=(
            "I can show your appointment summary. First, I need to confirm your name, date of birth, email, and phone."
        ),
        few_shot_example="Show me my appointment summary.",
    ),
)

WORKFLOW_CATALOG = {
    definition.workflow: definition for definition in WORKFLOW_DEFINITIONS
}
MENU_OPTION_TO_WORKFLOW = {
    definition.menu_option: definition.workflow for definition in WORKFLOW_DEFINITIONS
}
INTENT_LABELS = (
    SHOW_MENU_INTENT,
    *(definition.workflow.value for definition in WORKFLOW_DEFINITIONS),
)
FEW_SHOT_EXAMPLES = (
    {"text": SHOW_MENU_EXAMPLE, "intent": SHOW_MENU_INTENT},
    *(
        {
            "text": definition.few_shot_example,
            "intent": definition.workflow.value,
        }
        for definition in WORKFLOW_DEFINITIONS
    ),
)

MENU_PROMPT_RESPONSE = "\n".join(
    [
        "Here are some healthcare-focused options for your visit preparation:",
        *[
            f"{definition.menu_option}. {definition.menu_label}"
            for definition in WORKFLOW_DEFINITIONS
        ],
        "",
        "Reply with the option number or the option name.",
    ]
)


def build_intent_classifier_prompt(latest_message: str) -> str:
    """Build the classifier prompt from catalog-generated labels and examples."""

    lines = [
        "You are an intent classifier for a healthcare appointment preparation chatbot.",
        "Classify the user message into one of these intents:",
        f"{', '.join(INTENT_LABELS)}.",
        "Respond with only the intent label.",
        "",
        "Examples:",
    ]
    for example in FEW_SHOT_EXAMPLES:
        lines.extend(
            [
                f"User: {example['text']}",
                f"Intent: {example['intent']}",
                "",
            ]
        )
    lines.extend([f"User: {latest_message}", "Intent:"])
    return "\n".join(lines)
