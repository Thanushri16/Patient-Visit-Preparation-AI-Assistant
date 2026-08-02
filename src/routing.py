"""State-aware workflow routing and deterministic global-command transitions.

This module decides whether a user turn should:
- start a workflow from the menu or intent classifier
- handle a global command like restart, cancel, or go back
- continue the current workflow without rerouting
- keep emergency or completed conversations in their terminal state
"""

import re
from collections.abc import Callable
from enum import StrEnum

from pydantic import Field

try:
    from .chatbot_content import EMERGENCY_ESCALATION_RESPONSE
    from .guidance import looks_like_visit_information
    from .models import (
        ConversationPhase,
        ConversationState,
        DomainModel,
        VisitData,
        WorkflowType,
    )
    from .workflow_schemas import refresh_state_completeness
    from .workflow_catalog import (
        INTENT_CONFIDENCE_THRESHOLD,
        MENU_OPTION_TO_WORKFLOW,
        MENU_PROMPT_RESPONSE,
        SHOW_MENU_INTENT,
        SHOW_MENU_COMMANDS,
        WORKFLOW_CATALOG,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from chatbot_content import EMERGENCY_ESCALATION_RESPONSE
    from guidance import looks_like_visit_information
    from models import ConversationPhase, ConversationState, DomainModel, VisitData, WorkflowType
    from workflow_schemas import refresh_state_completeness
    from workflow_catalog import (
        INTENT_CONFIDENCE_THRESHOLD,
        MENU_OPTION_TO_WORKFLOW,
        MENU_PROMPT_RESPONSE,
        SHOW_MENU_INTENT,
        SHOW_MENU_COMMANDS,
        WORKFLOW_CATALOG,
    )


ROUTER_VERSION = "state_router_v1"


class RouteAction(StrEnum):
    CONTINUE = "continue"
    START_WORKFLOW = "start_workflow"
    SHOW_MENU = "show_menu"
    RESTART = "restart"
    CANCEL = "cancel"
    GO_BACK = "go_back"
    CHANGE_ANSWER = "change_answer"
    FALLBACK = "fallback"


class RouteDecision(DomainModel):
    """Result of routing one message and applying any state transition."""

    action: RouteAction
    handled: bool
    workflow: WorkflowType | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    source: str
    response: str | None = None
    # True when the routed message still carries content the workflow should
    # extract. Picking a numbered menu option conveys nothing to extract, but
    # "I've had a headache for three days" both selects the workflow and answers
    # its first questions — dropping it would make the bot re-ask what it was
    # just told.
    collect_message: bool = False


GLOBAL_COMMANDS = {
    RouteAction.SHOW_MENU: set(SHOW_MENU_COMMANDS),
    RouteAction.RESTART: {"restart", "start over", "restart conversation"},
    RouteAction.CANCEL: {"cancel", "cancel workflow", "stop this workflow"},
    RouteAction.GO_BACK: {"go back", "back", "previous"},
    RouteAction.CHANGE_ANSWER: {"change answer", "change my answer", "correct an answer"},
}


# A request to look at the record, as opposed to add to it.
SUMMARY_REQUEST_PATTERN = re.compile(
    r"\b(show|see|view|display|generate|check|validate)\b[^.?!]{0,40}\b"
    r"(summary|what you have|what i(?:'ve| have) (?:told|said|given))\b"
    r"|\bmy (visit |appointment )?summary\b"
    r"|\bwhat (do )?you have so far\b",
    flags=re.IGNORECASE,
)

# Wording that asks for something already recorded to be changed or removed.
CORRECTION_REQUEST_PATTERN = re.compile(
    r"\b(actually|instead|correction|wrong|incorrect|not right|mistake)\b"
    r"|\b(change|update|correct|fix|remove|delete|add|redo)\b"
    r"|\bi never said\b|\bi forgot\b|\bshould be\b|\bnot [a-z0-9]+,? (but|it'?s)\b",
    flags=re.IGNORECASE,
)


def is_summary_request(text: str) -> bool:
    return bool(SUMMARY_REQUEST_PATTERN.search(text))


def is_correction_request(text: str) -> bool:
    return bool(CORRECTION_REQUEST_PATTERN.search(text))


def normalize_route_text(text: str) -> str:
    """Normalize free-text commands so exact command matching is reliable."""
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_global_command(text: str) -> RouteAction | None:
    """Detect whether the message is a deterministic global control command.

    Multi-word commands also match inside a sentence, because people ask for
    them politely — "actually, can we start over?" is a restart. Single words
    stay exact-match, so "back pain" is not read as "go back".
    """
    normalized = normalize_route_text(text).rstrip("?!.")
    for action, commands in GLOBAL_COMMANDS.items():
        for command in commands:
            if normalized == command:
                return action
            if " " in command and command in normalized:
                return action
    return None


def _menu_decision(
    state: ConversationState,
    action: RouteAction,
    response: str = MENU_PROMPT_RESPONSE,
) -> RouteDecision:
    """Reset the conversation to the menu and return a handled routing decision."""
    state.phase = ConversationPhase.MENU
    state.workflow = None
    state.missing_fields = []
    state.validation_errors = {}
    state.confirmed = False
    return RouteDecision(
        action=action,
        handled=True,
        source="global_command",
        response=response,
    )


def _clear_completion_metadata(state: ConversationState) -> None:
    """Clear summary and persistence state when restarting or changing workflows."""
    state.confirmed = False
    state.summary_text = None
    state.confirmation_attempt_count = 0
    state.visit_id = None
    state.persisted_at = None
    state.persistence_error = None


def _start_workflow(
    state: ConversationState,
    workflow: WorkflowType,
    confidence: float,
    source: str,
) -> RouteDecision:
    """Move the conversation into a selected workflow and return the start response.

    Workflow handling is explicit by workflow type:
    - `APPOINTMENT_PREPARATION`: clear prior completion metadata, switch to `COLLECTING`, and start guided intake.
    - `REPORT_NEW_SYMPTOMS`: clear prior completion metadata, switch to `COLLECTING`, and start symptom collection.
    - `REVIEW_HEALTH_NOTES`: clear prior completion metadata, switch to `COLLECTING`, and collect identity/contact details before review.
    - `VIEW_SUMMARY`: clear prior completion metadata, switch to `COLLECTING`, and collect identity/contact details before summary review.
    - `REPORT_ALLERGY`: clear prior completion metadata, switch to `COLLECTING`, and start allergy collection.
    - `MEDICATION_QUESTION`: clear prior completion metadata, switch to `COLLECTING`, and start medication collection.
    - `EMERGENCY_SUPPORT`: switch to `ESCALATED`, mark the emergency flag, and return the emergency response.

    The function also clears or resets completion metadata where needed so a new workflow starts cleanly.
    """
    state.workflow = workflow
    if workflow in {
        WorkflowType.APPOINTMENT_PREPARATION,
        WorkflowType.REPORT_NEW_SYMPTOMS,
        WorkflowType.REPORT_ALLERGY,
        WorkflowType.MEDICATION_QUESTION,
    }:
        _clear_completion_metadata(state)
    else:
        state.confirmed = False
        state.confirmation_attempt_count = 0
        state.persistence_error = None
    state.validation_errors = {}

    if workflow is WorkflowType.EMERGENCY_SUPPORT:
        state.phase = ConversationPhase.ESCALATED
        state.emergency_detected = True
        state.missing_fields = []
    else:
        state.phase = ConversationPhase.COLLECTING
        refresh_state_completeness(state)

    definition = WORKFLOW_CATALOG[workflow]
    start_response = (
        EMERGENCY_ESCALATION_RESPONSE
        if workflow is WorkflowType.EMERGENCY_SUPPORT
        else definition.start_response
    )
    return RouteDecision(
        action=RouteAction.START_WORKFLOW,
        handled=True,
        workflow=workflow,
        confidence=confidence,
        source=source,
        response=start_response,
        # Any workflow started from a content-bearing message must extract that
        # message. Only a bare menu number conveys nothing to extract.
        collect_message=(
            source in {"intent_classifier", "visit_information"}
            and workflow is not WorkflowType.EMERGENCY_SUPPORT
        ),
    )


def _handle_global_command(
    state: ConversationState,
    action: RouteAction,
) -> RouteDecision:
    """Apply a deterministic state transition for a global command."""
    if action is RouteAction.SHOW_MENU:
        return _menu_decision(state, action)

    if action is RouteAction.RESTART:
        state.visit_data = VisitData()
        state.extraction_retry_count = 0
        state.validation_attempt_count = 0
        state.emergency_detected = False
        _clear_completion_metadata(state)
        return _menu_decision(
            state,
            action,
            "The conversation has been restarted.\n\n" + MENU_PROMPT_RESPONSE,
        )

    if action is RouteAction.CANCEL:
        return _menu_decision(
            state,
            action,
            "The current workflow has been cancelled. Your collected visit information is still available.\n\n"
            + MENU_PROMPT_RESPONSE,
        )

    if action is RouteAction.GO_BACK:
        if state.workflow and state.phase in {
            ConversationPhase.REVIEWING,
            ConversationPhase.AWAITING_CONFIRMATION,
        }:
            state.phase = ConversationPhase.COLLECTING
            refresh_state_completeness(state)
            return RouteDecision(
                action=action,
                handled=True,
                workflow=state.workflow,
                source="global_command",
                response="We can return to collecting your visit information. What would you like to update?",
            )
        return _menu_decision(state, action)

    if action is RouteAction.CHANGE_ANSWER and state.workflow is not None:
        state.phase = ConversationPhase.COLLECTING
        _clear_completion_metadata(state)
        refresh_state_completeness(state)
        return RouteDecision(
            action=action,
            handled=True,
            workflow=state.workflow,
            source="global_command",
            response="Tell me which answer you want to change and the corrected information.",
        )

    return _menu_decision(
        state,
        action,
        "Start a workflow before changing an answer.\n\n" + MENU_PROMPT_RESPONSE,
    )


def route_message(
    state: ConversationState,
    text: str,
    intent_classifier: Callable[[str], dict[str, object]] | None = None,
) -> RouteDecision:
    """Route one message using state first, then commands, then intent classification.

    The router handles emergency and completed states immediately, then checks
    deterministic global commands and menu selections, and finally falls back
    to intent classification when the conversation is at the menu.
    """

    if state.phase is ConversationPhase.ESCALATED:
        return RouteDecision(
            action=RouteAction.CONTINUE,
            handled=True,
            workflow=WorkflowType.EMERGENCY_SUPPORT,
            source="emergency_state",
            response=EMERGENCY_ESCALATION_RESPONSE,
        )

    if global_action := detect_global_command(text):
        return _handle_global_command(state, global_action)

    normalized = normalize_route_text(text)

    if state.phase is ConversationPhase.COMPLETED:
        # Confirming a summary is not the end of the conversation. Someone who
        # remembers an allergy a moment later must be able to add it, so a
        # correction reopens collection instead of hitting a dead end.
        if is_correction_request(normalized):
            state.phase = ConversationPhase.COLLECTING
            state.confirmed = False
            state.confirmation_attempt_count = 0
            refresh_state_completeness(state)
            return RouteDecision(
                action=RouteAction.CHANGE_ANSWER,
                handled=False,
                workflow=state.workflow,
                source="completed_state_correction",
            )
        if is_summary_request(normalized):
            state.phase = ConversationPhase.REVIEWING
            return RouteDecision(
                action=RouteAction.CONTINUE,
                handled=True,
                workflow=state.workflow,
                source="completed_state",
            )
        return RouteDecision(
            action=RouteAction.CONTINUE,
            handled=True,
            workflow=state.workflow,
            source="completed_state",
            response=(
                "Your visit summary is confirmed. You can say summary to see it "
                "again, tell me anything you want to change, or say restart to "
                "begin a new one."
            ),
        )

    # An explicit request to see the summary is honoured even mid-workflow;
    # otherwise the only way to see what has been collected is to finish.
    if is_summary_request(normalized) and state.workflow is not None:
        # Asking to see the record is the view-summary intent, whatever workflow
        # collected it, and the reported intent should say so. The collected data
        # is untouched, and the summary workflow accepts every field, so a
        # correction made from the summary still merges.
        state.workflow = WorkflowType.VIEW_SUMMARY
        state.phase = ConversationPhase.REVIEWING
        return RouteDecision(
            action=RouteAction.CONTINUE,
            handled=True,
            workflow=state.workflow,
            source="summary_request",
        )
    # A menu number is only a menu choice while the menu is what is on screen.
    # Mid-collection a bare number is an answer — a severity of 7, a policy
    # number, a house number — and reading it as "option 7" both discards the
    # answer and silently switches the workflow out from under the patient.
    at_the_menu = state.workflow is None or state.phase is ConversationPhase.MENU
    if at_the_menu and normalized in MENU_OPTION_TO_WORKFLOW:
        return _start_workflow(
            state,
            MENU_OPTION_TO_WORKFLOW[normalized],
            confidence=1.0,
            source="menu_option",
        )

    if state.workflow is not None and state.phase is not ConversationPhase.MENU:
        return RouteDecision(
            action=RouteAction.CONTINUE,
            handled=False,
            workflow=state.workflow,
            source="active_state",
        )

    # Content the patient supplied outranks a weak classifier label. The
    # classifier reaches for show_menu whenever a message does not look like a
    # workflow request, and "I'm Dana Whitfield, dana@example.com, 555-0100" is
    # exactly that shape — answering it with the option list threw the details
    # away. Nothing carrying recordable information is ever met with the menu.
    carries_information = looks_like_visit_information(text)

    intent_result = intent_classifier(text) if intent_classifier else {}
    intent = str(intent_result.get("intent", "unknown"))
    confidence = float(intent_result.get("confidence", 0.0))

    if intent == SHOW_MENU_INTENT and not carries_information:
        return _menu_decision(state, RouteAction.SHOW_MENU)

    try:
        workflow = WorkflowType(intent)
    except ValueError:
        workflow = None
    if workflow is not None and confidence >= INTENT_CONFIDENCE_THRESHOLD:
        return _start_workflow(state, workflow, confidence, source="intent_classifier")

    # The classifier has no label for "here are my contact details", but such a
    # message plainly belongs in the record. Starting intake keeps it; showing
    # the menu throws away what the patient just typed.
    if carries_information:
        return _start_workflow(
            state,
            WorkflowType.APPOINTMENT_PREPARATION,
            confidence=INTENT_CONFIDENCE_THRESHOLD,
            source="visit_information",
        )

    return _menu_decision(
        state,
        RouteAction.FALLBACK,
        "I'm sorry, I didn't understand that. Could you rephrase it in a short "
        "sentence, or pick one of these options?\n\n" + MENU_PROMPT_RESPONSE,
    )
