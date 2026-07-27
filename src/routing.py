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


GLOBAL_COMMANDS = {
    RouteAction.SHOW_MENU: set(SHOW_MENU_COMMANDS),
    RouteAction.RESTART: {"restart", "start over", "restart conversation"},
    RouteAction.CANCEL: {"cancel", "cancel workflow", "stop this workflow"},
    RouteAction.GO_BACK: {"go back", "back", "previous"},
    RouteAction.CHANGE_ANSWER: {"change answer", "change my answer", "correct an answer"},
}


def normalize_route_text(text: str) -> str:
    """Normalize free-text commands so exact command matching is reliable."""
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_global_command(text: str) -> RouteAction | None:
    """Detect whether the message is a deterministic global control command."""
    normalized = normalize_route_text(text)
    for action, commands in GLOBAL_COMMANDS.items():
        if normalized in commands:
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
    - `REPORT_ALLERGY`: clear prior completion metadata, switch to `COLLECTING`, and start allergy collection.
    - `MEDICATION_QUESTION`: clear prior completion metadata, switch to `COLLECTING`, and start medication collection.
    - `REVIEW_HEALTH_NOTES`: switch to `REVIEWING`, clear missing fields, and wait for summary review.
    - `VIEW_SUMMARY`: switch to `REVIEWING`, clear missing fields, and wait for summary review.
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
    elif workflow in {WorkflowType.REVIEW_HEALTH_NOTES, WorkflowType.VIEW_SUMMARY}:
        state.phase = ConversationPhase.REVIEWING
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

    if state.phase is ConversationPhase.COMPLETED:
        return RouteDecision(
            action=RouteAction.CONTINUE,
            handled=True,
            workflow=state.workflow,
            source="completed_state",
            response=(
                "This visit summary is already confirmed. "
                "Choose menu, restart, or change answer to continue."
            ),
        )

    normalized = normalize_route_text(text)
    if normalized in MENU_OPTION_TO_WORKFLOW:
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

    intent_result = intent_classifier(text) if intent_classifier else {}
    intent = str(intent_result.get("intent", "unknown"))
    confidence = float(intent_result.get("confidence", 0.0))

    if intent == SHOW_MENU_INTENT:
        return _menu_decision(state, RouteAction.SHOW_MENU)

    try:
        workflow = WorkflowType(intent)
    except ValueError:
        workflow = None
    if workflow is not None and confidence >= INTENT_CONFIDENCE_THRESHOLD:
        return _start_workflow(state, workflow, confidence, source="intent_classifier")

    return _menu_decision(
        state,
        RouteAction.FALLBACK,
        "I couldn't determine which workflow you need. Please choose an option.\n\n"
        + MENU_PROMPT_RESPONSE,
    )
