"""Orchestrates the typed healthcare prompt chain for each conversation turn."""

import os
from pathlib import Path
from time import perf_counter

from openai import OpenAI, OpenAIError

try:
    from .extraction import process_collection_turn
    from .moderation import moderate_text
    from .models import ChatMessage, ConversationPhase, ConversationState, WorkflowType
    from .observability import emit_chain_event
    from .persistence import JsonVisitRepository
    from .routing import ROUTER_VERSION, route_message
    from .summary_workflow import (
        begin_summary_review,
        classify_confirmation,
        confirmation_response,
    )
    from .chatbot_content import (
        API_KEY_ERROR_MESSAGE,
        BLOCKED_RESPONSE,
        DEFAULT_MODEL,
    )
    from .workflow_catalog import (
        INTENT_LABELS,
        MENU_PROMPT_RESPONSE,
        build_intent_classifier_prompt,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from extraction import process_collection_turn
    from moderation import moderate_text
    from models import ChatMessage, ConversationPhase, ConversationState, WorkflowType
    from observability import emit_chain_event
    from persistence import JsonVisitRepository
    from routing import ROUTER_VERSION, route_message
    from summary_workflow import (
        begin_summary_review,
        classify_confirmation,
        confirmation_response,
    )
    from chatbot_content import (
        API_KEY_ERROR_MESSAGE,
        BLOCKED_RESPONSE,
        DEFAULT_MODEL,
    )
    from workflow_catalog import (
        INTENT_LABELS,
        MENU_PROMPT_RESPONSE,
        build_intent_classifier_prompt,
    )


def _unknown_intent() -> dict[str, object]:
    return {
        "intent": "unknown",
        "confidence": 0.0,
        "top_score": 0.0,
        "margin": 0.0,
        "status": "unknown",
    }


def classify_intent(text: str, client: OpenAI | None = None) -> dict[str, object]:
    """Classify directly with the model; never run keyword scoring first."""

    if client is None or not isinstance(text, str) or not text.strip():
        return _unknown_intent()

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": build_intent_classifier_prompt(text)}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        label = response.choices[0].message.content.strip().lower().splitlines()[0]
    except (OpenAIError, AttributeError, IndexError, TypeError):
        return _unknown_intent()
    if label not in INTENT_LABELS:
        return _unknown_intent()
    return {
        "intent": label,
        "confidence": 0.85,
        "top_score": 0.85,
        "margin": 0.0,
        "status": "model",
    }


def load_api_key():
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            normalized_line = line.strip()
            if not normalized_line or normalized_line.startswith("#"):
                continue
            if normalized_line.startswith("export "):
                normalized_line = normalized_line[len("export "):].strip()
            if "=" not in normalized_line:
                continue
            key, value = normalized_line.split("=", 1)
            if key.strip() != "OPENAI_API_KEY":
                continue
            key_value = value.strip().strip('"').strip("'")
            if key_value:
                os.environ["OPENAI_API_KEY"] = key_value
                return key_value

    raise RuntimeError(
        API_KEY_ERROR_MESSAGE
    )


def _record_turn(messages: list[ChatMessage], prompt: str, response: str) -> None:
    """Store user-visible memory separately from the workflow state."""

    messages.extend(
        [
            ChatMessage(role="user", content=prompt),
            ChatMessage(role="assistant", content=response),
        ]
    )


def _hydrate_visit_from_repository(
    state: ConversationState,
    visit_repository: JsonVisitRepository | None,
) -> None:
    """Merge any previously stored person record into the current visit data."""

    if visit_repository is None or not state.visit_data.email:
        return

    stored_record = visit_repository.load_by_email(state.visit_data.email)
    if stored_record is None:
        return

    merged_visit_data = stored_record.visit_data.model_copy(deep=True)
    for field_name, field_value in state.visit_data.model_dump(exclude_none=True).items():
        setattr(merged_visit_data, field_name, field_value)
    merged_visit_data.email = stored_record.email
    state.visit_data = merged_visit_data

def get_chatbot_response(
    messages: list[ChatMessage],
    prompt: str,
    client: OpenAI | None,
    state: ConversationState,
    visit_repository: JsonVisitRepository | None = None,
) -> str:
    """Run one chat turn through moderation, routing, extraction, confirmation, and persistence.

    Flow:
    1. Moderate the user input and block or escalate unsafe content immediately.
    2. Route control commands or menu-style messages before any workflow processing.
    3. If the chatbot is awaiting confirmation, classify the reply as confirm, correct, or unclear.
    4. If the user is correcting the summary, send the correction back through extraction and validation.
    5. Otherwise continue the active workflow with structured extraction, validation, and the next question.
    6. When the summary is confirmed, save the visit locally if a repository is available.
    """
    input_moderation = moderate_text(prompt, stage="input")
    emit_chain_event(
        state,
        "input_guardrail",
        success=True,
        metadata={
            "action": input_moderation.action,
            "risk_level": input_moderation.risk_level,
            "reason_count": len(input_moderation.reasons),
        },
    )
    # Flow 1: stop unsafe input before any routing or workflow processing.
    if input_moderation.action in {"block", "escalate"}:
        # Input moderation output path: return the safety response immediately.
        safe_reply = input_moderation.response or BLOCKED_RESPONSE
        if input_moderation.action == "escalate":
            # Input moderation escalation path: switch the conversation into emergency support.
            state.phase = ConversationPhase.ESCALATED
            state.workflow = WorkflowType.EMERGENCY_SUPPORT
            state.emergency_detected = True
            state.missing_fields = []
        _record_turn(messages, prompt, safe_reply)
        return safe_reply
    # Input moderation else path: continue to routing because the message is safe enough to process.

    # Flow 2: route menu commands and global controls before workflow extraction.
    routing_phase_before = state.phase.value
    routing_started = perf_counter()
    route_decision = route_message(
        state,
        prompt,
        intent_classifier=lambda text: classify_intent(text, client),
    )
    emit_chain_event(
        state,
        "state_router",
        success=True,
        latency_ms=(perf_counter() - routing_started) * 1_000,
        prompt_version=ROUTER_VERSION,
        metadata={
            "action": route_decision.action.value,
            "source": route_decision.source,
            "handled": route_decision.handled,
        },
        phase_before=routing_phase_before,
    )
    # Flow 2a: if routing handled the turn, return the routed response immediately.
    # Routing is handled when the message is an explicit global command, a menu option,
    # an intent-classified workflow start, or a state-based continuation such as completed or escalated.
    if route_decision.handled:
        # Routing handled output path: show the review summary when the user entered review mode.
        if state.phase is ConversationPhase.REVIEWING:
            route_reply = begin_summary_review(state)
        # Routing handled output path: otherwise return the router's own response or the menu prompt.
        else:
            route_reply = route_decision.response or MENU_PROMPT_RESPONSE
        # Output moderation path for routed replies: sanitize or block the response if needed.
        route_output_moderation = moderate_text(route_reply, stage="output")
        if route_output_moderation.action in {"block", "sanitize"}:
            # Output moderation nested path: replace unsafe routed text with a safe response.
            route_reply = route_output_moderation.response or BLOCKED_RESPONSE
        _record_turn(messages, prompt, route_reply)
        return route_reply
    # Routing else path: no control command was handled, so continue into workflow processing.

    # Flow 3: if the chatbot is waiting on review, classify confirm vs correction.
    if state.phase is ConversationPhase.AWAITING_CONFIRMATION:
        # Confirmation input path: classify the user's reply against the displayed summary.
        confirmation_phase_before = state.phase.value
        confirmation = classify_confirmation(
            client,
            prompt,
            state.summary_text or begin_summary_review(state),
        )
        # Flow 4: a correction goes back through extraction and validation.
        if confirmation.action.value == "correct":
            # Confirmation correction path: re-enter collection with the correction text.
            state.phase = ConversationPhase.COLLECTING
            state.confirmed = False
            state.confirmation_attempt_count = 0
            correction_result = process_collection_turn(
                client,
                state,
                confirmation.correction_text or prompt,
            )
            if correction_result.merge_result.errors or correction_result.merge_result.missing_fields:
                # Correction output path with errors or missing data: keep asking for the missing or invalid fields.
                response_text = correction_result.response
            else:
                # Correction output path with a clean merge: regenerate the summary for another confirmation pass.
                response_text = begin_summary_review(state)
        else:
            # Flow 5: confirm or unclear replies use the confirmation response path.
            response_text = confirmation_response(state, confirmation)
            emit_chain_event(
                state,
                "confirmation_classifier",
                success=confirmation.action.value != "unclear",
                prompt_version="confirmation_classifier_v1",
                retry_count=state.confirmation_attempt_count,
                error_category=(
                    "unclear_confirmation" if confirmation.action.value == "unclear" else None
                ),
                metadata={"action": confirmation.action.value},
                phase_before=confirmation_phase_before,
            )
            if state.phase is ConversationPhase.COMPLETED and visit_repository is not None:
                persistence_started = perf_counter()
                try:
                    saved_path = visit_repository.save_confirmed(state)
                    response_text += f" It was saved locally with visit ID {state.visit_id}."
                    emit_chain_event(
                        state,
                        "confirmed_visit_persistence",
                        success=True,
                        latency_ms=(perf_counter() - persistence_started) * 1_000,
                        metadata={"file_extension": saved_path.suffix},
                    )
                except (OSError, TypeError, ValueError) as exc:
                    state.persistence_error = type(exc).__name__
                    response_text += " I couldn't save it locally; your session still has the summary."
                    emit_chain_event(
                        state,
                        "confirmed_visit_persistence",
                        success=False,
                        latency_ms=(perf_counter() - persistence_started) * 1_000,
                        error_category=type(exc).__name__,
                    )

        output_moderation = moderate_text(response_text, stage="output")
        if output_moderation.action in {"block", "sanitize"}:
            # Confirmation output moderation path: replace unsafe confirmation text before returning it.
            response_text = output_moderation.response or BLOCKED_RESPONSE
        _record_turn(messages, prompt, response_text)
        return response_text
    # Confirmation else path: the chatbot is not awaiting review, so continue to active collection.

    # Flow 6: continue the active collection workflow with extraction and validation.
    if state.phase is ConversationPhase.COLLECTING:
        # Collection input path: extract structured fields from the user's latest message.
        collection_result = process_collection_turn(client, state, prompt)
        if not collection_result.merge_result.errors and not collection_result.merge_result.missing_fields:
            if state.workflow in {
                WorkflowType.REVIEW_HEALTH_NOTES,
                WorkflowType.VIEW_SUMMARY,
            }:
                _hydrate_visit_from_repository(state, visit_repository)
            # Collection success path: enough information was collected, so show the summary for review.
            response_text = begin_summary_review(state)
        else:
            # Collection continuation path: return the next question or validation feedback.
            response_text = collection_result.response
        # Collection output moderation path: sanitize or block any unsafe response text.
        output_moderation = moderate_text(response_text, stage="output")
        if output_moderation.action in {"block", "sanitize"}:
            # Collection nested output path: replace unsafe extracted-flow text with a safe response.
            response_text = output_moderation.response or BLOCKED_RESPONSE
        _record_turn(messages, prompt, response_text)
        return response_text
    # Collection else path: no collection workflow is active, so fall back to the menu response.

    # Flow 7: fallback to the menu prompt when no workflow branch applies.
    fallback = MENU_PROMPT_RESPONSE
    _record_turn(messages, prompt, fallback)
    return fallback
