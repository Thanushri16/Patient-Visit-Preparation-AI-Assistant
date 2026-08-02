"""Orchestrates the typed healthcare prompt chain for each conversation turn."""

import os
import re
from pathlib import Path
from time import perf_counter

from openai import OpenAI, OpenAIError

try:
    from .extraction import build_acknowledgement, process_collection_turn
    from .guidance import (
        AMBIGUOUS_RESPONSE,
        FAREWELL_RESPONSE,
        GREETING_RESPONSE,
        OFF_TOPIC_RESPONSE,
        answer_state_query,
        build_supplementary_response,
        detect_ambiguous,
        detect_bare_greeting,
        detect_farewell,
        detect_off_topic,
        is_low_information,
        looks_non_english,
    )
    from .moderation import moderate_text
    from .models import ChatMessage, ConversationPhase, ConversationState, WorkflowType
    from .observability import emit_chain_event
    from .persistence import JsonVisitRepository
    from .routing import (
        ROUTER_VERSION,
        detect_global_command,
        is_summary_request,
        normalize_route_text,
        route_message,
    )
    from .summary_workflow import (
        begin_summary_review,
        build_change_summary,
        classify_confirmation,
        confirmation_response,
    )
    from .chatbot_content import (
        API_KEY_ERROR_MESSAGE,
        BLOCKED_RESPONSE,
        DEFAULT_MODEL,
        INJECTION_NEUTRALIZED_NOTICE,
        NON_ENGLISH_RESPONSE,
        UNCLEAR_INPUT_RESPONSE,
    )
    from .workflow_catalog import (
        INTENT_LABELS,
        MENU_OPTION_TO_WORKFLOW,
        MENU_PROMPT_RESPONSE,
        build_intent_classifier_prompt,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from extraction import build_acknowledgement, process_collection_turn
    from guidance import (
        AMBIGUOUS_RESPONSE,
        FAREWELL_RESPONSE,
        GREETING_RESPONSE,
        OFF_TOPIC_RESPONSE,
        answer_state_query,
        build_supplementary_response,
        detect_ambiguous,
        detect_bare_greeting,
        detect_farewell,
        detect_off_topic,
        is_low_information,
        looks_non_english,
    )
    from moderation import moderate_text
    from models import ChatMessage, ConversationPhase, ConversationState, WorkflowType
    from observability import emit_chain_event
    from persistence import JsonVisitRepository
    from routing import (
        ROUTER_VERSION,
        detect_global_command,
        is_summary_request,
        normalize_route_text,
        route_message,
    )
    from summary_workflow import (
        begin_summary_review,
        build_change_summary,
        classify_confirmation,
        confirmation_response,
    )
    from chatbot_content import (
        API_KEY_ERROR_MESSAGE,
        BLOCKED_RESPONSE,
        DEFAULT_MODEL,
        INJECTION_NEUTRALIZED_NOTICE,
        NON_ENGLISH_RESPONSE,
        UNCLEAR_INPUT_RESPONSE,
    )
    from workflow_catalog import (
        INTENT_LABELS,
        MENU_OPTION_TO_WORKFLOW,
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

# Workflows whose whole purpose is to display the record rather than add to it.
SUMMARY_WORKFLOWS = {WorkflowType.VIEW_SUMMARY, WorkflowType.REVIEW_HEALTH_NOTES}


def _finalize(
    state: ConversationState,
    messages: list[ChatMessage],
    prompt: str,
    response_text: str,
) -> str:
    """Apply output moderation, record the turn, and return the safe reply."""

    output_moderation = moderate_text(response_text, stage="output")
    if output_moderation.action in {"block", "sanitize"}:
        response_text = output_moderation.response or BLOCKED_RESPONSE
    _record_turn(messages, prompt, response_text)
    return response_text


def _compose(*segments: str) -> str:
    """Join the non-empty parts of a reply into one paragraph."""

    return " ".join(segment.strip() for segment in segments if segment and segment.strip())



# A sign-off that still carries an ask must be answered, not just acknowledged.
FAREWELL_CARRIES_REQUEST = re.compile(
    r"\b(summary|show|can you|could you|before i go|one more|also)\b",
    flags=re.IGNORECASE,
)

# Wording that asks for the summary as a document to be produced or checked,
# rather than as something to read.
JSON_SUMMARY_REQUEST = re.compile(
    r"\b(generate|json|schema|validate|export|structured)\b", flags=re.IGNORECASE
)


def _render_summary(state: ConversationState, prompt: str) -> str:
    """Render the visit summary in whichever form the request called for.

    A request to generate or validate a summary produces the JSON document that
    consumers check against the schema; a conversational request produces prose.
    Either way the canonical summary is what comes back. Substituting a sentence
    when the record is empty made the reply structurally different from every
    other summary and, being unrepeatable, not idempotent either — an empty
    summary is still a summary, and it shows exactly which fields are open.
    """

    if JSON_SUMMARY_REQUEST.search(prompt):
        return begin_summary_review(state, as_json=True)

    summary = begin_summary_review(state)
    if not state.visit_data.model_dump(exclude_none=True):
        return _compose(
            "Nothing has been recorded for this visit yet, so every field is "
            "still open:",
            summary,
        )
    return summary


def get_chatbot_response(
    messages: list[ChatMessage],
    prompt: str,
    client: OpenAI | None,
    state: ConversationState,
    visit_repository: JsonVisitRepository | None = None,
) -> str:
    """Run one chat turn through moderation, routing, extraction, confirmation, and persistence.

    Flow:
    1. Moderate the user input: escalate emergencies, refuse unsafe requests, and
       strip embedded instruction payloads from otherwise legitimate messages.
    2. Answer questions that only read back already-recorded state.
    3. Route control commands or menu-style messages before any workflow processing.
    4. If routing started a workflow from a content-bearing message, feed that same
       message into extraction rather than discarding it.
    5. If the chatbot is awaiting confirmation, classify the reply as confirm, correct, or unclear.
    6. Otherwise continue the active workflow with structured extraction, validation, and the next question.
    7. When the summary is confirmed, save the visit locally if a repository is available.
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
    if input_moderation.action == "escalate":
        # An active emergency ends intake and returns guidance specific to it.
        state.phase = ConversationPhase.ESCALATED
        state.workflow = WorkflowType.EMERGENCY_SUPPORT
        state.emergency_detected = True
        state.missing_fields = []
        state.visit_data.emergency_symptoms = [prompt]
        safe_reply = _compose(
            input_moderation.response or BLOCKED_RESPONSE,
            "I've noted what you told me so it is here when you come back.",
        )
        _record_turn(messages, prompt, safe_reply)
        return safe_reply

    if input_moderation.action in {"block", "redirect"}:
        # Refuse the unsafe request itself while leaving the conversation open.
        safe_reply = input_moderation.response or BLOCKED_RESPONSE
        _record_turn(messages, prompt, safe_reply)
        return safe_reply

    injection_notice = ""
    if input_moderation.action == "neutralize" and input_moderation.sanitized_text:
        # The message carried real health information around an instruction
        # payload; keep the information, drop the payload, and say so.
        prompt = input_moderation.sanitized_text
        injection_notice = INJECTION_NEUTRALIZED_NOTICE
    # Input moderation else path: continue to routing because the message is safe enough to process.

    # Flow 2: unreadable input never reaches the classifier or the extractor.
    # A bare menu number or a global command carries no prose and would look
    # like gibberish to that check, so routing gets first refusal on them.
    # A reply to a question the assistant just asked is data, whatever it looks
    # like. Screening it as unreadable, off-topic or non-English rejects the very
    # answer that was requested — an interpreter's name, a member ID, a dose.
    answering_a_question = (
        state.requested_field is not None
        and state.phase is ConversationPhase.COLLECTING
    )
    routable = (
        answering_a_question
        or normalize_route_text(prompt) in MENU_OPTION_TO_WORKFLOW
        or detect_global_command(prompt) is not None
    )
    if not routable:
        if is_low_information(prompt):
            return _finalize(state, messages, prompt, UNCLEAR_INPUT_RESPONSE)
        if looks_non_english(prompt):
            return _finalize(state, messages, prompt, NON_ENGLISH_RESPONSE)
        # Decline out-of-scope requests by name and answer a goodbye as a
        # goodbye. Falling through to the option list would answer neither.
        if detect_bare_greeting(prompt):
            return _finalize(state, messages, prompt, GREETING_RESPONSE)
        if detect_off_topic(prompt):
            return _finalize(state, messages, prompt, OFF_TOPIC_RESPONSE)
        # A goodbye that also asks for something — "that's everything, can you
        # show me the summary before I go?" — is a request first and a farewell
        # second, so it continues to routing and picks up the sign-off later.
        if (
            detect_farewell(prompt)
            and state.phase is not ConversationPhase.COLLECTING
            and not FAREWELL_CARRIES_REQUEST.search(prompt)
        ):
            return _finalize(state, messages, prompt, FAREWELL_RESPONSE)
        if detect_ambiguous(prompt):
            return _finalize(state, messages, prompt, AMBIGUOUS_RESPONSE)

    # Flow 3: questions about what has already been said are answered from state,
    # never by re-running extraction over the question itself.
    if state_answer := answer_state_query(prompt, state.visit_data):
        return _finalize(state, messages, prompt, state_answer)

    # Flow 4: route menu commands and global controls before workflow extraction.
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
    # Flow 4a: if routing fully handled the turn, return the routed response.
    # Routing is handled when the message is an explicit global command, a menu option,
    # or a state-based continuation such as completed or escalated. A workflow started
    # from a content-bearing message is *not* finished here: `collect_message` marks
    # that the same message still has to go through extraction below.
    if route_decision.handled and not route_decision.collect_message:
        # A summary workflow has nothing to collect before it can answer: it
        # renders whatever the session already holds, however partial.
        if route_decision.workflow in SUMMARY_WORKFLOWS:
            _hydrate_visit_from_repository(state, visit_repository)
            route_reply = _render_summary(state, prompt)
        # Routing handled output path: show the review summary when the user entered review mode.
        elif state.phase is ConversationPhase.REVIEWING:
            route_reply = begin_summary_review(state)
        # Routing handled output path: otherwise return the router's own response or the menu prompt.
        else:
            route_reply = route_decision.response or MENU_PROMPT_RESPONSE
        return _finalize(state, messages, prompt, route_reply)
    # Routing else path: no control command was handled, so continue into workflow processing.

    # Flow 5: if the chatbot is waiting on review, classify confirm vs correction.
    if state.phase is ConversationPhase.AWAITING_CONFIRMATION:
        # Asking to see the summary is not agreeing to it. Classifying "show me
        # my summary" as a confirmation silently finalised records the patient
        # had only asked to read.
        if is_summary_request(normalize_route_text(prompt)):
            return _finalize(
                state, messages, prompt, _render_summary(state, prompt)
            )
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
            if correction_result.merge_result.errors:
                # Only an invalid correction sends the user back for detail. A
                # correction that merely leaves other fields unanswered still
                # gets the corrected summary shown, because reviewing it is what
                # the user was in the middle of doing.
                response_text = correction_result.response
            else:
                # Correction output path with a clean merge: show what changed,
                # then re-offer the record for confirmation. Leading with the
                # delta is what the patient asked about; the full summary
                # follows so nothing is hidden.
                merge = correction_result.merge_result
                changed = merge.accepted_fields + merge.cleared_fields
                delta = build_change_summary(state.visit_data, changed)
                summary = begin_summary_review(state)
                response_text = _compose(
                    f"Updated:\n{delta}" if delta else "",
                    ("Nothing else has changed. " if delta else ""),
                    summary,
                )
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

        return _finalize(state, messages, prompt, response_text)
    # Confirmation else path: the chatbot is not awaiting review, so continue to active collection.

    # Flow 6: continue the active collection workflow with extraction and validation.
    if state.phase is ConversationPhase.COLLECTING:
        # Everything in the message that is not a field — a greeting, an expressed
        # worry, a general question about preparing — is answered alongside the
        # structured intake rather than instead of it.
        supplementary = build_supplementary_response(state, prompt)
        summary_requested = state.workflow in SUMMARY_WORKFLOWS

        # Collection input path: extract structured fields from the user's latest message.
        collection_result = process_collection_turn(client, state, prompt)
        merge_result = collection_result.merge_result
        if not merge_result.errors and not merge_result.missing_fields:
            if summary_requested:
                _hydrate_visit_from_repository(state, visit_repository)
            # Collection success path: enough information was collected, so show the summary.
            acknowledgement = build_acknowledgement(state, merge_result)
            summary = (
                _render_summary(state, prompt)
                if summary_requested
                else begin_summary_review(state)
            )
            farewell = (
                "Take care, and good luck at your appointment."
                if detect_farewell(prompt)
                else ""
            )
            response_text = (
                summary
                if summary_requested
                else _compose(
                    injection_notice, supplementary, acknowledgement, summary, farewell
                )
            )
        else:
            # Collection continuation path: return the next question or validation feedback.
            # A message that both supplies information and asks to see the record
            # gets both: the update is applied and the summary is shown, rather
            # than silently dropping the half that did not drive the routing.
            trailing_summary = (
                begin_summary_review(state)
                if is_summary_request(normalize_route_text(prompt))
                else ""
            )
            response_text = _compose(
                injection_notice,
                supplementary,
                collection_result.response,
                trailing_summary,
            )
        return _finalize(state, messages, prompt, response_text)
    # Collection else path: no collection workflow is active, so fall back to the menu response.

    # Flow 7: fallback to the menu prompt when no workflow branch applies.
    fallback = _compose(build_supplementary_response(state, prompt), MENU_PROMPT_RESPONSE)
    _record_turn(messages, prompt, fallback)
    return fallback
