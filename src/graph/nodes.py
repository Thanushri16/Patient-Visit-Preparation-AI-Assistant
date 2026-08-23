"""Graph nodes: thin adapters over the existing chain modules.

**No domain logic lives here.** Every node calls the function the chain already
calls — `moderate_text`, `route_message`, `process_collection_turn`,
`classify_confirmation` — and does nothing else. That rule is what makes
equivalence achievable rather than aspirational: a node that reimplements a
decision has to be verified against the original, and a node that delegates to
it cannot disagree with it.

Where a node needs a helper the chain keeps private, it imports that helper
rather than restating it. `_finalize` and `_render_summary` are the chain's own
implementations; copying their bodies here would be exactly the duplication this
rule exists to prevent.
"""

from __future__ import annotations

from time import perf_counter

try:
    from .. import chatbot
    from ..guidance import (
        AMBIGUOUS_RESPONSE, FAREWELL_RESPONSE, GREETING_RESPONSE, OFF_TOPIC_RESPONSE,
        answer_state_query, build_supplementary_response, detect_ambiguous,
        detect_bare_greeting, detect_farewell, detect_off_topic, is_low_information,
        looks_non_english,
    )
    from ..chatbot_content import (
        BLOCKED_RESPONSE, INJECTION_NEUTRALIZED_NOTICE, NON_ENGLISH_RESPONSE,
        UNCLEAR_INPUT_RESPONSE,
    )
    from ..models import ConversationPhase, WorkflowType
    from ..moderation import moderate_text
    from ..observability import emit_chain_event
    from ..routing import (
        ROUTER_VERSION, detect_global_command, is_summary_request,
        normalize_route_text, route_message,
    )
    from ..workflow_catalog import MENU_OPTION_TO_WORKFLOW, MENU_PROMPT_RESPONSE
    from .deps import deps_from
except ImportError:  # pragma: no cover - allows running as a script
    import chatbot
    from guidance import (
        AMBIGUOUS_RESPONSE, FAREWELL_RESPONSE, GREETING_RESPONSE, OFF_TOPIC_RESPONSE,
        answer_state_query, build_supplementary_response, detect_ambiguous,
        detect_bare_greeting, detect_farewell, detect_off_topic, is_low_information,
        looks_non_english,
    )
    from chatbot_content import (
        BLOCKED_RESPONSE, INJECTION_NEUTRALIZED_NOTICE, NON_ENGLISH_RESPONSE,
        UNCLEAR_INPUT_RESPONSE,
    )
    from models import ConversationPhase, WorkflowType
    from moderation import moderate_text
    from observability import emit_chain_event
    from routing import (
        ROUTER_VERSION, detect_global_command, is_summary_request,
        normalize_route_text, route_message,
    )
    from workflow_catalog import MENU_OPTION_TO_WORKFLOW, MENU_PROMPT_RESPONSE
    from graph.deps import deps_from


def input_guardrail(state, config=None):
    """Flow 1. Moderate before anything else touches the message."""

    conversation = state["conversation"]
    decision = moderate_text(state["user_message"], stage="input")
    emit_chain_event(
        conversation, "input_guardrail", success=True,
        metadata={
            "action": decision.action,
            "risk_level": decision.risk_level,
            "reason_count": len(decision.reasons),
        },
    )
    update = {"moderation_action": decision.action}

    if decision.action == "escalate":
        conversation.phase = ConversationPhase.ESCALATED
        conversation.workflow = WorkflowType.EMERGENCY_SUPPORT
        conversation.emergency_detected = True
        conversation.missing_fields = []
        conversation.visit_data.emergency_symptoms = [state["user_message"]]
        update["reply"] = chatbot._compose(
            decision.response or BLOCKED_RESPONSE,
            "I've noted what you told me so it is here when you come back.",
        )
    elif decision.action in {"block", "redirect"}:
        update["reply"] = decision.response or BLOCKED_RESPONSE
    elif decision.action == "neutralize" and decision.sanitized_text:
        update["user_message"] = decision.sanitized_text
        update["injection_notice"] = INJECTION_NEUTRALIZED_NOTICE
    return update


def preconversation_checks(state, config=None):
    """Flow 2. Unreadable or out-of-scope input never reaches a model."""

    conversation, prompt = state["conversation"], state["user_message"]
    answering = (
        conversation.requested_field is not None
        and conversation.phase is ConversationPhase.COLLECTING
    )
    routable = (
        answering
        or normalize_route_text(prompt) in MENU_OPTION_TO_WORKFLOW
        or detect_global_command(prompt) is not None
    )
    if routable:
        return {"branch": "route"}

    for matched, reply in (
        (is_low_information(prompt), UNCLEAR_INPUT_RESPONSE),
        (looks_non_english(prompt), NON_ENGLISH_RESPONSE),
        (detect_bare_greeting(prompt), GREETING_RESPONSE),
        (detect_off_topic(prompt), OFF_TOPIC_RESPONSE),
        (
            detect_farewell(prompt)
            and conversation.phase is not ConversationPhase.COLLECTING
            and not chatbot.FAREWELL_CARRIES_REQUEST.search(prompt),
            FAREWELL_RESPONSE,
        ),
        (detect_ambiguous(prompt), AMBIGUOUS_RESPONSE),
    ):
        if matched:
            return {"branch": "aside", "aside_reply": reply}
    return {"branch": "route"}


def direct_reply(state, config=None):
    """The pre-check's own answer, taken as the turn's reply."""

    return {"reply": state["aside_reply"]}


def state_recall(state, config=None):
    """Flow 3. Answer from VisitData, never by re-running extraction."""

    answer = answer_state_query(state["user_message"], state["conversation"].visit_data)
    return {"reply": answer, "branch": "aside"} if answer else {"branch": "route"}


def route(state, config=None):
    """Flow 4. State-aware routing, before any workflow processing."""

    conversation, deps = state["conversation"], deps_from(config)
    phase_before = conversation.phase.value
    started = perf_counter()
    decision = route_message(
        conversation,
        state["user_message"],
        intent_classifier=lambda text: chatbot.classify_intent(text, deps.client),
    )
    emit_chain_event(
        conversation, "state_router", success=True,
        latency_ms=(perf_counter() - started) * 1_000, prompt_version=ROUTER_VERSION,
        metadata={
            "action": decision.action.value,
            "source": decision.source,
            "handled": decision.handled,
        },
        phase_before=phase_before,
    )
    handled = decision.handled and not decision.collect_message
    return {
        "route_action": decision.action.value,
        "branch": "route_handled" if handled else _next_branch(conversation),
        "aside_reply": decision.response or "",
    }


def _next_branch(conversation) -> str:
    if conversation.phase is ConversationPhase.AWAITING_CONFIRMATION:
        return "confirm"
    if conversation.phase is ConversationPhase.COLLECTING:
        return "collect"
    return "fallback"


def route_handled(state, config=None):
    """Flow 4a. Routing finished the turn: summary, review, or a static reply."""

    conversation, deps = state["conversation"], deps_from(config)
    if conversation.workflow in chatbot.SUMMARY_WORKFLOWS:
        chatbot._hydrate_visit_from_repository(conversation, deps.visit_repository)
        return {"reply": chatbot._render_summary(conversation, state["user_message"])}
    if conversation.phase is ConversationPhase.REVIEWING:
        return {"reply": chatbot.begin_summary_review(conversation)}
    return {"reply": state.get("aside_reply") or MENU_PROMPT_RESPONSE}


def confirm(state, config=None):
    """Flow 5. Classify confirm, correct, or unclear against the shown summary."""

    conversation, deps = state["conversation"], deps_from(config)
    return {
        "reply": chatbot.handle_confirmation_turn(
            conversation, state["user_message"], deps.client, deps.visit_repository
        )
    }


def collect(state, config=None):
    """Flow 6. Extraction, validation, and the next question."""

    conversation, deps = state["conversation"], deps_from(config)
    return {
        "reply": chatbot.handle_collection_turn(
            conversation,
            state["user_message"],
            deps.client,
            deps.visit_repository,
            deps.knowledge_branch,
            state.get("injection_notice", ""),
        )
    }


def fallback(state, config=None):
    """Flow 7. No workflow branch applied."""

    conversation, deps = state["conversation"], deps_from(config)
    knowledge = chatbot._knowledge_answer(
        conversation, state["user_message"], deps.knowledge_branch
    )
    return {
        "reply": chatbot._compose(
            build_supplementary_response(
                conversation, state["user_message"], knowledge_text=knowledge
            ),
            "" if knowledge else MENU_PROMPT_RESPONSE,
        )
    }


def output_guardrail(state, config=None):
    """Moderate the reply and record the turn, exactly as the chain does."""

    return {
        "reply": chatbot._finalize(
            state["conversation"],
            state["messages"],
            state["original_message"],
            state["reply"],
        )
    }


def record_escalation(state, config=None):
    """An escalated turn is recorded without output moderation, as in the chain."""

    chatbot._record_turn(
        state["messages"], state["original_message"], state["reply"]
    )
    return {}
