"""Graph construction: the chain's seven flows as nodes and conditional edges.

The point of this file is that the routing is *data*. In `get_chatbot_response`
the same control flow is a sequence of early returns spread over 290 lines, and
reading it means holding the whole function in your head. Here each branch is a
named edge that can be inspected, diagrammed, and tested on its own.

Nothing here decides anything. Every edge reads a decision a node already made
and stored in state — deliberately, because recomputing a decision at the edge
is the subtle way this migration would diverge: two calls to the same detector
can disagree once a model sits behind it, and the edge would then take a
different branch than the node believed it had taken.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

try:
    from . import nodes
    from .state import AssistantState
except ImportError:  # pragma: no cover - allows running as a script
    import nodes
    from state import AssistantState


def _after_guardrail(state) -> str:
    action = state.get("moderation_action", "")
    if action == "escalate":
        # Escalation is recorded but never passes the output filter, matching
        # the chain: the emergency text is a constant the guardrail would only
        # be able to damage.
        return "record_escalation"
    if action in {"block", "redirect"}:
        return "output_guardrail"
    return "preconversation_checks"


def _after_prechecks(state) -> str:
    return "direct_reply" if state.get("branch") == "aside" else "state_recall"


def _after_state_recall(state) -> str:
    return "output_guardrail" if state.get("branch") == "aside" else "route"


def _after_route(state) -> str:
    return state.get("branch", "fallback")


def build_graph(checkpointer=None):
    """Compile the assistant graph.

    `MemorySaver` matches today's in-memory sessions with their 15-minute TTL.
    `PostgresSaver` against the knowledge database would make sessions durable
    and resumable — most of FR-2 — but that is a behaviour change and belongs
    after Part B, not inside it.
    """

    graph = StateGraph(AssistantState)

    graph.add_node("input_guardrail", nodes.input_guardrail)
    graph.add_node("record_escalation", nodes.record_escalation)
    graph.add_node("preconversation_checks", nodes.preconversation_checks)
    graph.add_node("direct_reply", nodes.direct_reply)
    graph.add_node("state_recall", nodes.state_recall)
    graph.add_node("route", nodes.route)
    graph.add_node("route_handled", nodes.route_handled)
    graph.add_node("confirm", nodes.confirm)
    graph.add_node("collect", nodes.collect)
    graph.add_node("fallback", nodes.fallback)
    graph.add_node("output_guardrail", nodes.output_guardrail)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail", _after_guardrail,
        ["record_escalation", "output_guardrail", "preconversation_checks"],
    )
    graph.add_conditional_edges(
        "preconversation_checks", _after_prechecks, ["direct_reply", "state_recall"]
    )
    graph.add_conditional_edges(
        "state_recall", _after_state_recall, ["output_guardrail", "route"]
    )
    graph.add_conditional_edges(
        "route", _after_route,
        ["route_handled", "confirm", "collect", "fallback"],
    )

    for node in ("direct_reply", "route_handled", "confirm", "collect", "fallback"):
        graph.add_edge(node, "output_guardrail")
    graph.add_edge("record_escalation", END)
    graph.add_edge("output_guardrail", END)

    return graph.compile(checkpointer=checkpointer)
