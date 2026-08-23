"""Run one turn through the graph, with the chain's call signature.

`run_turn` deliberately mirrors `get_chatbot_response`: same arguments, same
return type, same in-place mutation of `ConversationState`. That is what lets
`/chat` dispatch to either orchestrator on a flag and lets the equivalence
harness compare them without adapting one side to the other — an adapter would
be a place for a difference to hide.
"""

from __future__ import annotations

try:
    from ..models import ChatMessage, ConversationState
    from .build import build_graph
    from .deps import TurnDeps
except ImportError:  # pragma: no cover - allows running as a script
    from models import ChatMessage, ConversationState
    from graph.build import build_graph
    from graph.deps import TurnDeps

# Compiled once. Construction walks the node and edge definitions, and doing it
# per turn would add latency the chain does not pay.
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_turn(
    messages: list[ChatMessage],
    prompt: str,
    client=None,
    state: ConversationState = None,
    visit_repository=None,
    knowledge_branch=None,
) -> str:
    """Answer one turn through the graph. Signature matches the chain's."""

    result = get_graph().invoke(
        {
            "session_id": state.session_id,
            "user_message": prompt,
            "original_message": prompt,
            "conversation": state,
            "messages": messages,
            "reply": "",
            "branch": "",
            "injection_notice": "",
            "aside_reply": "",
        },
        config={
            "configurable": {
                "deps": TurnDeps(
                    client=client,
                    visit_repository=visit_repository,
                    knowledge_branch=knowledge_branch,
                )
            }
        },
    )
    return result["reply"]
