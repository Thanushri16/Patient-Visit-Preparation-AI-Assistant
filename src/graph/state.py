"""Typed state for the LangGraph orchestration.

`ConversationState` is carried whole as a nested Pydantic model rather than
flattened into graph keys. Flattening would duplicate the source of truth and
force every existing module to be rewritten against a TypedDict — which is the
behaviour change Part B forbids, dressed up as a refactor.

Only serializable turn data lives here. The OpenAI client, the visit repository
and the knowledge branch are dependencies, passed through `RunnableConfig`, so
that a checkpointed state stays a record of the conversation rather than a
snapshot of the process that ran it.
"""

from __future__ import annotations

from typing import Any, TypedDict

try:
    from ..models import ChatMessage, ConversationState
except ImportError:  # pragma: no cover - allows running as a script
    from models import ChatMessage, ConversationState


class AssistantState(TypedDict, total=False):
    """One turn, as it moves through the graph."""

    # Turn I/O
    session_id: str
    user_message: str          # the prompt as the chain sees it, after sanitising
    original_message: str      # before injection stripping, for the record
    reply: str

    # The existing typed state, carried whole
    conversation: ConversationState
    messages: list[ChatMessage]

    # Decisions a node made, so an edge can read them without recomputing.
    # Recomputing would be the subtle way this migration diverges: two calls to
    # the same detector on the same message can disagree once a model is behind
    # it, and the edge would then take a different branch than the node assumed.
    moderation_action: str
    route_action: str
    branch: str
    injection_notice: str
    aside_reply: str

    # Dependencies, resolved once per turn from config.
    deps: dict[str, Any]
