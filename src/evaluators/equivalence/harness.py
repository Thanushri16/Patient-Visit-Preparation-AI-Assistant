"""Compare two orchestrators turn by turn.

Part B re-expresses the chain as a LangGraph state graph and must not change
behaviour. That claim is only worth as much as the thing checking it, so this
harness exists before the graph does, and is first verified by running the chain
against itself — where every field must match exactly, because it is the same
code twice.

What it compares, and why each matters:

*   **The reply text**, byte for byte. Menu prompts, refusals and fallbacks are
    constants; if one differs, a code path diverged.
*   **The conversation state**, field by field, as `model_dump(mode="json")`.
    A reply can be identical while the record behind it is wrong, and the record
    is what the visit summary is built from.
*   **The RAG block**, separately from the rest of state, because it is the
    newest and least exercised part.

Divergence is reported per field rather than as a boolean. "The two differ" is
not actionable; "state.visit_data.symptom_onset differs on turn 3" is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Sequence


class Mode(StrEnum):
    """How strictly a conversation is compared.

    The chain is not reproducible against itself. Measured over three runs of
    eight conversations, one to two diverged every time — always the extractor,
    always on `visit_data.visit_reason`, cascading into the reply because the
    chain acknowledges what it captured. Roughly 15% of conversations.

    That makes "byte-identical replies on every turn" unachievable as a
    migration criterion: a perfect reimplementation would fail it, and a real
    bug would be indistinguishable from extractor jitter. So the criterion is
    split by whether a model is in the loop.
    """

    # No model call on this path: moderation, menu routing, the never-route
    # policy ladder, state recall, summary rendering from state. Every field is
    # compared, and any difference is a defect.
    STRICT = "strict"

    # A model decides something here — intent, extraction, follow-up wording,
    # confirmation, grounded generation. Free text is not compared; the
    # decisions it drives are.
    STRUCTURAL = "structural"


# What a STRUCTURAL comparison checks. These are the fields a migration bug
# would move and extractor jitter would not: which workflow the turn entered,
# what phase it left the conversation in, whether safety fired, what the
# knowledge branch decided and which documents it cited.
#
# Deliberately short. Every field NOT listed here is a field the loose mode
# cannot catch, so this list is the honest statement of what Part B's weaker
# criterion actually verifies.
STRUCTURAL_FIELDS = (
    "state.phase",
    "state.workflow",
    "state.emergency_detected",
    "state.confirmed",
    "state.rag.status",
    "state.rag.source",
)


@dataclass
class TurnDivergence:
    """One field that differs on one turn."""

    turn: int
    message: str
    field: str
    left: Any
    right: Any

    def describe(self) -> str:
        return (
            f"turn {self.turn} ({self.message[:40]!r}) {self.field}: "
            f"{self.left!r} != {self.right!r}"
        )


@dataclass
class ConversationComparison:
    """Every divergence across one replayed conversation."""

    conversation_id: str
    mode: Mode = Mode.STRICT
    turns: int = 0
    divergences: list[TurnDivergence] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        return not self.divergences


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested state into dotted paths so a diff names the exact field."""

    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(value, list):
        flat = {}
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
        return flat
    return {prefix: value}


# Fields that legitimately differ between two runs of the same code and say
# nothing about behaviour. Kept deliberately short: every entry here is a thing
# the harness can no longer catch, so an over-long list quietly turns a strict
# comparison into a loose one.
VOLATILE_FIELDS = (
    # Each side is replayed under its own session id so their state cannot
    # interfere, so this one differs by construction.
    "session_id",
    "visit_id",
    "persisted_at",
    "rag.retrieval_latency_ms",
    "rag.total_latency_ms",
)


def _is_volatile(path: str) -> bool:
    return any(path == f or path.endswith(f".{f}") for f in VOLATILE_FIELDS)


def compare_turn(
    turn: int, message: str, left: dict, right: dict, mode: Mode = Mode.STRICT
) -> list[TurnDivergence]:
    """Compare one turn between two orchestrators, at the given strictness."""

    divergences: list[TurnDivergence] = []
    left_state = _flatten(left.get("state") or {}, "state")
    right_state = _flatten(right.get("state") or {}, "state")

    if mode is Mode.STRUCTURAL:
        # Citations are compared as the set of documents cited, not as text: a
        # migration must not change which sources back an answer, but the
        # sentence around them is the model's.
        left_cites = sorted(
            v for k, v in left_state.items() if k.endswith(".document_id")
        )
        right_cites = sorted(
            v for k, v in right_state.items() if k.endswith(".document_id")
        )
        if left_cites != right_cites:
            divergences.append(
                TurnDivergence(turn, message, "cited_documents", left_cites, right_cites)
            )
        # An answer appearing on one side and not the other is structural even
        # when the wording is not compared.
        if bool(left.get("reply")) != bool(right.get("reply")):
            divergences.append(
                TurnDivergence(turn, message, "reply_present", bool(left.get("reply")),
                               bool(right.get("reply")))
            )
        for path in STRUCTURAL_FIELDS:
            if left_state.get(path) != right_state.get(path):
                divergences.append(
                    TurnDivergence(
                        turn, message, path, left_state.get(path), right_state.get(path)
                    )
                )
        return divergences

    if left.get("reply") != right.get("reply"):
        divergences.append(
            TurnDivergence(turn, message, "reply", left.get("reply"), right.get("reply"))
        )
    for path in sorted(set(left_state) | set(right_state)):
        if _is_volatile(path):
            continue
        if left_state.get(path) != right_state.get(path):
            divergences.append(
                TurnDivergence(
                    turn, message, path, left_state.get(path), right_state.get(path)
                )
            )
    return divergences


def compare_conversation(
    conversation_id: str,
    messages: Sequence[str],
    run_left: Callable[[str, str], dict],
    run_right: Callable[[str, str], dict],
    mode: Mode = Mode.STRICT,
) -> ConversationComparison:
    """Replay one conversation through both orchestrators, in order.

    Each side gets its own session id so their state cannot interfere, and the
    turns are replayed sequentially because a session is a sequence — comparing
    isolated turns would miss exactly the state-carryover bugs a migration
    causes.
    """

    comparison = ConversationComparison(conversation_id=conversation_id, mode=mode)
    left_session = f"{conversation_id}-left"
    right_session = f"{conversation_id}-right"

    for index, message in enumerate(messages, start=1):
        left = run_left(left_session, message)
        right = run_right(right_session, message)
        comparison.turns += 1
        comparison.divergences.extend(
            compare_turn(index, message, left, right, mode)
        )
    return comparison


@dataclass
class EquivalenceReport:
    """The result of replaying a set of conversations through both sides."""

    comparisons: list[ConversationComparison] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        return all(c.equivalent for c in self.comparisons)

    def summary(self) -> dict[str, object]:
        diverged = [c for c in self.comparisons if not c.equivalent]
        by_field: dict[str, int] = {}
        for comparison in diverged:
            for divergence in comparison.divergences:
                by_field[divergence.field] = by_field.get(divergence.field, 0) + 1
        return {
            "conversations": len(self.comparisons),
            "strict": sum(1 for c in self.comparisons if c.mode is Mode.STRICT),
            "structural": sum(1 for c in self.comparisons if c.mode is Mode.STRUCTURAL),
            "turns": sum(c.turns for c in self.comparisons),
            "equivalent": self.equivalent,
            "diverged_conversations": [c.conversation_id for c in diverged],
            "divergences_by_field": dict(sorted(by_field.items())),
            "examples": [
                d.describe() for c in diverged for d in c.divergences[:3]
            ][:20],
        }
