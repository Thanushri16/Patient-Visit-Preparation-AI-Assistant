"""Execute whole conversation flows against the assistant's HTTP API.

Turns within a flow are strictly sequential on a single session: each reply must
arrive before the next message is sent, because the point of these cases is what
the assistant remembers. Different flows are independent, so they run
concurrently under the same adaptive governor as the single-turn suite.
"""

import asyncio
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any
from uuid import uuid4

import httpx

from .conversation_loader import ConversationFlow, FlowTurn
from .rate_limiter import (
    AdaptiveConcurrency,
    RateLimitStats,
    RetryPolicy,
    retry_with_backoff,
)


@dataclass
class FlowTurnResult:
    """The request, the response, and the state snapshot for one turn."""

    turn: FlowTurn
    reply: str = ""
    intent: str = "unknown"
    state: dict[str, Any] = field(default_factory=dict)
    is_emergency: bool = False
    safety_triggered: bool = False
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def visit_data(self) -> dict[str, Any]:
        data = self.state.get("visit_data")
        return data if isinstance(data, dict) else {}

    @property
    def phase(self) -> str:
        return str(self.state.get("phase") or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn.to_dict(),
            "reply": self.reply,
            "intent": self.intent,
            "state": self.state,
            "is_emergency": self.is_emergency,
            "safety_triggered": self.safety_triggered,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


@dataclass
class ConversationRun:
    """Every turn result for one conversation, in the order they were sent."""

    flow: ConversationFlow
    session_id: str
    turns: list[FlowTurnResult]

    @property
    def error(self) -> str | None:
        errors = [
            f"turn {result.turn.number}: {result.error}"
            for result in self.turns
            if result.error
        ]
        return "; ".join(errors) if errors else None

    @property
    def final(self) -> FlowTurnResult | None:
        return self.turns[-1] if self.turns else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow": self.flow.to_dict(),
            "session_id": self.session_id,
            "turns": [result.to_dict() for result in self.turns],
            "error": self.error,
        }


async def execute_conversation(
    client: httpx.AsyncClient,
    flow: ConversationFlow,
    *,
    turn_delay: float,
    policy: RetryPolicy,
    stats: RateLimitStats,
    governor: AdaptiveConcurrency | None = None,
) -> ConversationRun:
    """Send every turn of one flow in order on a single session.

    A turn that exhausts its retry budget stops the conversation: later turns
    depend on state the failed turn would have produced, so continuing would
    measure a session that never existed.
    """

    session_id = f"conv-{flow.conv_id.lower()}-{uuid4().hex[:8]}"
    results: list[FlowTurnResult] = []

    for turn in flow.turns:
        started = perf_counter()

        async def post_turn(message: str = turn.message) -> dict[str, Any]:
            response = await client.post(
                "/chat", json={"message": message, "session_id": session_id}
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("The /chat response was not a JSON object.")
            return payload

        payload, error = await retry_with_backoff(
            post_turn, policy=policy, stats=stats, governor=governor
        )
        latency_ms = (perf_counter() - started) * 1_000

        if error is not None:
            results.append(
                FlowTurnResult(
                    turn=turn,
                    latency_ms=latency_ms,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            break

        if governor is not None:
            await governor.record_success()
        results.append(
            FlowTurnResult(
                turn=turn,
                reply=str(payload.get("reply") or ""),
                intent=str(payload.get("intent") or "unknown"),
                state=payload.get("state") if isinstance(payload.get("state"), dict) else {},
                is_emergency=payload.get("is_emergency") is True,
                safety_triggered=payload.get("safety_triggered") is True,
                latency_ms=latency_ms,
            )
        )
        if turn_delay > 0:
            await asyncio.sleep(turn_delay)

    return ConversationRun(flow=flow, session_id=session_id, turns=results)


async def run_conversation_batch(
    client: httpx.AsyncClient,
    flows: list[ConversationFlow],
    *,
    governor: AdaptiveConcurrency,
    policy: RetryPolicy,
    stats: RateLimitStats,
    turn_delay: float = 0.2,
) -> list[ConversationRun]:
    """Run a batch of flows concurrently, each internally sequential."""

    async def run_one(flow: ConversationFlow) -> ConversationRun:
        await governor.acquire()
        try:
            return await execute_conversation(
                client,
                flow,
                turn_delay=turn_delay,
                policy=policy,
                stats=stats,
                governor=governor,
            )
        finally:
            await governor.release()

    tasks = [asyncio.create_task(run_one(flow)) for flow in flows]
    return list(await asyncio.gather(*tasks))
