"""Execute benchmark conversations against the healthcare assistant HTTP API."""

import asyncio
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable
from uuid import uuid4

import httpx

from .rate_limiter import (
    AdaptiveConcurrency,
    RateLimitStats,
    RetryPolicy,
    retry_with_backoff,
)
from .test_loader import BenchmarkScenario


@dataclass
class TurnResult:
    """Captured input, output, timing, and error data for one API turn."""

    turn_number: int
    message: str
    response: dict[str, Any] | None
    status_code: int | None
    latency_ms: float
    error: str | None = None


@dataclass
class ScenarioRun:
    """All API turn results for one independently isolated benchmark scenario."""

    scenario: BenchmarkScenario
    session_id: str
    turns: list[TurnResult]

    @property
    def final_response(self) -> dict[str, Any]:
        for turn in reversed(self.turns):
            if turn.response is not None:
                return turn.response
        return {}

    @property
    def error(self) -> str | None:
        errors = [turn.error for turn in self.turns if turn.error]
        return "; ".join(errors) if errors else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "session_id": self.session_id,
            "turns": [asdict(turn) for turn in self.turns],
            "final_response": self.final_response,
            "error": self.error,
        }


ProgressCallback = Callable[[BenchmarkScenario, ScenarioRun], Awaitable[None] | None]


async def execute_scenario(
    client: httpx.AsyncClient,
    scenario: BenchmarkScenario,
    *,
    turn_delay: float = 0.5,
    governor: AdaptiveConcurrency | None = None,
    policy: RetryPolicy | None = None,
    stats: RateLimitStats | None = None,
) -> ScenarioRun:
    """Run one scenario with a fresh session shared only by its own turns.

    Every turn is issued through the retry helper, so an upstream rate limit
    costs a backoff rather than the scenario. Only a turn that exhausts its
    retry budget is recorded as an error, and the scenario then stops early
    because later turns depend on the state the failed turn would have set.
    """

    retry_policy = policy or RetryPolicy()
    retry_stats = stats if stats is not None else RateLimitStats()
    session_id = f"bench-{scenario.test_id.lower()}-{uuid4().hex[:8]}"
    turns: list[TurnResult] = []

    for index, message in enumerate(scenario.messages, start=1):
        started = perf_counter()

        async def post_turn() -> dict[str, Any]:
            response = await client.post(
                "/chat",
                json={"message": message, "session_id": session_id},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("The /chat response was not a JSON object.")
            return payload

        payload, error = await retry_with_backoff(
            post_turn,
            policy=retry_policy,
            stats=retry_stats,
            governor=governor,
        )
        latency_ms = round((perf_counter() - started) * 1_000, 2)

        if error is not None:
            turns.append(
                TurnResult(
                    turn_number=index,
                    message=message,
                    response=None,
                    status_code=getattr(getattr(error, "response", None), "status_code", None),
                    latency_ms=latency_ms,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            break

        if governor is not None:
            await governor.record_success()
        turns.append(
            TurnResult(
                turn_number=index,
                message=message,
                response=payload,
                status_code=200,
                latency_ms=latency_ms,
            )
        )
        if index < len(scenario.messages) and turn_delay > 0:
            await asyncio.sleep(turn_delay)

    return ScenarioRun(scenario=scenario, session_id=session_id, turns=turns)


async def run_scenario_batch(
    client: httpx.AsyncClient,
    scenarios: list[BenchmarkScenario],
    *,
    governor: AdaptiveConcurrency,
    policy: RetryPolicy,
    stats: RateLimitStats,
    turn_delay: float = 0.5,
    progress_callback: ProgressCallback | None = None,
) -> list[ScenarioRun]:
    """Run one batch, letting the governor decide how many run at once.

    Results come back in the order the scenarios were supplied even though the
    work interleaves, so the caller's checkpoint stays in spreadsheet order.
    """

    async def run_one(scenario: BenchmarkScenario) -> ScenarioRun:
        await governor.acquire()
        try:
            result = await execute_scenario(
                client,
                scenario,
                turn_delay=turn_delay,
                governor=governor,
                policy=policy,
                stats=stats,
            )
        finally:
            await governor.release()
        if progress_callback:
            callback_result = progress_callback(scenario, result)
            if asyncio.iscoroutine(callback_result):
                await callback_result
        return result

    tasks = [asyncio.create_task(run_one(scenario)) for scenario in scenarios]
    return list(await asyncio.gather(*tasks))


async def run_scenarios(
    scenarios: list[BenchmarkScenario],
    *,
    base_url: str,
    concurrency: int = 1,
    timeout_seconds: float = 30.0,
    turn_delay: float = 0.5,
    progress_callback: ProgressCallback | None = None,
) -> list[ScenarioRun]:
    """Run every scenario at a fixed concurrency, for single-shot or test use."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    governor = AdaptiveConcurrency(
        min_concurrency=concurrency,
        max_concurrency=concurrency,
        start_concurrency=concurrency,
    )
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(timeout_seconds),
        limits=limits,
    ) as client:
        return await run_scenario_batch(
            client,
            scenarios,
            governor=governor,
            policy=RetryPolicy(),
            stats=RateLimitStats(),
            turn_delay=turn_delay,
            progress_callback=progress_callback,
        )
