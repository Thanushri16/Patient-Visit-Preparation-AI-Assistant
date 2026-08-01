"""Execute benchmark conversations against the healthcare assistant HTTP API."""

import asyncio
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable
from uuid import uuid4

import httpx

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
) -> ScenarioRun:
    """Run one scenario with a fresh session shared only by its own turns."""

    session_id = f"bench-{scenario.test_id.lower()}-{uuid4().hex[:8]}"
    turns: list[TurnResult] = []
    for index, message in enumerate(scenario.messages, start=1):
        started = perf_counter()
        try:
            response = await client.post(
                "/chat",
                json={"message": message, "session_id": session_id},
            )
            latency_ms = (perf_counter() - started) * 1_000
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("The /chat response was not a JSON object.")
            turns.append(
                TurnResult(
                    turn_number=index,
                    message=message,
                    response=payload,
                    status_code=response.status_code,
                    latency_ms=round(latency_ms, 2),
                )
            )
        except (httpx.HTTPError, ValueError) as exc:
            turns.append(
                TurnResult(
                    turn_number=index,
                    message=message,
                    response=None,
                    status_code=getattr(getattr(exc, "response", None), "status_code", None),
                    latency_ms=round((perf_counter() - started) * 1_000, 2),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            break
        if index < len(scenario.messages) and turn_delay > 0:
            await asyncio.sleep(turn_delay)
    return ScenarioRun(scenario=scenario, session_id=session_id, turns=turns)


async def run_scenarios(
    scenarios: list[BenchmarkScenario],
    *,
    base_url: str,
    concurrency: int = 1,
    timeout_seconds: float = 30.0,
    turn_delay: float = 0.5,
    progress_callback: ProgressCallback | None = None,
) -> list[ScenarioRun]:
    """Run scenarios in spreadsheet order, with bounded cross-session concurrency."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(timeout_seconds),
        limits=limits,
    ) as client:
        async def run_one(scenario: BenchmarkScenario) -> ScenarioRun:
            async with semaphore:
                result = await execute_scenario(
                    client,
                    scenario,
                    turn_delay=turn_delay,
                )
                if progress_callback:
                    callback_result = progress_callback(scenario, result)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                return result

        tasks = [asyncio.create_task(run_one(scenario)) for scenario in scenarios]
        return list(await asyncio.gather(*tasks))
