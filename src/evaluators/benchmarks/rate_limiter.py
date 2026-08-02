"""Adaptive concurrency control and rate-limit-aware retries for benchmark runs.

The benchmark drives an HTTP API that itself makes several upstream LLM calls per
turn, so the practical throughput ceiling is the LLM provider's rate limit rather
than anything local. This module keeps the suite inside that ceiling without
hard-coding a number for it:

- `AdaptiveConcurrency` starts at a deliberately small parallelism and uses AIMD
  (additive increase, multiplicative decrease) to find a level the provider
  tolerates. Rate-limit signals shrink it immediately; sustained success grows it
  by one at a time.
- `retry_with_backoff` retries transient failures with exponential backoff plus
  full jitter, honouring a server-supplied `Retry-After` when one is available.

Neither helper ever raises for a rate limit: exhausting the retry budget returns
control to the caller, which records the case as permanently failed and moves on.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


RATE_LIMIT_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})
RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "overloaded",
    "capacity",
    "429",
    "quota",
)


def _status_code_of(error: BaseException) -> int | None:
    """Pull an HTTP status code off any of the shapes our clients raise."""

    for attribute in ("status_code", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_rate_limit_error(error: BaseException) -> bool:
    """Report whether an exception represents provider throttling or overload."""

    status_code = _status_code_of(error)
    if status_code in RATE_LIMIT_STATUS_CODES:
        return True
    if type(error).__name__ in {"RateLimitError", "InternalServerError", "APIStatusError"}:
        return True
    text = str(error).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def is_retryable_error(error: BaseException) -> bool:
    """Report whether retrying the same call could plausibly succeed."""

    if is_rate_limit_error(error):
        return True
    if type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "TimeoutException",
    }:
        return True
    status_code = _status_code_of(error)
    return status_code is not None and status_code >= 500


def retry_after_seconds(error: BaseException) -> float | None:
    """Return a server-requested wait, when the response carries one."""

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for header_name in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        raw_value = headers.get(header_name) if hasattr(headers, "get") else None
        if raw_value is None:
            continue
        try:
            seconds = float(str(raw_value).rstrip("s"))
        except ValueError:
            continue
        if seconds >= 0:
            return seconds
    return None


@dataclass
class RetryPolicy:
    """Bounded exponential backoff with full jitter."""

    max_attempts: int = 6
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0

    def delay_for(self, attempt: int, *, server_hint: float | None = None) -> float:
        """Return the sleep before `attempt` (1-based), preferring a server hint."""

        ceiling = min(self.base_delay * (self.multiplier ** (attempt - 1)), self.max_delay)
        # Full jitter spreads a thundering herd of retries across the window
        # instead of replaying the same burst that caused the throttle.
        jittered = random.uniform(0.0, ceiling)
        if server_hint is not None:
            return min(max(server_hint, jittered), self.max_delay)
        return jittered


@dataclass
class RateLimitStats:
    """Counters describing how hard the run pushed against the provider."""

    attempts: int = 0
    rate_limit_hits: int = 0
    retries: int = 0
    permanent_failures: int = 0
    concurrency_increases: int = 0
    concurrency_decreases: int = 0
    total_backoff_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "rate_limit_hits": self.rate_limit_hits,
            "retries": self.retries,
            "permanent_failures": self.permanent_failures,
            "concurrency_increases": self.concurrency_increases,
            "concurrency_decreases": self.concurrency_decreases,
            "total_backoff_seconds": round(self.total_backoff_seconds, 2),
        }


@dataclass
class AdaptiveConcurrency:
    """AIMD concurrency governor shared by every in-flight benchmark case.

    The governor owns a permit count rather than an `asyncio.Semaphore` because
    the limit has to shrink while requests are already in flight. `acquire`
    blocks until the live permit count is below the current limit, so lowering
    the limit takes effect on the next acquisition without cancelling work.
    """

    min_concurrency: int = 1
    max_concurrency: int = 8
    start_concurrency: int = 2
    increase_after_successes: int = 12
    decrease_factor: float = 0.5
    stats: RateLimitStats = field(default_factory=RateLimitStats)

    def __post_init__(self) -> None:
        self._limit = max(self.min_concurrency, min(self.start_concurrency, self.max_concurrency))
        self._in_flight = 0
        self._consecutive_successes = 0
        self._cooldown_until = 0.0
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    async def acquire(self) -> None:
        """Wait for a permit, respecting any cooldown from a recent throttle."""

        async with self._condition:
            await self._condition.wait_for(lambda: self._in_flight < self._limit)
            self._in_flight += 1
        cooldown = self._cooldown_until - asyncio.get_running_loop().time()
        if cooldown > 0:
            await asyncio.sleep(cooldown)

    async def release(self) -> None:
        async with self._condition:
            self._in_flight -= 1
            self._condition.notify_all()

    async def record_success(self) -> None:
        """Grow the limit by one after a sustained run of clean responses."""

        async with self._condition:
            self._consecutive_successes += 1
            if (
                self._consecutive_successes >= self.increase_after_successes
                and self._limit < self.max_concurrency
            ):
                self._limit += 1
                self._consecutive_successes = 0
                self.stats.concurrency_increases += 1
                self._condition.notify_all()

    async def record_rate_limit(self, *, cooldown: float = 0.0) -> None:
        """Halve the limit and pause new acquisitions for `cooldown` seconds.

        Counting the throttle itself is the retry helper's job — it sees every
        rate-limited attempt, including those made without a governor.
        """

        async with self._condition:
            self._consecutive_successes = 0
            new_limit = max(self.min_concurrency, int(self._limit * self.decrease_factor))
            if new_limit < self._limit:
                self._limit = new_limit
                self.stats.concurrency_decreases += 1
            if cooldown > 0:
                self._cooldown_until = max(
                    self._cooldown_until,
                    asyncio.get_running_loop().time() + cooldown,
                )


async def retry_with_backoff(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    stats: RateLimitStats,
    governor: AdaptiveConcurrency | None = None,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> tuple[T | None, BaseException | None]:
    """Run `operation`, retrying transient failures with jittered backoff.

    Returns `(result, None)` on success and `(None, last_error)` once the budget
    is exhausted or the failure is not retryable. It never raises, so one
    unrecoverable case can never abort the surrounding evaluation.
    """

    last_error: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        stats.attempts += 1
        try:
            return await operation(), None
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            # Interruption is the operator's decision, not a failed call, and
            # must stop the run rather than be booked as a permanent failure.
            raise
        except BaseException as error:  # Recorded and reported, never propagated.
            last_error = error
            if not is_retryable_error(error) or attempt == policy.max_attempts:
                break
            delay = policy.delay_for(attempt, server_hint=retry_after_seconds(error))
            if is_rate_limit_error(error):
                stats.rate_limit_hits += 1
                if governor is not None:
                    await governor.record_rate_limit(cooldown=delay)
            stats.retries += 1
            stats.total_backoff_seconds += delay
            if on_retry is not None:
                on_retry(attempt, delay, error)
            await asyncio.sleep(delay)

    stats.permanent_failures += 1
    return None, last_error


def run_blocking_with_backoff(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    stats: RateLimitStats,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> tuple[T | None, BaseException | None]:
    """Synchronous sibling of `retry_with_backoff` for blocking SDK clients."""

    import time

    last_error: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        stats.attempts += 1
        try:
            return operation(), None
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # Recorded and reported, never propagated.
            last_error = error
            if not is_retryable_error(error) or attempt == policy.max_attempts:
                break
            delay = policy.delay_for(attempt, server_hint=retry_after_seconds(error))
            if is_rate_limit_error(error):
                stats.rate_limit_hits += 1
            stats.retries += 1
            stats.total_backoff_seconds += delay
            if on_retry is not None:
                on_retry(attempt, delay, error)
            time.sleep(delay)

    stats.permanent_failures += 1
    return None, last_error
