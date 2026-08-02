"""Unit tests for adaptive concurrency, backoff, and checkpoint resumption."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluators.benchmarks.checkpoint import CheckpointStore, batched  # noqa: E402
from src.evaluators.benchmarks.rate_limiter import (  # noqa: E402
    AdaptiveConcurrency,
    RateLimitStats,
    RetryPolicy,
    is_rate_limit_error,
    is_retryable_error,
    retry_after_seconds,
    retry_with_backoff,
    run_blocking_with_backoff,
)


class FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class RateLimitError(Exception):
    def __init__(self, message="Rate limit reached", status_code=429, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = FakeResponse(status_code, headers)


class ErrorClassificationTests(unittest.TestCase):
    def test_throttling_and_overload_are_recognized(self):
        self.assertTrue(is_rate_limit_error(RateLimitError()))
        self.assertTrue(is_rate_limit_error(Exception("Error code: 529 overloaded")))
        self.assertTrue(is_rate_limit_error(RateLimitError(status_code=503)))

    def test_ordinary_failures_are_not_retried(self):
        self.assertFalse(is_rate_limit_error(ValueError("bad payload")))
        self.assertFalse(is_retryable_error(ValueError("bad payload")))

    def test_server_supplied_retry_after_is_honoured(self):
        error = RateLimitError(headers={"retry-after": "7"})

        self.assertEqual(retry_after_seconds(error), 7.0)
        self.assertIsNone(retry_after_seconds(ValueError("no response")))


class RetryPolicyTests(unittest.TestCase):
    def test_backoff_grows_but_stays_within_the_ceiling(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=10.0, multiplier=2.0)

        for attempt in range(1, 8):
            delay = policy.delay_for(attempt)
            self.assertGreaterEqual(delay, 0.0)
            self.assertLessEqual(delay, 10.0)

    def test_jitter_spreads_retries_across_the_window(self):
        policy = RetryPolicy(base_delay=8.0, max_delay=60.0)

        delays = {policy.delay_for(3) for _ in range(40)}

        # Identical delays would replay the same burst that caused the throttle.
        self.assertGreater(len(delays), 1)


class RetryWithBackoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_failure_is_retried_and_then_succeeds(self):
        attempts = []

        async def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise RateLimitError()
            return "ok"

        stats = RateLimitStats()
        governor = AdaptiveConcurrency(start_concurrency=4, max_concurrency=8)

        result, error = await retry_with_backoff(
            flaky,
            policy=RetryPolicy(base_delay=0.001, max_delay=0.002),
            stats=stats,
            governor=governor,
        )

        self.assertEqual(result, "ok")
        self.assertIsNone(error)
        self.assertEqual(stats.retries, 2)
        self.assertEqual(stats.rate_limit_hits, 2)
        self.assertLess(governor.limit, 4)

    async def test_exhausted_budget_returns_the_error_instead_of_raising(self):
        async def always_throttled():
            raise RateLimitError()

        stats = RateLimitStats()

        result, error = await retry_with_backoff(
            always_throttled,
            policy=RetryPolicy(max_attempts=3, base_delay=0.001, max_delay=0.002),
            stats=stats,
        )

        # One unrecoverable case must never abort the surrounding evaluation.
        self.assertIsNone(result)
        self.assertIsInstance(error, RateLimitError)
        self.assertEqual(stats.permanent_failures, 1)

    async def test_non_retryable_error_fails_on_the_first_attempt(self):
        calls = []

        async def bad_request():
            calls.append(1)
            raise ValueError("malformed")

        stats = RateLimitStats()

        _, error = await retry_with_backoff(
            bad_request,
            policy=RetryPolicy(max_attempts=5, base_delay=0.001),
            stats=stats,
        )

        self.assertEqual(len(calls), 1)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(stats.retries, 0)

    def test_blocking_variant_retries_and_reports(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 2:
                raise RateLimitError()
            return "judged"

        stats = RateLimitStats()

        result, error = run_blocking_with_backoff(
            flaky,
            policy=RetryPolicy(base_delay=0.001, max_delay=0.002),
            stats=stats,
        )

        self.assertEqual(result, "judged")
        self.assertIsNone(error)
        self.assertEqual(stats.retries, 1)


class AdaptiveConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_limit_grows_only_after_sustained_success(self):
        governor = AdaptiveConcurrency(
            start_concurrency=2,
            max_concurrency=5,
            increase_after_successes=3,
        )

        for _ in range(2):
            await governor.record_success()
        self.assertEqual(governor.limit, 2)

        await governor.record_success()
        self.assertEqual(governor.limit, 3)

    async def test_rate_limit_halves_the_limit_immediately(self):
        governor = AdaptiveConcurrency(start_concurrency=8, max_concurrency=8)

        await governor.record_rate_limit()

        self.assertEqual(governor.limit, 4)
        self.assertEqual(governor.stats.concurrency_decreases, 1)

    async def test_limit_never_falls_below_the_configured_floor(self):
        governor = AdaptiveConcurrency(
            min_concurrency=2, start_concurrency=3, max_concurrency=8
        )

        for _ in range(6):
            await governor.record_rate_limit()

        self.assertEqual(governor.limit, 2)

    async def test_acquire_blocks_beyond_the_current_limit(self):
        governor = AdaptiveConcurrency(start_concurrency=1, max_concurrency=1)
        await governor.acquire()

        blocked = asyncio.create_task(governor.acquire())
        await asyncio.sleep(0)
        self.assertFalse(blocked.done())

        await governor.release()
        await asyncio.wait_for(blocked, timeout=1)
        self.assertTrue(blocked.done())
        await governor.release()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "checkpoint.jsonl"

    def tearDown(self):
        self.directory.cleanup()

    def _result(self, test_id, status="PASS"):
        return {"scenario": {"test_id": test_id}, "status": status}

    def test_completed_cases_survive_a_restart(self):
        store = CheckpointStore(self.path)
        store.append_batch([self._result("TC-001"), self._result("TC-002")])

        resumed = CheckpointStore(self.path).load()

        self.assertEqual(sorted(resumed), ["TC-001", "TC-002"])

    def test_a_truncated_final_line_is_ignored_not_fatal(self):
        store = CheckpointStore(self.path)
        store.append_batch([self._result("TC-001")])
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"scenario": {"test_id": "TC-00')

        resumed = store.load()

        # A process killed mid-write loses that one record and re-runs it.
        self.assertEqual(list(resumed), ["TC-001"])

    def test_appending_preserves_earlier_batches(self):
        store = CheckpointStore(self.path)
        store.append_batch([self._result("TC-001")])
        store.append_batch([self._result("TC-002", status="FAIL")])

        resumed = store.load()

        self.assertEqual(len(resumed), 2)
        self.assertEqual(resumed["TC-002"]["status"], "FAIL")
        self.assertEqual(len(self.path.read_text().strip().splitlines()), 2)
        json.loads(self.path.read_text().splitlines()[0])

    def test_batches_cover_every_item_without_overlap(self):
        items = list(range(23))

        chunks = list(batched(items, 10))

        self.assertEqual([len(chunk) for chunk in chunks], [10, 10, 3])
        self.assertEqual([item for chunk in chunks for item in chunk], items)


if __name__ == "__main__":
    unittest.main()
