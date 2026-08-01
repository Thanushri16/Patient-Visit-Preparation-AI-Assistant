"""Command-line entry point for the Excel-driven healthcare assistant benchmark.

The suite is rate-limit aware by construction. Scenarios are processed in small
batches whose parallelism is chosen at runtime by an AIMD governor rather than
fixed up front, every API call retries throttling with jittered exponential
backoff, and each finished batch is checkpointed so an interrupted run resumes
where it stopped. A case that exhausts its retry budget is recorded as ERROR and
skipped; the run always continues to the full report.
"""

import argparse
import asyncio
import os
from pathlib import Path
import sys

import httpx
from openai import OpenAI

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.chatbot import load_api_key
    from src.evaluators.benchmarks.checkpoint import CheckpointStore, batched
    from src.evaluators.benchmarks.evaluator import evaluate_run
    from src.evaluators.benchmarks.rate_limiter import (
        AdaptiveConcurrency,
        RateLimitStats,
        RetryPolicy,
    )
    from src.evaluators.benchmarks.report_generator import print_report, write_reports
    from src.evaluators.benchmarks.test_loader import load_scenarios, split_scenarios
    from src.evaluators.benchmarks.test_runner import run_scenario_batch
else:
    from ...chatbot import load_api_key
    from .checkpoint import CheckpointStore, batched
    from .evaluator import evaluate_run
    from .rate_limiter import AdaptiveConcurrency, RateLimitStats, RetryPolicy
    from .report_generator import print_report, write_reports
    from .test_loader import load_scenarios, split_scenarios
    from .test_runner import run_scenario_batch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILE = PROJECT_ROOT / "src" / "evaluators" / "healthcare_assistant_benchmark_210.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "benchmarks"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "checkpoint.jsonl"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the healthcare assistant Excel benchmark.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="Benchmark .xlsx file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--category", help="Run one exact category name")
    parser.add_argument("--test-id", action="append", help="Run a test ID; may be repeated")
    parser.add_argument("--batch-size", type=int, default=10, help="Scenarios per checkpoint")
    parser.add_argument("--min-concurrency", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--start-concurrency", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--batch-pause", type=float, default=0.0, help="Seconds between batches")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--turn-delay", type=float, default=0.3)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard an existing checkpoint and evaluate every scenario again",
    )
    parser.add_argument("--version", default="0.2.0")
    return parser


async def _evaluate_all(args, scenarios, judge_client, checkpoint: CheckpointStore) -> list[dict]:
    """Run every outstanding scenario batch-by-batch, checkpointing as it goes."""

    completed = checkpoint.load()
    pending = [scenario for scenario in scenarios if scenario.test_id not in completed]
    if completed:
        print(
            f"Resuming from checkpoint: {len(completed)} already evaluated, "
            f"{len(pending)} remaining.",
            flush=True,
        )

    # The governor and the retry helper share one counter set so the end-of-run
    # summary reports throttling and concurrency changes together.
    stats = RateLimitStats()
    governor = AdaptiveConcurrency(
        min_concurrency=max(1, args.min_concurrency),
        max_concurrency=max(1, args.max_concurrency),
        start_concurrency=max(1, args.start_concurrency),
        stats=stats,
    )
    policy = RetryPolicy(max_attempts=max(1, args.max_retries))

    limits = httpx.Limits(
        max_connections=governor.max_concurrency,
        max_keepalive_connections=governor.max_concurrency,
    )
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=httpx.Timeout(args.timeout),
        limits=limits,
    ) as client:
        for batch_number, batch in enumerate(batched(pending, args.batch_size), start=1):
            runs = await run_scenario_batch(
                client,
                batch,
                governor=governor,
                policy=policy,
                stats=stats,
                turn_delay=args.turn_delay,
            )

            batch_results = []
            for run in runs:
                # The judge is a blocking SDK call, so it runs off the event loop
                # with its own retry budget; a judge that stays rate-limited marks
                # the single case ERROR instead of stopping the suite.
                result = await asyncio.to_thread(
                    evaluate_run,
                    run,
                    judge_client=judge_client,
                    judge_model=args.judge_model,
                    retry_policy=policy,
                    retry_stats=stats,
                )
                batch_results.append(result)
                scenario = run.scenario
                print(
                    f"[{scenario.test_id}] {scenario.category} / "
                    f"{scenario.subcategory} ... {result['status']}",
                    flush=True,
                )

            checkpoint.append_batch(batch_results)
            print(
                f"-- batch {batch_number} checkpointed "
                f"({len(batch_results)} cases, concurrency={governor.limit}, "
                f"rate_limit_hits={stats.rate_limit_hits})",
                flush=True,
            )
            if args.batch_pause > 0:
                await asyncio.sleep(args.batch_pause)

    print(f"\nRate-limit behaviour: {stats.to_dict()}", flush=True)

    # Report in spreadsheet order regardless of the order batches completed.
    completed = checkpoint.load()
    return [completed[s.test_id] for s in scenarios if s.test_id in completed]


def main() -> int:
    """Load, execute, evaluate, report, and return nonzero on errors only."""

    args = _build_parser().parse_args()
    scenarios = load_scenarios(args.file)
    if args.category:
        scenarios = [scenario for scenario in scenarios if scenario.category == args.category]
    if args.test_id:
        wanted = {test_id.upper() for test_id in args.test_id}
        scenarios = [scenario for scenario in scenarios if scenario.test_id.upper() in wanted]
    if not scenarios:
        print("No benchmark scenarios matched the supplied filters.")
        return 2

    singles, multi_turn = split_scenarios(scenarios)
    print(
        f"Loaded {len(scenarios)} scenarios "
        f"({len(singles)} single-turn, {len(multi_turn)} multi-turn).",
        flush=True,
    )

    checkpoint = CheckpointStore(args.checkpoint)
    if args.fresh:
        checkpoint.clear()

    judge_client: OpenAI | None = None
    if not args.no_judge:
        try:
            api_key = load_api_key()
            os.environ["OPENAI_API_KEY"] = api_key
            judge_client = OpenAI(max_retries=0)
        except RuntimeError as exc:
            print(f"Warning: LLM judge disabled: {exc}", flush=True)

    results = asyncio.run(_evaluate_all(args, scenarios, judge_client, checkpoint))

    summary, paths = write_reports(
        results,
        output_dir=args.output_dir,
        version=args.version,
    )
    print_report(summary, results)
    print("\nReports:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 1 if summary["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
