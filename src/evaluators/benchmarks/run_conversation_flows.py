"""Run the multi-turn conversation-flow benchmark against the assistant API.

This measures a different failure mode from the single-turn `Scenarios` suite.
A scenario asks whether one reply was right; a flow asks whether a whole session
held together — whether what the patient said in turn two is still true in turn
ten, whether a correction fully landed, and whether an escalation stayed
escalated. The two are reported side by side and never averaged, because a high
single-turn intent score tells you nothing about session integrity.
"""

import argparse
import asyncio
import json
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
    from src.evaluators.benchmarks.conversation_evaluator import evaluate_conversation
    from src.evaluators.benchmarks.conversation_loader import load_conversation_flows
    from src.evaluators.benchmarks.conversation_runner import run_conversation_batch
    from src.evaluators.benchmarks.rate_limiter import (
        AdaptiveConcurrency,
        RateLimitStats,
        RetryPolicy,
    )
else:
    from ...chatbot import load_api_key
    from .checkpoint import CheckpointStore, batched
    from .conversation_evaluator import evaluate_conversation
    from .conversation_loader import load_conversation_flows
    from .conversation_runner import run_conversation_batch
    from .rate_limiter import AdaptiveConcurrency, RateLimitStats, RetryPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILE = PROJECT_ROOT / "src" / "evaluators" / "healthcare_assistant_benchmark.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "conversations"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multi-turn conversation benchmark.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--conv-id", action="append", help="Run one Conv ID; may be repeated")
    parser.add_argument("--limit", type=int, help="Run only the first N conversations")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--min-concurrency", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--start-concurrency", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--turn-delay", type=float, default=0.2)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--scenario-summary",
        type=Path,
        help="A benchmark_summary_*.json from the Scenarios suite, reported alongside",
    )
    return parser


def _load_completed(store: CheckpointStore) -> dict[str, dict]:
    """Checkpoint records are keyed by Conv ID rather than scenario test ID."""

    if not store.path.exists():
        return {}
    completed: dict[str, dict] = {}
    for line in store.path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        conv_id = record.get("flow", {}).get("conv_id")
        if conv_id:
            completed[str(conv_id)] = record
    return completed


async def _evaluate_all(args, flows, judge_client, store: CheckpointStore) -> list[dict]:
    completed = _load_completed(store)
    pending = [flow for flow in flows if flow.conv_id not in completed]
    if completed:
        print(
            f"Resuming: {len(completed)} conversations already evaluated, "
            f"{len(pending)} remaining.",
            flush=True,
        )

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
        for number, batch in enumerate(batched(pending, args.batch_size), start=1):
            runs = await run_conversation_batch(
                client,
                batch,
                governor=governor,
                policy=policy,
                stats=stats,
                turn_delay=args.turn_delay,
            )
            results = []
            for run in runs:
                result = await asyncio.to_thread(
                    evaluate_conversation,
                    run,
                    judge_client=judge_client,
                    judge_model=args.judge_model,
                    retry_policy=policy,
                    retry_stats=stats,
                )
                results.append(result)
                print(
                    f"[{run.flow.conv_id}] {run.flow.category} / {run.flow.name} "
                    f"({len(run.turns)} turns) ... {result['status']}",
                    flush=True,
                )
            store.append_batch(results)
            print(
                f"-- batch {number} checkpointed (concurrency={governor.limit}, "
                f"rate_limit_hits={stats.rate_limit_hits})",
                flush=True,
            )

    print(f"\nRate-limit behaviour: {stats.to_dict()}", flush=True)
    completed = _load_completed(store)
    return [completed[flow.conv_id] for flow in flows if flow.conv_id in completed]


def build_summary(results: list[dict]) -> dict:
    """Aggregate conversation verdicts and the three session-integrity scores."""

    def rate(name: str) -> dict:
        applicable = [
            result for result in results
            if result.get(name, {}).get("applicable", True) and result["status"] != "ERROR"
        ]
        passed = sum(1 for result in applicable if result.get(name, {}).get("score") == "PASS")
        return {
            "passed": passed,
            "applicable": len(applicable),
            "rate": round(100 * passed / len(applicable), 1) if applicable else None,
        }

    passed = sum(1 for result in results if result["status"] == "PASS")
    # Graded turn expectations, aggregated across every turn of every flow, so
    # a run that improves without flipping a whole conversation still shows it.
    met = sum(result.get("turn_expectations_met", 0) for result in results)
    total_expectations = sum(result.get("turn_expectations_total", 0) for result in results)

    categories: dict[str, dict] = {}
    for result in results:
        category = result["flow"]["category"]
        entry = categories.setdefault(category, {"total": 0, "pass": 0})
        entry["total"] += 1
        entry["pass"] += result["status"] == "PASS"

    return {
        "total": len(results),
        "pass": passed,
        "fail": sum(1 for result in results if result["status"] == "FAIL"),
        "error": sum(1 for result in results if result["status"] == "ERROR"),
        "pass_rate": round(100 * passed / len(results), 1) if results else 0.0,
        "turn_expectations": {
            "met": met,
            "total": total_expectations,
            "rate": round(100 * met / total_expectations, 1) if total_expectations else None,
        },
        "state_persistence": rate("state_persistence"),
        "recovery": rate("recovery"),
        "tone_and_safety": rate("tone_and_safety"),
        "categories": categories,
    }


def print_report(results: list[dict], scenario_summary: dict | None) -> None:
    border = "=" * 78
    print(f"\n{border}")
    print("  CONVERSATION-FLOW BENCHMARK — full-session integrity")
    print("  A conversation fails only on a turn error, corrupted state, or an")
    print("  emergency/injection turn that did not override the normal flow.")
    print(border)
    for result in results:
        flow = result["flow"]
        rate = result.get("turn_expectation_rate")
        graded = f"{rate:>5.1f}%" if rate is not None else "    — "
        print(
            f"  {flow['conv_id']:>8}  {result['status']:<5} turns {graded}  "
            f"{flow['name'][:38]:<38}"
        )
        print(f"            {result['reason'][:150]}")

    summary = build_summary(results)
    expectations = summary["turn_expectations"]
    print(f"\n{border}")
    print(f"  Conversations passed: {summary['pass']}/{summary['total']}  ({summary['pass_rate']}%)")
    print(
        f"    {'Turn expectations':<26} {expectations['met']}/{expectations['total']}"
        f"  ({expectations['rate']}%)   [diagnostic, does not fail a conversation]"
    )
    for name, label in (
        ("state_persistence", "State persistence"),
        ("recovery", "Recovery correctness"),
        ("tone_and_safety", "Tone/safety consistency"),
    ):
        entry = summary[name]
        rate = f"{entry['rate']}%" if entry["rate"] is not None else "n/a"
        print(f"    {label:<26} {entry['passed']}/{entry['applicable']}  ({rate})")

    print("\n  By flow category:")
    for category, entry in summary["categories"].items():
        print(f"    {category:<28} {entry['pass']}/{entry['total']}")

    if scenario_summary:
        print(f"\n{border}")
        print("  SIDE BY SIDE — these measure different failure modes, not one score")
        print(border)
        print(
            f"    Single-turn scenarios (Scenarios sheet): "
            f"{scenario_summary['pass']}/{scenario_summary['total']} "
            f"({scenario_summary['overall_pass_rate']}%)"
        )
        print(
            f"    Full-session flows (Conversation Flows): "
            f"{summary['pass']}/{summary['total']} ({summary['pass_rate']}%)"
        )
        print(
            "    The first measures per-turn understanding; the second measures whether\n"
            "    a session stays coherent across corrections, interruptions and escalations."
        )


def main() -> int:
    args = _build_parser().parse_args()
    flows = load_conversation_flows(args.file)
    if args.conv_id:
        wanted = {value.upper() for value in args.conv_id}
        flows = [flow for flow in flows if flow.conv_id.upper() in wanted]
    if args.limit:
        flows = flows[: args.limit]
    if not flows:
        print("No conversation flows matched the supplied filters.")
        return 2

    turn_count = sum(len(flow.turns) for flow in flows)
    print(f"Loaded {len(flows)} conversations ({turn_count} turns).", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (args.output_dir / "checkpoint.jsonl")
    store = CheckpointStore(checkpoint_path)
    if args.fresh:
        store.clear()

    judge_client: OpenAI | None = None
    if not args.no_judge:
        try:
            os.environ["OPENAI_API_KEY"] = load_api_key()
            judge_client = OpenAI(max_retries=0)
        except RuntimeError as exc:
            print(f"Warning: session judge disabled: {exc}", flush=True)

    results = asyncio.run(_evaluate_all(args, flows, judge_client, store))

    scenario_summary = None
    if args.scenario_summary and args.scenario_summary.exists():
        scenario_summary = json.loads(args.scenario_summary.read_text(encoding="utf-8"))

    summary = build_summary(results)
    report_path = args.output_dir / "conversation_flow_report.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "scenario_summary": scenario_summary,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print_report(results, scenario_summary)
    print(f"\nReport: {report_path}")
    return 1 if summary["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
