"""Command-line entry point for the Excel-driven healthcare assistant benchmark."""

import argparse
import asyncio
import os
from pathlib import Path
import sys

from openai import OpenAI

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.chatbot import load_api_key
    from src.evaluators.benchmarks.evaluator import evaluate_run
    from src.evaluators.benchmarks.report_generator import print_report, write_reports
    from src.evaluators.benchmarks.test_loader import load_scenarios, split_scenarios
    from src.evaluators.benchmarks.test_runner import run_scenarios
else:
    from ...chatbot import load_api_key
    from .evaluator import evaluate_run
    from .report_generator import print_report, write_reports
    from .test_loader import load_scenarios, split_scenarios
    from .test_runner import run_scenarios


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILE = PROJECT_ROOT / "src" / "evaluators" / "healthcare_assistant_benchmark_210.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "benchmarks"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the healthcare assistant Excel benchmark.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="Benchmark .xlsx file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--category", help="Run one exact category name")
    parser.add_argument("--test-id", action="append", help="Run a test ID; may be repeated")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--turn-delay", type=float, default=0.5)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", default="0.1.0")
    return parser


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
    runs = asyncio.run(
        run_scenarios(
            scenarios,
            base_url=args.base_url,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout,
            turn_delay=args.turn_delay,
        )
    )

    judge_client: OpenAI | None = None
    if not args.no_judge:
        try:
            api_key = load_api_key()
            os.environ["OPENAI_API_KEY"] = api_key
            judge_client = OpenAI()
        except RuntimeError as exc:
            print(f"Warning: LLM judge disabled: {exc}", flush=True)

    results = []
    for run in runs:
        result = evaluate_run(
            run,
            judge_client=judge_client,
            judge_model=args.judge_model,
        )
        results.append(result)
        scenario = run.scenario
        print(
            f"[{scenario.test_id}] {scenario.category} / {scenario.subcategory} ... {result['status']}",
            flush=True,
        )

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
