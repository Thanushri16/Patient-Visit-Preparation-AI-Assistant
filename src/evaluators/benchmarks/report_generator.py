"""Aggregate benchmark scores and write console, full, summary, and failure reports."""

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any


def build_summary(results: list[dict[str, Any]], *, version: str) -> dict[str, Any]:
    """Calculate totals, rates, category breakdowns, and average judge dimensions."""

    counts = {status: sum(result["status"] == status for result in results) for status in ("PASS", "FAIL", "ERROR")}
    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        category_groups[result["scenario"]["category"]].append(result)

    categories: list[dict[str, Any]] = []
    for category, items in category_groups.items():
        passed = sum(item["status"] == "PASS" for item in items)
        failed = sum(item["status"] == "FAIL" for item in items)
        errors = sum(item["status"] == "ERROR" for item in items)
        categories.append(
            {
                "category": category,
                "total": len(items),
                "pass": passed,
                "fail": failed,
                "error": errors,
                "pass_rate": round(100 * passed / len(items), 1) if items else 0.0,
            }
        )

    judge_dimensions = (
        "intent_handled",
        "behavior_correct",
        "criteria_met",
        "safety_compliant",
        "tone_appropriate",
    )
    judged = [result["judge"] for result in results if isinstance(result.get("judge"), dict)]
    judge_averages = {
        dimension: round(sum(float(item.get(dimension, 0)) for item in judged) / len(judged), 4)
        if judged
        else None
        for dimension in judge_dimensions
    }

    return {
        "version": version,
        "generated_at": datetime.now().astimezone().isoformat(),
        "total": len(results),
        "pass": counts["PASS"],
        "fail": counts["FAIL"],
        "error": counts["ERROR"],
        "overall_pass_rate": round(100 * counts["PASS"] / len(results), 1) if results else 0.0,
        "judge_averages": judge_averages,
        "categories": categories,
    }


def print_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Print a compact report and actionable failure list without optional UI packages."""

    border = "═" * 67
    print(border)
    print(f"  BENCHMARK REPORT — Healthcare Assistant v{summary['version']}")
    print(f"  Date: {summary['generated_at']}")
    print(
        f"  Total: {summary['total']} | Pass: {summary['pass']} | "
        f"Fail: {summary['fail']} | Error: {summary['error']}"
    )
    print(f"  Overall Pass Rate: {summary['overall_pass_rate']:.1f}%")
    print(border)
    print("\nCategory Breakdown:")
    print(f"{'Category':44} {'Total':>5} {'Pass':>5} {'Fail':>5} {'Error':>5} {'Rate':>7}")
    print("-" * 78)
    for category in summary["categories"]:
        name = category["category"][:44]
        print(
            f"{name:44} {category['total']:>5} {category['pass']:>5} "
            f"{category['fail']:>5} {category['error']:>5} {category['pass_rate']:>6.1f}%"
        )

    failures = [result for result in results if result["status"] != "PASS"]
    print("\nFailed and Errored Tests:")
    if not failures:
        print("  None")
        return
    for result in failures:
        scenario = result["scenario"]
        final_response = result.get("final_response", {})
        judge = result.get("judge") or {}
        reason = result.get("error") or result.get("judge_error") or judge.get("reason") or "A required check failed."
        print(f"  {scenario['test_id']} | {scenario['category']} | {scenario['subcategory']} | {result['status']}")
        print(f"    Input: {scenario['user_message']}")
        print(f"    Expected: {scenario['expected_intent']} | Got: {final_response.get('intent', 'n/a')}")
        print(f"    Reason: {reason}")


def write_reports(
    results: list[dict[str, Any]],
    *,
    output_dir: Path,
    version: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Write timestamped full, aggregate, and failure-only JSON reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = build_summary(results, version=version)
    failures = [result for result in results if result["status"] != "PASS"]
    paths = {
        "results": output_dir / f"benchmark_results_{timestamp}.json",
        "summary": output_dir / f"benchmark_summary_{timestamp}.json",
        "failures": output_dir / f"benchmark_failures_{timestamp}.json",
    }
    paths["results"].write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["failures"].write_text(
        json.dumps({"summary": summary, "failures": failures}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary, paths
