"""Compare the Part C experiment arms (C5).

Reads the per-arm reports the matrix wrote and lays them side by side. Metrics
are recomputed from the stored per-case results rather than read from each
report's summary block, so the holdout slice comes out of the same runs the
matrix already paid for instead of a second matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluators.rag.dataset import load_cases  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "rag"

# Deterministic gates first: a strategy that answers more but refuses less
# safely is not an improvement, and reading the quality columns before the
# safety ones invites exactly that trade.
SAFETY = (
    "near_miss_resistance",
    "never_route_compliance",
    "wrong_document_grounding",
    "forbidden_claims",
    "gap_disclosure",
    "citation_validation",
)

# Quality, read only after the safety columns above.
QUALITY = ("outcome_accuracy", "answerable_answered", "fact_coverage")


def slice_metrics(cases: list[dict], splits: dict[str, str], split: str) -> dict:
    rows = [c for c in cases if split == "all" or splits.get(c["question_id"]) == split]
    if not rows:
        return {}
    # "Passed" is not stored per case; the outcome comparison is what the
    # deterministic report scores on, so it is recomputed here rather than
    # inferred from a field that does not exist.
    correct = sum(1 for r in rows if r["actual_outcome"] == r["expected_outcome"])
    return {
        "cases": len(rows),
        "outcome_acc_split": round(100.0 * correct / len(rows), 1),
        "answered": sum(1 for r in rows if r.get("answered")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="all", choices=("all", "tune", "holdout"))
    args = parser.parse_args(argv)

    splits = {c.question_id: c.split for c in load_cases()}
    reports = sorted(REPORT_DIR.glob("rag_partC_[AB]_*.json"))
    if not reports:
        print("No arm reports found. Run the matrix first.", file=sys.stderr)
        return 1

    rows = []
    for path in reports:
        data = json.loads(path.read_text())
        metrics = data.get("metrics", {})
        sliced = slice_metrics(data.get("cases", []), splits, args.split)
        rows.append(
            {
                "arm": data.get("arm", path.stem),
                "strategy": data.get("strategy"),
                "window": data.get("window_size"),
                "top_k": data.get("settings", {}).get("top_k"),
                "promote": data.get("promotion", {}).get("promote"),
                **{key: metrics.get(key) for key in SAFETY},
                **{key: metrics.get(key) for key in QUALITY},
                **sliced,
            }
        )

    order = ["arm", "top_k", "cases", *SAFETY, *QUALITY, "outcome_acc_split", "promote"]
    widths = {key: max(len(key), *(len(str(r.get(key))) for r in rows)) for key in order}
    print(f"split={args.split}\n")
    print("  ".join(key.ljust(widths[key]) for key in order))
    print("  ".join("-" * widths[key] for key in order))
    for row in rows:
        print("  ".join(str(row.get(key)).ljust(widths[key]) for key in order))

    out = REPORT_DIR / f"rag_partC_comparison_{args.split}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
