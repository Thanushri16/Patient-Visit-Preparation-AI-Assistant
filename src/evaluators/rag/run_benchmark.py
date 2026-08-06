"""Run the RAG benchmark against the knowledge branch.

    uv run python -m src.evaluators.rag.run_benchmark
    uv run python -m src.evaluators.rag.run_benchmark --group near_miss
    uv run python -m src.evaluators.rag.run_benchmark --limit 10

Needs the knowledge store running and an API key. Writes a JSON report to
reports/rag/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI  # noqa: E402

from rag.config import SETTINGS, env_value  # noqa: E402
from rag.embeddings import build_embed_model  # noqa: E402
from rag.pipeline import answer_knowledge_question  # noqa: E402
from rag.retrievers import BasicChunkRetriever  # noqa: E402
from rag.store import KnowledgeStore  # noqa: E402

from evaluators.rag.dataset import load_cases  # noqa: E402
from evaluators.rag.deterministic import Report, score_case  # noqa: E402
from evaluators.rag.shadow import ShadowReport, classify  # noqa: E402
from rag.policy import evaluate as route_of  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "rag"

# How a branch status maps onto the outcomes the benchmark declares.
STATUS_TO_OUTCOME = {
    "generated": "answered",
    "partially_answered": "partially_answered",
    "insufficient_evidence": "fallback",
    "curated_refusal": "curated_refusal",
    "anaphylaxis_note": "anaphylaxis_note",
    "curated_fallback": "curated_answer",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG benchmark.")
    parser.add_argument("--group", action="append", help="only these groups")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", default="primary")
    parser.add_argument(
        "--split",
        choices=("all", "tune", "holdout"),
        default="all",
        help="tune on one half, report the other; see BenchmarkCase.split",
    )
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.group:
        cases = [c for c in cases if c.group in set(args.group)]
    # The chat layer answers these before the branch is reached, so running them
    # here would measure the wrong thing.
    cases = [c for c in cases if c.group != "non_rag"]
    if args.split != "all":
        cases = [c for c in cases if c.split == args.split]
    if args.limit:
        cases = cases[: args.limit]

    retriever = BasicChunkRetriever(KnowledgeStore(), build_embed_model())
    client = OpenAI(api_key=env_value("OPENAI_API_KEY"))

    report = Report()
    shadow = ShadowReport()
    generated = cited_ok = 0

    for index, case in enumerate(cases, start=1):
        result = answer_knowledge_question(case.question, retriever, client, mode=args.mode)
        outcome = STATUS_TO_OUTCOME.get(result.status, result.status)
        cited = tuple(dict.fromkeys(c.document_id for c in result.citations))
        scored = score_case(case, result.text, outcome, cited)
        scored.reason = "; ".join(result.notes)[:160]
        report.results.append(scored)

        # Citation validation rate: of the answers that were generated, how many
        # carried a citation that resolved. An answer is only traceable if it did.
        if result.status in {"generated", "partially_answered"} and result.source == "rag":
            generated += 1
            cited_ok += bool(result.citations)

        shadow.observations.append(
            classify(
                question_id=case.question_id,
                group=case.group,
                expected_outcome=case.expected_outcome,
                # What today's assistant would have said, independent of RAG.
                curated_text=route_of(case.question).response,
                rag_answered=scored.answered,
                rag_cited=bool(result.citations),
                gap_disclosed=scored.gap_disclosed,
            )
        )

        flag = "ok  " if scored.passed else "FAIL"
        print(f"[{index:>2}/{len(cases)}] {flag} {case.question_id} "
              f"{case.expected_outcome} -> {outcome}")

    metrics = report.metrics()
    metrics["citation_validation"] = (
        round(100.0 * cited_ok / generated, 1) if generated else 0.0
    )
    divergence = shadow.counts()
    promotion = shadow.promotion(
        near_miss_resistance=float(metrics["near_miss_resistance"]),
        gap_disclosure=float(metrics["gap_disclosure"]),
        citation_validation=float(metrics["citation_validation"]),
    )
    print("\n" + json.dumps(metrics, indent=2))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"rag_benchmark_{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": stamp,
                "mode": args.mode,
                "split": args.split,
                "settings": SETTINGS.model_dump(),
                "metrics": metrics,
                "divergence": divergence,
                "promotion": promotion,
                "cases": [vars(r) for r in report.results],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
