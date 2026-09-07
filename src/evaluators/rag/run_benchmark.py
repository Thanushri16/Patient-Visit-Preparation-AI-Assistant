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

from rag.config import SENTENCE_TABLE_NAME, SETTINGS, env_value  # noqa: E402
from rag.embeddings import build_embed_model  # noqa: E402
from rag.pipeline import answer_knowledge_question  # noqa: E402
from rag.retrievers import BasicChunkRetriever, SentenceWindowRetriever  # noqa: E402
from rag.store import KnowledgeStore  # noqa: E402

from evaluators.rag.dataset import load_cases  # noqa: E402
from evaluators.rag.deterministic import Report, score_case  # noqa: E402
from evaluators.rag.judged import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    JudgeRequest,
    judge_all,
    summarise,
)
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
    parser.add_argument(
        "--judge",
        action="store_true",
        help="also run the DeepEval metrics; costs several model calls per case",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=3,
        help="parallel judge calls; too high loses scores to rate limits",
    )
    # Defaults to the configured stage rather than a literal. The old default
    # was "primary", a stage config.py does not implement -- harmless, because
    # pipeline.py branches on "shadow" alone, but it meant every report claimed
    # a mode the application cannot be run in.
    parser.add_argument("--mode", default=SETTINGS.mode)
    parser.add_argument(
        "--strategy",
        choices=("basic", "sentence_window"),
        default="basic",
        help="which retriever to measure; sentence_window reads the sentence table",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="sentences either side of the match; ignored unless --strategy "
             "sentence_window",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="retrieval depth; defaults to SETTINGS.top_k",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        help="context budget; defaults to SETTINGS.max_context_tokens. The "
             "matrix's equal-budget condition sets this explicitly",
    )
    parser.add_argument(
        "--label",
        help="name for this arm in the report filename; defaults to the strategy",
    )
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

    # Retrieval depth and context budget are read from SETTINGS by the pipeline,
    # so an arm that varies them has to say so here rather than pass an argument
    # down: the equal-budget condition changes what every downstream consumer
    # sees, not just the retriever.
    if args.top_k is not None:
        SETTINGS.top_k = args.top_k
    if args.max_context_tokens is not None:
        SETTINGS.max_context_tokens = args.max_context_tokens

    embed_model = build_embed_model()
    if args.strategy == "sentence_window":
        retriever = SentenceWindowRetriever(
            KnowledgeStore(table_name=SENTENCE_TABLE_NAME),
            embed_model,
            window_size=args.window,
        )
    else:
        retriever = BasicChunkRetriever(KnowledgeStore(), embed_model)
    client = OpenAI(api_key=env_value("OPENAI_API_KEY"))

    arm = args.label or (
        args.strategy
        if args.strategy == "basic"
        else f"{args.strategy}_w{args.window}"
    )
    print(
        f"arm={arm} strategy={retriever.strategy} window={retriever.window_size} "
        f"top_k={SETTINGS.top_k} budget={SETTINGS.max_context_tokens} "
        f"min_similarity={SETTINGS.min_similarity} split={args.split} "
        f"cases={len(cases)}\n"
    )

    report = Report()
    shadow = ShadowReport()
    generated = cited_ok = 0
    # Collected during the loop, judged in one concurrent batch afterwards.
    judge_queue: list[JudgeRequest] = []

    for index, case in enumerate(cases, start=1):
        result = answer_knowledge_question(case.question, retriever, client, mode=args.mode)
        outcome = STATUS_TO_OUTCOME.get(result.status, result.status)
        cited = tuple(dict.fromkeys(c.document_id for c in result.citations))
        scored = score_case(case, result.text, outcome, cited, source=result.source)
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
                answer_source=result.source,
            )
        )

        if args.judge and scored.answered:
            # Only answered cases are judged. Faithfulness of "I don't have
            # documentation on that" is not a meaningful question, and scoring
            # refusals would drag a generation metric toward whatever the judge
            # makes of a refusal rather than of an answer.
            context = [
                source.text
                for source in (result.evidence.supporting if result.evidence else ())
            ]
            judge_queue.append(
                JudgeRequest(
                    question_id=case.question_id,
                    group=case.group,
                    question=case.question,
                    answer=result.text,
                    context=tuple(context),
                    expected=case.expected_answer.strip()
                    or " ".join(case.expected_facts or case.expected_covered_facts)
                    or None,
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
    judged = (
        judge_all(
            judge_queue, model=args.judge_model, max_concurrent=args.judge_concurrency
        )
        if judge_queue
        else []
    )
    judged_summary = summarise(judged) if args.judge else None

    print("\n" + json.dumps(metrics, indent=2))
    print("\ndivergence: " + json.dumps({k: v for k, v in divergence.items() if v}))
    print("promotion:  " + json.dumps(promotion["checks"]))
    print(f"promote to preferred: {promotion['promote']}  (split={args.split})")

    if judged_summary:
        # Its own block, never folded into the deterministic figures above: the
        # two measure different things and fail differently, and averaging them
        # would hide which half of the pipeline is at fault.
        print(f"\njudged (DeepEval, model={args.judge_model}):")
        print(json.dumps(judged_summary, indent=2))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"rag_partC_{arm}_{args.split}.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": stamp,
                "mode": args.mode,
                "split": args.split,
                "arm": arm,
                "strategy": retriever.strategy,
                "window_size": retriever.window_size,
                "settings": SETTINGS.model_dump(),
                "metrics": metrics,
                "judged": judged_summary,
                "judge_model": args.judge_model if args.judge else None,
                "judged_cases": [vars(j) for j in judged],
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
