"""Compare the similarity distributions of the two retrieval strategies.

`min_similarity` was tuned against chunk embeddings in Part A. A sentence and a
400-token chunk are different objects, so the same cosine floor is not
necessarily the same filter for both -- and if it is not, part of any difference
the experiment matrix measures is the threshold rather than the strategy.

This answers that before the matrix runs, and cheaply: retrieval only, no
generation, no judging.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.config import SENTENCE_TABLE_NAME, SETTINGS  # noqa: E402
from rag.embeddings import build_embed_model  # noqa: E402
from rag.retrievers import BasicChunkRetriever, SentenceWindowRetriever  # noqa: E402
from rag.store import KnowledgeStore  # noqa: E402

from evaluators.rag.dataset import load_cases  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "rag"


def profile(retriever, cases, floor: float) -> dict:
    tops: list[float] = []
    below = 0
    answerable_below = 0

    for case in cases:
        sources = retriever.retrieve(case.question)
        top = sources[0].similarity if sources else 0.0
        tops.append(top)
        if top < floor:
            below += 1
            if case.should_answer:
                answerable_below += 1

    tops_sorted = sorted(tops)
    return {
        "questions": len(tops),
        "top_similarity_mean": round(statistics.fmean(tops), 4),
        "top_similarity_median": round(statistics.median(tops), 4),
        "top_similarity_min": round(min(tops), 4),
        "top_similarity_max": round(max(tops), 4),
        "p10": round(tops_sorted[len(tops_sorted) // 10], 4),
        # The number that matters: questions the floor rejects outright, and how
        # many of those the corpus can actually answer.
        "below_floor": below,
        "answerable_below_floor": answerable_below,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="holdout", choices=("all", "tune", "holdout"))
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--floor", type=float, default=SETTINGS.min_similarity)
    args = parser.parse_args(argv)

    cases = [c for c in load_cases() if c.group != "non_rag"]
    if args.split != "all":
        cases = [c for c in cases if c.split == args.split]

    embed_model = build_embed_model()
    arms = {
        "basic": BasicChunkRetriever(KnowledgeStore(), embed_model),
        f"sentence_window_w{args.window}": SentenceWindowRetriever(
            KnowledgeStore(table_name=SENTENCE_TABLE_NAME), embed_model, args.window
        ),
    }

    result = {
        "split": args.split,
        "floor": args.floor,
        "arms": {name: profile(r, cases, args.floor) for name, r in arms.items()},
    }
    print(json.dumps(result, indent=2))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"rag_partC_similarity_profile_{args.split}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
