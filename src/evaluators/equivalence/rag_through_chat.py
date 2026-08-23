"""Run the Part A RAG benchmark questions through /chat on both orchestrators.

`src/evaluators/rag/run_benchmark.py` calls `answer_knowledge_question` directly,
so it never touches `app.answer_turn` and cannot tell the two orchestrators
apart -- running it under ORCHESTRATOR=graph reproduces its numbers by
construction rather than by evidence. This asks the question that one cannot:
does the knowledge branch behave the same when the *graph* is the thing routing
to it?

Compared per question: the RAG status and source, the set of cited document ids,
and the reply itself. Model calls are replayed from the equivalence cache, so a
difference here is the orchestrator rather than generation jitter.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluators.rag.dataset import load_cases  # noqa: E402
from evaluators.equivalence.runner import recording_app  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "partb"


def _rag_fields(state: dict) -> dict:
    rag = state.get("rag") or {}
    citations = rag.get("citations") or []
    return {
        "status": rag.get("status"),
        "source": rag.get("source"),
        "documents": sorted(
            c.get("document_id") for c in citations if isinstance(c, dict)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--split", default="all", choices=["all", "tune", "holdout"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    cases = [c for c in load_cases() if args.split == "all" or c.split == args.split]
    if args.limit:
        cases = cases[: args.limit]

    with recording_app(offline=not args.record) as (test_client, cache):

        def ask(orchestrator: str, case, tag: str) -> dict:
            payload = test_client.post(
                "/chat",
                json={
                    "message": case.question,
                    "session_id": f"{tag}-{case.question_id}",
                    "orchestrator": orchestrator,
                },
            ).json()
            return payload

        diverged = []
        fields = Counter()
        for case in cases:
            left = ask("chain", case, "rag-chain")
            right = ask("graph", case, "rag-graph")

            problems = []
            if left["reply"] != right["reply"]:
                problems.append("reply")
            lf, rf = _rag_fields(left.get("state") or {}), _rag_fields(right.get("state") or {})
            for key in ("status", "source", "documents"):
                if lf[key] != rf[key]:
                    problems.append(f"rag.{key}")

            for problem in problems:
                fields[problem] += 1
            if problems:
                diverged.append(
                    {
                        "question_id": case.question_id,
                        "question": case.question,
                        "fields": problems,
                        "chain": lf,
                        "graph": rf,
                    }
                )

        summary = {
            "questions": len(cases),
            "split": args.split,
            "diverged": len(diverged),
            "divergent_fields": dict(fields),
            "cache": {"hits": cache.stats.hits, "misses": cache.stats.misses},
            "cache_complete": cache.stats.misses == 0,
            "equivalent": not diverged,
        }

    print(json.dumps(summary, indent=2))
    for row in diverged[:10]:
        print(f"\n{row['question_id']}: {row['fields']}")
        print(f"  chain: {row['chain']}")
        print(f"  graph: {row['graph']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # Stable filename, as in run_equivalence.py -- but only a complete run may
    # claim it. `--limit 3` overwrote the canonical 68-question result once;
    # a partial run must not be able to masquerade as the full one just because
    # it finished later.
    complete = args.limit is None and args.split == "all"
    name = (
        "partB_rag_through_chat.json"
        if complete
        else f"partB_rag_through_chat_partial_{args.split}"
        f"{'_limit%d' % args.limit if args.limit else ''}.json"
    )
    path = REPORT_DIR / name
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "summary": summary,
                "diverged": diverged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {path}")
    return 0 if summary["equivalent"] and summary["cache_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
