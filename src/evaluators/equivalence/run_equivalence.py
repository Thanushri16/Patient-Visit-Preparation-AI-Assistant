"""Part B equivalence run: chain versus graph.

    uv run python -m src.evaluators.equivalence.run_equivalence
    uv run python -m src.evaluators.equivalence.run_equivalence --record

Records model calls on the first pass so the comparison is reproducible; after
that it replays offline, where a cache miss is a loud failure rather than a
quiet live call.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluators.equivalence.conversations import CONVERSATIONS, calibrate  # noqa: E402
from evaluators.equivalence.harness import (  # noqa: E402
    EquivalenceReport, Mode, compare_conversation,
)
from evaluators.equivalence.runner import recording_app  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "partb"


def _runner(test_client, orchestrator: str):
    def run(session_id: str, message: str) -> dict:
        response = test_client.post(
            "/chat",
            json={
                "message": message,
                "session_id": session_id,
                "orchestrator": orchestrator,
            },
        ).json()
        return {"reply": response["reply"], "state": response.get("state", {})}

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare chain and graph.")
    parser.add_argument(
        "--record", action="store_true",
        help="allow live model calls to (re)fill the cache",
    )
    parser.add_argument(
        "--warm-passes", type=int, default=3,
        help="how many times to warm the cache before comparing",
    )
    parser.add_argument(
        "--strict-all", action="store_true",
        help="compare every conversation strictly; valid only with a warm cache",
    )
    args = parser.parse_args(argv)

    if args.record:
        # Warm the cache through the chain alone, so recording cannot be biased
        # by whichever orchestrator happened to ask first.
        #
        # Warmed more than once, because one pass does not cover the prompt
        # space. The chain is non-deterministic upstream: extraction can return
        # "sore throat" on one pass and "a sore throat" on the next, and that
        # text is interpolated into the RAG prompts further down, so a single
        # pass records one variant and the comparison run asks for another.
        # Each additional pass records the variants the previous one missed;
        # the loop stops when a pass adds nothing.
        for attempt in range(1, args.warm_passes + 1):
            with recording_app(offline=False) as (test_client, cache):
                run = _runner(test_client, "chain")
                for name, spec in CONVERSATIONS.items():
                    for message in spec.messages:
                        run(f"warm{attempt}-{name}", message)
                print(f"recording pass {attempt}: {cache.stats.describe()}")
                if cache.stats.misses == 0:
                    break

    with recording_app(offline=not args.record) as (test_client, cache):
        chain = _runner(test_client, "chain")
        graph = _runner(test_client, "graph")

        # The declaration is checked before it is relied on: a STRICT
        # conversation that cannot reproduce itself is misdeclared.
        misdeclared = calibrate(chain, repeats=1)

        report = EquivalenceReport()
        for name, spec in CONVERSATIONS.items():
            mode = Mode.STRICT if (args.strict_all or not args.record) else spec.mode
            report.comparisons.append(
                compare_conversation(name, spec.messages, chain, graph, mode)
            )

        summary = report.summary()
        summary["cache"] = {
            "hits": cache.stats.hits,
            "misses": cache.stats.misses,
            "hit_rate": cache.stats.hit_rate,
        }
        summary["misdeclared_strict"] = list(misdeclared)

        # A miss during comparison is not a cache inefficiency, it is a hole in
        # the result. The application catches its own model-call failures --
        # generation retries then falls back, the answerability guard fails open
        # -- so an offline miss becomes a fallback reply rather than an error,
        # and both orchestrators produce the *same* fallback. The comparison
        # then passes by agreeing about nothing. Six such misses were passing
        # silently before this check existed.
        summary["cache_complete"] = cache.stats.misses == 0

    print(json.dumps(summary, indent=2))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # A stable filename, matching reports/rag/. Timestamped output accumulated
    # near-identical files that had to be pruned by hand; the run is recorded
    # inside the file, so the name does not need to carry it too.
    path = REPORT_DIR / "partB_equivalence.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": stamp,
                "summary": summary,
                "conversations": [
                    {
                        "id": c.conversation_id,
                        "mode": c.mode.value,
                        "turns": c.turns,
                        "equivalent": c.equivalent,
                        "divergences": [d.describe() for d in c.divergences],
                    }
                    for c in report.comparisons
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {path}")
    if not summary["cache_complete"]:
        print(
            f"\nINCOMPLETE: {cache.stats.misses} cache misses during comparison. "
            "Those turns compared two identical fallback replies, not two real "
            "answers. Re-record with more --warm-passes before believing the "
            "result.",
        )

    return (
        0
        if summary["equivalent"] and not misdeclared and summary["cache_complete"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
