"""Model-judged metrics, via DeepEval.

A second reporting section, never a replacement for the deterministic one. The
two measure different things and fail differently, and the plan's rule holds:
they are reported side by side and never averaged into a single score.

What lives here is what cannot be checked by matching strings — whether a claim
is entailed by the retrieved context, whether the answer addresses the question,
whether the retrieved context was any good independently of the answer written
from it.

What deliberately does NOT live here:

*   **The safety gates.** Near-miss resistance, never-route compliance and
    forbidden claims stay deterministic. "Was the retriever called" is a fact,
    not a judgement, and a gate that moves with a judge's model version can
    loosen silently when a vendor ships an update.
*   **Fact coverage.** The deterministic matcher requires every number in an
    expected fact to appear exactly, because on this corpus the numbers are the
    substance — fasting windows, screening intervals, ages. A judge scoring
    semantic similarity will accept a paraphrase that changes a quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# The judge is pinned and recorded with every run. A judged score is only
# comparable against another produced by the same judge, so an unrecorded model
# makes two runs incomparable without anything looking wrong.
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

# DeepEval's own pass/fail line per metric. Reported, but not gated on.
DEFAULT_THRESHOLD = 0.7


@dataclass
class JudgedCase:
    """One case's judged scores, or the reason it was not judged."""

    question_id: str
    group: str
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    skipped: str | None = None


def _test_case(question_id: str, question: str, answer: str, context, expected):
    from deepeval.test_case import LLMTestCase

    # `name` is how a result is matched back to its benchmark case. Matching on
    # the question text would break the moment two cases shared wording.
    return LLMTestCase(
        name=question_id,
        input=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=list(context),
    )


def build_metrics(model: str = DEFAULT_JUDGE_MODEL, threshold: float = DEFAULT_THRESHOLD):
    """Return the five metrics the plan names, in report order.

    Generation metrics first, then retrieval metrics: an answer can be faithful
    to context that should never have been retrieved, and reading them in that
    order makes the distinction hard to miss.
    """

    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    common = dict(model=model, threshold=threshold, include_reason=True)
    return [
        FaithfulnessMetric(**common),
        AnswerRelevancyMetric(**common),
        ContextualPrecisionMetric(**common),
        ContextualRecallMetric(**common),
        ContextualRelevancyMetric(**common),
    ]


# Contextual precision and recall compare the retrieved context against what a
# correct answer would contain, so they need an expected output.
#
# The benchmark carries a written reference answer per answerable case, and that
# is what these two are scored against. Before it existed they were fed the
# expected-fact fragments joined together, which was a poor target twice over:
# statement decomposition over concatenated fragments produces units nobody
# would write, and the facts are a selected subset rather than a whole answer,
# so contextual recall was really re-asking the deterministic fact check in a
# fuzzier, unrepeatable form.
#
# Fragments remain the fallback where no reference answer exists. A case with
# neither is skipped by DeepEval's own missing-parameter handling rather than
# being handed a fabricated target.


@dataclass
class JudgeRequest:
    """One answered case queued for judging."""

    question_id: str
    group: str
    question: str
    answer: str
    context: tuple[str, ...]
    expected: str | None = None


def judge_all(
    requests: Sequence[JudgeRequest],
    model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    max_concurrent: int = 3,
    throttle: float = 0.5,
) -> list[JudgedCase]:
    """Judge every request in one concurrent batch.

    Batched rather than case-by-case because the sequential version was
    unusable: five metrics measured one at a time cost roughly four minutes per
    case, so a 35-case run took over two hours and nobody would put that in a
    feedback loop. The judge calls are independent, so they parallelise cleanly.

    Concurrency does not cost reproducibility here, because there was none to
    lose -- judged scores vary run to run whatever the ordering. That is why
    nothing is gated on them, and why the deterministic metrics are the ones the
    promotion gates read.

    `max_concurrent` is 3, well below DeepEval's default of 20, because these
    calls share a rate limit with the benchmark's own generation calls. At 8 a
    holdout run lost seven metric scores to RateLimitError -- which is worse
    than slow, because each metric's mean is then computed over a different
    subset of cases and the metrics stop being comparable with each other. A
    dropped score is silent by design here (`ignore_errors=True` keeps one bad
    metric from abandoning the run), so the only protection is not provoking it.
    """

    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

    judged = {
        r.question_id: JudgedCase(question_id=r.question_id, group=r.group)
        for r in requests
    }

    runnable = []
    for request in requests:
        if not request.context:
            judged[request.question_id].skipped = "no retrieved context"
        elif not request.answer.strip():
            judged[request.question_id].skipped = "no answer"
        else:
            runnable.append(request)
    if not runnable:
        return list(judged.values())

    result = evaluate(
        test_cases=[
            _test_case(r.question_id, r.question, r.answer, r.context, r.expected)
            for r in runnable
        ],
        metrics=build_metrics(model, threshold),
        async_config=AsyncConfig(
            run_async=True, max_concurrent=max_concurrent, throttle_value=throttle
        ),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        # A judge failure on one metric records that metric as unscored rather
        # than abandoning the run; a case missing an expected output skips the
        # two metrics that need one.
        error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
    )

    for test_result in result.test_results:
        case = judged.get(test_result.name)
        if case is None:
            continue
        for metric in test_result.metrics_data or []:
            key = metric.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            if metric.error:
                case.reasons[key] = f"not scored: {metric.error}"[:240]
            elif metric.score is not None:
                case.scores[key] = round(float(metric.score), 3)
                if metric.reason:
                    case.reasons[key] = str(metric.reason)[:240]

    return list(judged.values())


def summarise(cases: Sequence[JudgedCase], threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Aggregate judged scores.

    Mean and share at or above threshold, per metric, never combined. Averaging
    faithfulness with contextual recall would hide which half of the pipeline is
    at fault, which is the one thing these metrics are for.
    """

    judged = [c for c in cases if c.scores]
    names = sorted({name for c in judged for name in c.scores})

    summary: dict[str, object] = {
        "judged_cases": len(judged),
        "skipped_cases": len([c for c in cases if c.skipped]),
        "threshold": threshold,
    }
    for name in names:
        values = [c.scores[name] for c in judged if name in c.scores]
        summary[name] = {
            "mean": round(sum(values) / len(values), 3),
            "at_or_above_threshold": round(
                100.0 * sum(1 for v in values if v >= threshold) / len(values), 1
            ),
            "n": len(values),
        }
    return summary
