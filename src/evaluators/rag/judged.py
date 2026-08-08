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


def _test_case(question: str, answer: str, context: Sequence[str], expected: str | None):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
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

    common = dict(model=model, threshold=threshold, async_mode=False, include_reason=True)
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
        "contextual_precision": ContextualPrecisionMetric(**common),
        "contextual_recall": ContextualRecallMetric(**common),
        "contextual_relevancy": ContextualRelevancyMetric(**common),
    }


# Contextual precision and recall compare the retrieved context against what a
# correct answer would contain, so they need an expected output.
#
# The benchmark now carries a written reference answer per answerable case, and
# that is what these two are scored against. Before it existed they were fed the
# expected-fact fragments joined together, which was a poor target twice over:
# statement decomposition over concatenated fragments produces units nobody
# would write, and the facts are a selected subset rather than a whole answer,
# so contextual recall was really re-asking the deterministic fact check in a
# fuzzier, unrepeatable form.
#
# Fragments remain the fallback for a case with no reference answer, and a case
# with neither skips these two metrics rather than being given a made-up target.
NEEDS_EXPECTED = ("contextual_precision", "contextual_recall")


def judge_case(
    metrics: dict,
    *,
    question_id: str,
    group: str,
    question: str,
    answer: str,
    context: Sequence[str],
    expected_facts: Sequence[str],
    expected_answer: str = "",
) -> JudgedCase:
    """Score one answered case. Returns the reason when a case is not judged."""

    result = JudgedCase(question_id=question_id, group=group)

    if not context:
        result.skipped = "no retrieved context"
        return result
    if not answer.strip():
        result.skipped = "no answer"
        return result

    expected = expected_answer.strip() or (
        " ".join(expected_facts) if expected_facts else None
    )
    case = _test_case(question, answer, context, expected)

    for name, metric in metrics.items():
        if name in NEEDS_EXPECTED and not expected:
            continue
        try:
            metric.measure(case)
            if metric.score is not None:
                result.scores[name] = round(float(metric.score), 3)
                if getattr(metric, "reason", None):
                    result.reasons[name] = str(metric.reason)[:240]
        except Exception as error:  # noqa: BLE001 - a judge outage is not a failure
            result.reasons[name] = f"not scored: {type(error).__name__}"
    return result


def summarise(cases: Sequence[JudgedCase], threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Aggregate judged scores.

    Reports the mean and the share at or above the threshold for each metric
    separately. No combined figure: averaging faithfulness with contextual
    recall would hide which half of the pipeline is at fault, which is the one
    thing these metrics are for.
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
        if not values:
            continue
        summary[name] = {
            "mean": round(sum(values) / len(values), 3),
            "at_or_above_threshold": round(
                100.0 * sum(1 for v in values if v >= threshold) / len(values), 1
            ),
            "n": len(values),
        }
    return summary
