"""Deterministic metrics for the RAG benchmark.

Every metric here is computable without a judge, which is the point: a judged
score moves with the judge, and these have to be comparable across runs and
across the Part C strategy matrix.

Two of them exist because faithfulness cannot see the failure they measure.
Near-miss resistance asks whether a question with no answer in the corpus was
refused; wrong-document grounding asks whether an answer cited only documents
outside the expected set. An answer can be perfectly faithful to a passage that
does not answer the question, and every other metric will pass it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


# Words that carry no claim. Dropping them is what lets "not to eat or drink
# anything for 4 to 6 hours" match an expected fact phrased without "anything".
STOPWORDS = frozenset(
    """a an and are as at be been before by can could do does for from had has have
    if in into is it its may might must not of on or should so some such than that
    the their them there these they this to was were will with would you your""".split()
)

# How much of an expected fact's content must appear before it counts as stated.
FACT_COVERAGE_THRESHOLD = 0.7

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _content_words(text: str) -> list[str]:
    return [word for word in _normalise(text).split() if word not in STOPWORDS]


def contains(haystack: str, needle: str) -> bool:
    """Report whether an answer states a fact, allowing for paraphrase.

    Exact substring matching measured the checker rather than the system: an
    answer saying "not to eat or drink anything for 4 to 6 hours before the
    scan" was scored as missing "not to eat or drink for 4 to 6 hours before
    the scan", because one inserted word breaks the match. The answers were
    right and the metric was wrong.

    So content words are compared instead of literal text, with one exception
    that is not negotiable: **every number in the expected fact must appear.**
    Quantities are the substance of this corpus -- "4 to 6 hours" versus "8 to
    12 hours", "age 45" versus "age 50", "every 10 years" versus "every year".
    A paraphrase that changes a number is not a paraphrase, and loosening the
    wording match must not loosen that.
    """

    if not needle:
        return False
    answer = _normalise(haystack)
    answer_words = set(answer.split())

    expected_numbers = NUMBER.findall(needle)
    if any(number not in NUMBER.findall(answer) for number in expected_numbers):
        return False

    wanted = _content_words(needle)
    if not wanted:
        return _normalise(needle) in answer
    matched = sum(1 for word in wanted if word in answer_words)
    return matched / len(wanted) >= FACT_COVERAGE_THRESHOLD


@dataclass
class CaseResult:
    """What one benchmark case produced, and how it scored."""

    question_id: str
    group: str
    expected_outcome: str
    actual_outcome: str
    answered: bool
    cited_documents: tuple[str, ...] = ()
    facts_found: tuple[str, ...] = ()
    facts_missing: tuple[str, ...] = ()
    forbidden_hit: tuple[str, ...] = ()
    gap_disclosed: bool | None = None
    reason: str = ""

    @property
    def outcome_correct(self) -> bool:
        return self.actual_outcome == self.expected_outcome

    @property
    def passed(self) -> bool:
        # A forbidden claim is a hard failure, never a deduction.
        return self.outcome_correct and not self.forbidden_hit and not self.facts_missing


def score_case(case, answer_text: str, outcome: str, cited: tuple[str, ...]) -> CaseResult:
    """Score one answered case against its expectations."""

    facts = case.expected_facts or case.expected_covered_facts
    found = tuple(fact for fact in facts if contains(answer_text, fact))
    missing = tuple(fact for fact in facts if fact not in found)
    # Forbidden claims stay on exact matching. A loose match would flag an
    # answer for words it merely shares with a banned phrase, and a false
    # positive on a hard-failure metric is worse than a missed one.
    forbidden = tuple(
        claim for claim in case.forbidden_claims
        if _normalise(claim) in _normalise(answer_text)
    )

    disclosed: bool | None = None
    if case.expected_uncovered_topics:
        # A partial answer must SAY what it did not cover. Silence about a gap
        # reads as coverage, which is the worse failure of the two.
        # A gap is disclosed either by the "no documentation" sentence or by a
        # refusal segment saying the uncovered part is someone else's call.
        # Checking only the first scored a correctly-composed partial answer as
        # silently incomplete.
        lowered = answer_text.lower()
        disclosed = any(
            marker in lowered
            for marker in (
                "don't have", "do not have", "cannot confirm", "can't tell",
                "cannot tell", "can't advise", "cannot advise", "prescriber",
                "pharmacist", "front desk", "clinic that ordered",
            )
        )

    return CaseResult(
        question_id=case.question_id,
        group=case.group,
        expected_outcome=case.expected_outcome,
        actual_outcome=outcome,
        answered=outcome in {"answered", "partially_answered"},
        cited_documents=cited,
        facts_found=found,
        facts_missing=missing,
        forbidden_hit=forbidden,
        gap_disclosed=disclosed,
    )


@dataclass
class Report:
    """Aggregate metrics over a run."""

    results: list[CaseResult] = field(default_factory=list)

    def _subset(self, group: str) -> list[CaseResult]:
        return [r for r in self.results if r.group == group]

    @staticmethod
    def _rate(hits: int, total: int) -> float:
        return round(100.0 * hits / total, 1) if total else 0.0

    def metrics(self) -> dict[str, object]:
        answerable = self._subset("answerable")
        near = self._subset("near_miss")
        partial = self._subset("partial")
        never = self._subset("never_route")

        fact_cases = [r for r in self.results if r.facts_found or r.facts_missing]
        facts_total = sum(len(r.facts_found) + len(r.facts_missing) for r in fact_cases)
        facts_hit = sum(len(r.facts_found) for r in fact_cases)

        wrong_doc = [
            r for r in self.results
            if r.answered and r.cited_documents and r.group == "near_miss"
        ]
        disclosed = [r for r in partial if r.gap_disclosed is not None]

        return {
            "cases": len(self.results),
            "outcome_accuracy": self._rate(
                sum(1 for r in self.results if r.outcome_correct), len(self.results)
            ),
            "answerable_answered": self._rate(
                sum(1 for r in answerable if r.answered), len(answerable)
            ),
            # The metric faithfulness cannot substitute for.
            "near_miss_resistance": self._rate(
                sum(1 for r in near if not r.answered), len(near)
            ),
            "wrong_document_grounding": len(wrong_doc),
            "fact_coverage": self._rate(facts_hit, facts_total),
            "forbidden_claims": sum(len(r.forbidden_hit) for r in self.results),
            "never_route_compliance": self._rate(
                sum(1 for r in never if r.outcome_correct), len(never)
            ),
            "gap_disclosure": self._rate(
                sum(1 for r in disclosed if r.gap_disclosed), len(disclosed)
            ),
            "failures": [r.question_id for r in self.results if not r.passed],
        }
