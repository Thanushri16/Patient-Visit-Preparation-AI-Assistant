"""Deterministic evidence sufficiency check.

Whether there is enough evidence to answer is application logic, never the
model's. That is the same rule the rest of this codebase holds to — the
classifier chooses a workflow, the extractor pulls stated fields, but ordering,
completeness and every safety decision are deterministic. Here it matters more
than usual: a model asked "is this enough to answer?" while holding text that
looks relevant will say yes, because relevance is what it can see.

The hard part is not an empty result. It is a plausible-but-wrong one: a
question with no answer in the corpus pulling a closely related passage that
outscores a correct answer to a different question. That is measured here, not
hypothesised, and no similarity threshold separates it — so three further guards
run past the threshold, in increasing order of cost:

1.  **Category consistency.** Retrieval that agrees with none of the categories
    the question could belong to is not evidence about that question.
2.  **Score dispersion.** Correct retrieval usually clusters — several nodes,
    often from one document, one clearly ahead. A single mid-scoring node with
    nothing behind it is the shape of a near miss, so it must clear a higher bar
    alone than it would with support.
3.  **Answerability.** One cheap model call asking whether the context contains
    the answer rather than merely relating to the topic. Off by default; its
    output is advisory to a deterministic decision, so a "no" forces a fallback
    but a "yes" cannot overrule guards 1 and 2.

This module does not split compound questions (A.4.1) and does not know about
never-route topics (A6). Both sit above it.

LlamaIndex's `SimilarityPostprocessor` covers the score floor below and nothing
else. The floor is the weakest of these checks -- the Part A sweep held
near-miss resistance at 10/10 across floors from 0.30 to 0.55, so what separates
a right answer from a wrong one is guard 3, which has no framework equivalent.
See section 6.0 of documentation/rag_architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

try:
    from .config import SETTINGS
    from .query import infer_categories
    from .store import RetrievedChunk
except ImportError:  # pragma: no cover - allows running as a script
    from config import SETTINGS
    from query import infer_categories
    from store import RetrievedChunk


class EvidenceVerdict(StrEnum):
    SUFFICIENT = "sufficient"
    NO_RESULTS = "no_results"
    BELOW_THRESHOLD = "below_threshold"
    TOO_FEW_SUPPORTING = "too_few_supporting"
    WRONG_CATEGORY = "wrong_category"
    ISOLATED_MATCH = "isolated_match"
    NOT_ANSWERABLE = "not_answerable"


@dataclass(frozen=True)
class EvidenceDecision:
    """Whether to answer, and the reason, so a failure is diagnosable."""

    verdict: EvidenceVerdict
    supporting: tuple[RetrievedChunk, ...]
    top_similarity: float
    reason: str
    guard: str | None = None          # which A.4.2 guard fired, if any
    expected_categories: frozenset[str] = frozenset()

    @property
    def sufficient(self) -> bool:
        return self.verdict is EvidenceVerdict.SUFFICIENT


def check_evidence(
    sources: Sequence[RetrievedChunk],
    question: str = "",
    min_similarity: float | None = None,
    min_supporting_nodes: int | None = None,
    enforce_category: bool | None = None,
    isolated_similarity: float | None = None,
) -> EvidenceDecision:
    """Decide whether the retrieved sources are enough to answer from.

    Only sources at or above the similarity floor count as supporting, and the
    answer is generated from those rather than from everything retrieved: a node
    that was not good enough to justify answering is not good enough to be
    quoted in the answer either.
    """

    floor = min_similarity if min_similarity is not None else SETTINGS.min_similarity
    needed = (
        min_supporting_nodes
        if min_supporting_nodes is not None
        else SETTINGS.min_supporting_nodes
    )

    if not sources:
        return EvidenceDecision(
            verdict=EvidenceVerdict.NO_RESULTS,
            supporting=(),
            top_similarity=0.0,
            reason="retrieval returned nothing",
        )

    ranked = sorted(sources, key=lambda source: source.similarity, reverse=True)
    top = ranked[0].similarity
    supporting = tuple(source for source in ranked if source.similarity >= floor)

    if not supporting:
        return EvidenceDecision(
            verdict=EvidenceVerdict.BELOW_THRESHOLD,
            supporting=(),
            top_similarity=top,
            reason=f"best match {top:.3f} is below the {floor:.2f} floor",
        )

    if len(supporting) < needed:
        return EvidenceDecision(
            verdict=EvidenceVerdict.TOO_FEW_SUPPORTING,
            supporting=supporting,
            top_similarity=top,
            reason=(
                f"{len(supporting)} source(s) cleared the floor, {needed} required"
            ),
        )

    # ---- A.4.2 guard 1: category consistency --------------------------------
    #
    # Only a veto. It never sends a query anywhere and never restricts the
    # search; it objects when what came back belongs to none of the categories
    # the question could be about. When the question names no subject the
    # inference is empty and the guard abstains, because "cannot tell" must not
    # read as "mismatch".
    expected = infer_categories(question) if question else frozenset()
    category_on = (
        enforce_category
        if enforce_category is not None
        else SETTINGS.enforce_category_consistency
    )
    if category_on and expected:
        found = {source.category for source in supporting}
        if not (found & expected):
            return EvidenceDecision(
                verdict=EvidenceVerdict.WRONG_CATEGORY,
                supporting=(),
                top_similarity=top,
                reason=(
                    f"question looks like {sorted(expected)} but the evidence is "
                    f"{sorted(found)}"
                ),
                guard="category_consistency",
                expected_categories=expected,
            )

    # ---- A.4.2 guard 2: score dispersion ------------------------------------
    #
    # A cheap stand-in for guard 3, and it only runs when guard 3 is off.
    #
    # Its premise was that correct retrieval clusters, so a lone supporter is the
    # signature of a near miss. Measured, that premise does not hold: it fired on
    # three questions, catching one true near miss ("what does my A1C result
    # mean") and wrongly refusing two real ones, including "what does the ABCDE
    # rule mean for moles" -- a narrow question the corpus answers completely, in
    # exactly one chunk. The scores do not separate them either: the true miss
    # sat at 0.350 and the real answer at 0.382.
    #
    # Guard 3 rejects the A1C question on its own, so with guard 3 running this
    # one catches nothing unique and costs real answers. It stays as the
    # deterministic fallback for a configuration that disables guard 3, where
    # some protection past guard 1 is better than none.
    guard_three_running = (
        SETTINGS.answerability_check if enforce_category is None else False
    )
    isolated_floor = (
        isolated_similarity
        if isolated_similarity is not None
        else SETTINGS.isolated_node_similarity
    )
    if (
        not guard_three_running
        and len(supporting) == 1
        and supporting[0].similarity < isolated_floor
    ):
        return EvidenceDecision(
            verdict=EvidenceVerdict.ISOLATED_MATCH,
            supporting=(),
            top_similarity=top,
            reason=(
                f"one unsupported match at {supporting[0].similarity:.3f}, "
                f"below the {isolated_floor:.2f} bar for a lone source"
            ),
            guard="score_dispersion",
            expected_categories=expected,
        )

    return EvidenceDecision(
        verdict=EvidenceVerdict.SUFFICIENT,
        supporting=supporting,
        top_similarity=top,
        reason=f"{len(supporting)} source(s) at or above {floor:.2f}",
        expected_categories=expected,
    )


# ---------------------------------------------------------------------------
# A.4.2 guard 3: answerability
# ---------------------------------------------------------------------------

# Deliberately framed around SUBJECT, not completeness.
#
# The first version asked whether the text "states the specific fact asked for",
# and rejected 6 of 7 questions the corpus answers well — a correct passage
# worded differently from the question reads as not stating the fact. The
# failure this guard exists to catch is narrower than that: text about a
# DIFFERENT test or procedure than the one asked about. So that is what it is
# asked, and it is told to say YES on a partial answer.
ANSWERABILITY_PROMPT = """You are checking whether retrieved reference text is
about the right subject. You are NOT deciding whether it answers the question.

Does the text below cover the test, procedure or topic that the question is
about?

YES - the text covers that subject, whatever it says about it. This INCLUDES a
      general guide to a class of tests covering a specific test in that class:
      a page about preparing for lab tests covers a blood test or a urine test;
      a page about colorectal screening covers a stool test.
NO  - the text is about a DIFFERENT, SIBLING test than the one asked about. A
      question about a PET scan with text about a CT scan. A question about an
      upper endoscopy with text about a colonoscopy. A question about one blood
      result with text about an unrelated test.

The distinction is whether the text's subject INCLUDES what was asked about
(answer YES) or SUBSTITUTES a different one for it (answer NO).

Judge subject only. Do not consider whether the text is complete, whether it
states a direct yes or no, or whether it uses the same words as the question.

Reply with exactly one word: YES or NO."""


# Guard 3 verdicts, keyed on the question and the node judged.
#
# The guard is an LLM call, so it is not reproducible: the same question at
# temperature 0 was measured flipping between "generated" and
# "insufficient_evidence" across three consecutive runs, and per-chunk early
# exit amplifies it, since one borderline verdict changes the whole outcome.
#
# That is not survivable. A benchmark whose numbers move between identical runs
# cannot detect a regression, and Part B's success criterion is precisely that
# the numbers do not move. The corpus and the question are both fixed, so a
# verdict is stable data — caching it makes repeat runs reproducible and removes
# most of the guard's cost at the same time.
_ANSWERABILITY_CACHE: dict[tuple[str, str], bool] = {}


def clear_answerability_cache() -> None:
    """Drop cached verdicts. Needed after a re-ingest changes node contents."""

    _ANSWERABILITY_CACHE.clear()


def check_answerability(client, question: str, sources: Sequence[RetrievedChunk]) -> bool | None:
    """Ask whether any retrieved chunk covers the question's subject.

    Returns None when the check could not be made, which callers must treat as
    "no information" rather than as a refusal — an outage must not silently turn
    every answer into a fallback.

    Asked **per chunk, not over the concatenated set**, and that is the whole
    trick. Judging six chunks at once dilutes the one that matters: a question
    about whether a hearing test hurts retrieves the risks section alongside
    several about tones and headphones, and asked about the pile the model says
    no. Asked about the risks chunk alone it says yes. Four rewordings of the
    prompt could not fix that, because the problem was the question being put,
    not the words putting it.

    Early exit on the first YES, so the common case — a question the corpus
    answers — costs one call. Only a genuine near miss pays for the whole set,
    which is the right way round.
    """

    verdicts: list[bool] = []
    for source in sources:
        key = (question.strip(), source.node_id)
        cached = _ANSWERABILITY_CACHE.get(key)
        if cached is not None:
            if cached:
                return True
            verdicts.append(False)
            continue

        block = f"{source.title} — {source.section or ''}\n{source.text}"
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ANSWERABILITY_PROMPT},
                    {"role": "user", "content": f"{block}\n\nQuestion: {question}"},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            verdict = (response.choices[0].message.content or "").strip().upper()
        except Exception:  # noqa: BLE001 - an outage must not become a refusal
            continue

        if verdict.startswith("YES"):
            _ANSWERABILITY_CACHE[key] = True
            return True
        if verdict.startswith("NO"):
            _ANSWERABILITY_CACHE[key] = False
            verdicts.append(False)

    if not verdicts:
        # Nothing could be judged at all: no information, not a refusal.
        return None
    return False


def apply_answerability(
    decision: EvidenceDecision, client, question: str
) -> EvidenceDecision:
    """Downgrade a sufficient decision when the context does not answer it.

    Advisory to a deterministic decision, exactly as the plan specifies: a "no"
    forces the fallback, a "yes" changes nothing. The guard can only ever remove
    an answer, never add one, so a confidently wrong model cannot talk the
    pipeline into answering something guards 1 and 2 rejected.
    """

    if not decision.sufficient:
        return decision

    verdict = check_answerability(client, question, decision.supporting)
    if verdict is not False:
        return decision

    return EvidenceDecision(
        verdict=EvidenceVerdict.NOT_ANSWERABLE,
        supporting=(),
        top_similarity=decision.top_similarity,
        reason="the retrieved text relates to the topic but does not answer it",
        guard="answerability",
        expected_categories=decision.expected_categories,
    )
