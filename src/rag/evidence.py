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
    # One node alone, with nothing corroborating it, is the signature of a near
    # miss: a single passage that happens to sit close to the question while the
    # rest of the corpus stays away. Correct retrieval on this corpus clusters,
    # because a document's sections share vocabulary. So a lone supporter has to
    # clear a higher bar than it would as part of a group.
    isolated_floor = (
        isolated_similarity
        if isolated_similarity is not None
        else SETTINGS.isolated_node_similarity
    )
    if len(supporting) == 1 and supporting[0].similarity < isolated_floor:
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
ANSWERABILITY_PROMPT = """You are checking retrieved reference text, not answering.

Is the text about the SAME test, procedure or topic the question asks about?

Answer YES if it is, even when it answers only part of the question, or uses
different words than the question does.

Answer NO only when the text is about a DIFFERENT test or procedure — for
example, a question about a PET scan and text about a CT scan, or a question
about an upper endoscopy and text about a colonoscopy.

Reply with exactly one word: YES or NO."""


def check_answerability(client, question: str, sources: Sequence[RetrievedChunk]) -> bool | None:
    """Ask whether the context contains the answer, not merely the topic.

    Returns None when the check could not be made, which callers must treat as
    "no information" rather than as a refusal — an outage must not silently turn
    every answer into a fallback.

    This is the only guard that costs a model call, and it is the only one that
    can catch a near miss inside the right category: "how do I prepare for a PET
    scan" retrieves the CT preparation section, which is genuinely imaging,
    genuinely about preparation, and genuinely not about PET. Guards 1 and 2
    cannot see that. The model can, when asked this narrow question instead of
    being asked to answer.
    """

    context = "\n\n".join(
        f"[{n}] {s.title} — {s.section or ''}\n{s.text}"
        for n, s in enumerate(sources, start=1)
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANSWERABILITY_PROMPT},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
            ],
            temperature=0.0,
            max_tokens=3,
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
    except Exception:  # noqa: BLE001 - an outage must not become a refusal
        return None

    if verdict.startswith("YES"):
        return True
    if verdict.startswith("NO"):
        return False
    return None


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
