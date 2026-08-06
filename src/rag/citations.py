"""Citation binding and validation.

The generation prompt asks the model to mark each claim with the number of the
context block it came from. Asking is not the guarantee — this module is. An
answer whose markers do not resolve, or which makes claims with no marker at
all, is rejected and never reaches the patient.

The rule the plan states is "every claim is traceable". Enforcing it exactly
would require knowing what a claim is, which is a judgement call. What is
checkable deterministically is weaker but real: every marker resolves to a
source that was actually retrieved, and an answer that asserts something carries
at least one marker. That catches the failures that matter — an invented `[5]`
when four blocks were supplied, and a fluent uncited paragraph produced from the
model's own knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

try:
    from .store import RetrievedChunk
except ImportError:  # pragma: no cover - allows running as a script
    from store import RetrievedChunk


# "[1]", "[2]" — and "[1][2]" or "[1, 2]" as separate markers.
MARKER = re.compile(r"\[(\d+)\]")

# Wording the assistant uses when it is declining rather than asserting. Such a
# sentence carries no claim about the world, so requiring a citation on it would
# force a citation onto a refusal — the same mistake the output guardrail
# already avoids by ignoring matches the surrounding text negates.
NON_ASSERTING = re.compile(
    r"\b(i don'?t have|i cannot|i can'?t|no documentation|not something i can)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Citation:
    """One resolved source reference, ready to return to the caller."""

    marker: str
    document_id: str
    title: str
    section: str | None
    page_number: int | None
    source_url: str | None
    last_updated: str | None


@dataclass(frozen=True)
class CitationResult:
    """The outcome of validating one generated answer."""

    valid: bool
    citations: tuple[Citation, ...]
    problem: str | None = None


def extract_markers(answer: str) -> list[int]:
    """Return the marker numbers used in an answer, in order of first use."""

    seen: list[int] = []
    for found in MARKER.finditer(answer):
        number = int(found.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def validate_citations(
    answer: str, sources: Sequence[RetrievedChunk]
) -> CitationResult:
    """Bind an answer's markers to the sources it was generated from.

    Sources are numbered from 1 in the order they were given to the model, which
    is the order `build_context_blocks` produced and the order the prompt
    displays. Any other numbering would make a correct marker resolve to the
    wrong document, which is worse than an unresolvable one because it looks
    right.
    """

    markers = extract_markers(answer)

    if not markers:
        if NON_ASSERTING.search(answer):
            # A refusal or a fallback asserts nothing, so it needs no source.
            return CitationResult(valid=True, citations=())
        return CitationResult(
            valid=False,
            citations=(),
            problem="the answer makes claims with no citation",
        )

    citations: list[Citation] = []
    for number in markers:
        if number < 1 or number > len(sources):
            return CitationResult(
                valid=False,
                citations=(),
                problem=(
                    f"marker [{number}] does not match any of the "
                    f"{len(sources)} sources supplied"
                ),
            )
        source = sources[number - 1]
        citations.append(
            Citation(
                marker=f"[{number}]",
                document_id=source.document_id,
                title=source.title,
                section=source.section,
                page_number=source.page_number,
                source_url=source.source_url,
                last_updated=source.last_updated,
            )
        )

    return CitationResult(valid=True, citations=tuple(citations))
