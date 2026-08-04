"""The knowledge branch, end to end.

One entry point: a question in, an answer out. It composes the pieces the
earlier steps built — the route policy, the retriever, the evidence check,
grounded generation — and applies the rollout mode.

Everything here is deterministic control flow. The only model calls are the
query embedding inside the retriever and the answer inside generation, and
neither decides whether to answer.

Two things this does NOT do, and both belong to the caller in the chat layer:

*   Output moderation. The generated answer must pass `moderate_text(...,
    stage="output")` like every other reply, so a retrieved passage cannot
    become a diagnosis or a dosage instruction. It is applied where every other
    reply is moderated rather than here, so there is one place that rule lives.
*   Emergency escalation and state recall, which are steps 1 and 2 of the
    section 3.6 ladder. Both concern the whole turn, not the knowledge branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence

try:
    from .citations import Citation
    from .config import SETTINGS
    from .evidence import EvidenceDecision, apply_answerability, check_evidence
    from .generation import (
        GroundedAnswer,
        generate_answer,
        insufficient_evidence_answer,
    )
    from .policy import RouteDecision, RouteOutcome, evaluate
    from .query import split_question
    from .retrievers import Retriever, assemble_context
except ImportError:  # pragma: no cover - allows running as a script
    from citations import Citation
    from config import SETTINGS
    from evidence import EvidenceDecision, apply_answerability, check_evidence
    from generation import (
        GroundedAnswer,
        generate_answer,
        insufficient_evidence_answer,
    )
    from policy import RouteDecision, RouteOutcome, evaluate
    from query import split_question
    from retrievers import Retriever, assemble_context


@dataclass
class KnowledgeAnswer:
    """Everything one knowledge turn produced, for the reply and for telemetry."""

    text: str = ""
    citations: tuple[Citation, ...] = ()
    status: str = "not_requested"
    source: str = ""                      # what produced the text
    route: RouteDecision | None = None
    evidence: EvidenceDecision | None = None
    retrieved: int = 0
    context_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    subquestions: tuple[str, ...] = ()
    uncovered: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)


def answer_knowledge_question(
    question: str,
    retriever: Retriever,
    chat_client,
    mode: str | None = None,
) -> KnowledgeAnswer:
    """Answer a knowledge question, or decline to.

    The rollout mode decides what the patient sees when both a curated answer and
    a retrieved one are available:

        shadow    - the curated answer, always. Retrieval still runs so its
                    output can be compared, but it is never shown.
        preferred - the retrieved answer when the evidence supports it, the
                    curated answer otherwise.
        primary   - the retrieved answer, with the explicit fallback when there
                    is no evidence. Curated answers remain only for topics the
                    corpus does not cover.
    """

    started = perf_counter()
    stage = (mode or SETTINGS.mode).strip()

    # A.4.1: split BEFORE the route policy, not after.
    #
    # Running the policy on the whole message first meant one refusing half
    # refused the whole turn: "how long do I fast, and should I take my tablet?"
    # was declined entirely, withholding a fasting answer the corpus states
    # plainly. The refusal belongs to the part that earned it, so each part gets
    # its own pass down the ladder.
    if SETTINGS.split_compound_questions:
        parts = split_question(question, max_parts=SETTINGS.max_subquestions)
        if parts.is_compound:
            return _answer_compound(
                question, parts.parts, retriever, chat_client, stage, started
            )

    route = evaluate(question)

    # A refusal or a safety notice ends the turn. The retriever is not called:
    # a decision made after retrieval is a decision made with the wrong answer
    # already in hand.
    if not route.retrieval_allowed:
        return KnowledgeAnswer(
            text=route.response or "",
            status=route.outcome.value,
            source="policy",
            route=route,
            total_latency_ms=(perf_counter() - started) * 1000.0,
            notes=[route.reason],
        )

    sources = retriever.retrieve(question)
    decision = check_evidence(sources, question=question)
    if decision.sufficient and SETTINGS.answerability_check:
        decision = apply_answerability(decision, chat_client, question)
    answer = KnowledgeAnswer(
        route=route,
        evidence=decision,
        retrieved=len(sources),
        retrieval_latency_ms=getattr(sources, "latency_ms", 0.0),
    )

    if not decision.sufficient:
        _fill_from_fallback(answer, route, stage, decision.reason)
        answer.total_latency_ms = (perf_counter() - started) * 1000.0
        return answer

    context = assemble_context(decision.supporting)
    if not context.sources:
        # Every supporting source was larger than the whole budget. Refusing is
        # the right outcome: a fragment cannot be cited.
        _fill_from_fallback(answer, route, stage, "no source fits the context budget")
        answer.total_latency_ms = (perf_counter() - started) * 1000.0
        return answer

    generated = generate_answer(chat_client, question, context.sources)
    answer.context_tokens = generated.context_tokens
    answer.input_tokens = generated.input_tokens
    answer.output_tokens = generated.output_tokens

    if not generated.grounded:
        _fill_from_fallback(answer, route, stage, generated.problem or "not grounded")
        answer.total_latency_ms = (perf_counter() - started) * 1000.0
        return answer

    if stage == "shadow" and route.response:
        # Shadow mode: the patient sees today's answer. The grounded one is kept
        # on the record so the two can be compared without changing behaviour.
        answer.text = route.response
        answer.status = "generated"
        answer.source = "curated"
        answer.notes.append(f"shadow: withheld grounded answer for {route.topic}")
        answer.citations = ()
    else:
        answer.text = generated.text
        answer.citations = generated.citations
        answer.status = "generated"
        answer.source = "rag"

    answer.total_latency_ms = (perf_counter() - started) * 1000.0
    return answer


def _fill_from_fallback(
    answer: KnowledgeAnswer, route: RouteDecision, stage: str, reason: str
) -> None:
    """Use the curated answer when there is one, otherwise the safe fallback.

    This is why nothing regresses. Every question the assistant answers today has
    a curated answer sitting behind the retriever, so a retrieval failure returns
    today's behaviour rather than an apology.
    """

    answer.notes.append(reason)
    if route.response:
        answer.text = route.response
        answer.status = "curated_fallback"
        answer.source = "curated"
        return

    fallback = insufficient_evidence_answer()
    answer.text = fallback.text
    answer.status = "insufficient_evidence"
    answer.source = "fallback"


def _answer_compound(
    question: str,
    parts: tuple[str, ...],
    retriever: Retriever,
    chat_client,
    stage: str,
    started: float,
) -> KnowledgeAnswer:
    """Answer the covered parts of a compound question and name the rest.

    The gap sentence is not optional. An answer that silently covers two of
    three parts reads as complete, which is the worse failure: the patient has
    no way to tell that the question they cared about went unanswered.
    """

    answer = KnowledgeAnswer(subquestions=parts)
    covered: list[str] = []
    uncovered: list[str] = []
    citations: list[Citation] = []
    refusals: list[str] = []
    any_retrieved = False

    for part in parts:
        # Each part goes through the whole ladder. A never-route part inside a
        # compound question does not sink the turn -- it gets its refusal while
        # the rest proceeds.
        piece = answer_knowledge_question(part, retriever, chat_client, mode=stage)
        answer.retrieved += piece.retrieved
        answer.context_tokens += piece.context_tokens
        answer.input_tokens += piece.input_tokens
        answer.output_tokens += piece.output_tokens
        answer.retrieval_latency_ms += piece.retrieval_latency_ms

        if piece.source == "policy":
            refusals.append(piece.text)
            uncovered.append(part)
        elif piece.source in {"rag", "curated"} and piece.status in {
            "generated",
            "curated_fallback",
        }:
            covered.append(piece.text)
            citations.extend(piece.citations)
            any_retrieved = any_retrieved or piece.source == "rag"
        else:
            uncovered.append(part)

    answer.uncovered = tuple(uncovered)
    answer.citations = tuple(citations)
    answer.total_latency_ms = (perf_counter() - started) * 1000.0

    if not covered:
        answer.text = " ".join(refusals) if refusals else insufficient_evidence_answer().text
        answer.status = "curated_refusal" if refusals else "insufficient_evidence"
        answer.source = "policy" if refusals else "fallback"
        return answer

    segments = list(covered) + refusals
    if uncovered and not refusals:
        segments.append(
            "I don't have documentation covering "
            + _join(uncovered)
            + " — the clinic that ordered the test can confirm that, and I'll "
            "note the question for your visit."
        )
    answer.text = " ".join(segment.strip() for segment in segments if segment.strip())
    answer.status = "partially_answered" if uncovered else "generated"
    # "rag" only when retrieval actually produced some of it. A composed answer
    # whose covered half came from the curated table has nothing to cite, and
    # labelling it a RAG answer made citation validation count a legitimate
    # uncited reply as a failure.
    answer.source = "rag" if any_retrieved else "curated"
    return answer


def _join(parts: Sequence[str]) -> str:
    """Render the uncovered parts as the patient asked them."""

    cleaned = [part.rstrip("?.").strip().lower() for part in parts]
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + " or " + cleaned[-1]
