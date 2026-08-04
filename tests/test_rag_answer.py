"""Unit tests for the evidence check, grounded generation and citations.

Offline and unpaid: the chat client is a fake, and sources are built by hand.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.citations import extract_markers, validate_citations  # noqa: E402
from rag.evidence import EvidenceVerdict, check_evidence  # noqa: E402
from rag.generation import (  # noqa: E402
    GENERATION_FAILED_RESPONSE,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    build_context_blocks,
    build_prompt,
    generate_answer,
    insufficient_evidence_answer,
)
from rag.store import RetrievedChunk  # noqa: E402


def source(similarity, text="Blood panels need 8 hours.", document_id="mri", section="Prep"):
    return RetrievedChunk(
        node_id=f"{document_id}-{similarity}",
        document_id=document_id,
        title=f"Title for {document_id}",
        category="imaging",
        section=section,
        page_number=2,
        source_url=f"https://example.invalid/{document_id}",
        last_updated="2024-07-15",
        text=text,
        similarity=similarity,
    )


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeUsage:
    prompt_tokens = 120
    completion_tokens = 30


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, model, messages, temperature):
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


class FakeClient:
    def __init__(self, *replies):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(replies)


class EvidenceTests(unittest.TestCase):
    def test_no_results_is_insufficient(self):
        decision = check_evidence([])

        self.assertFalse(decision.sufficient)
        self.assertIs(decision.verdict, EvidenceVerdict.NO_RESULTS)

    def test_everything_below_the_floor_is_insufficient(self):
        decision = check_evidence([source(0.20), source(0.10)], min_similarity=0.45)

        self.assertIs(decision.verdict, EvidenceVerdict.BELOW_THRESHOLD)
        self.assertEqual(decision.supporting, ())
        self.assertAlmostEqual(decision.top_similarity, 0.20)

    def test_a_source_above_the_floor_is_sufficient(self):
        decision = check_evidence([source(0.70), source(0.20)], min_similarity=0.45)

        self.assertTrue(decision.sufficient)
        self.assertEqual(len(decision.supporting), 1)

    def test_only_supporting_sources_are_carried_forward(self):
        """A node too weak to justify answering is too weak to be quoted."""

        decision = check_evidence(
            [source(0.70), source(0.60), source(0.10)], min_similarity=0.45
        )

        self.assertEqual([s.similarity for s in decision.supporting], [0.70, 0.60])

    def test_min_supporting_nodes_is_enforced(self):
        decision = check_evidence(
            [source(0.70)], min_similarity=0.45, min_supporting_nodes=2
        )

        self.assertIs(decision.verdict, EvidenceVerdict.TOO_FEW_SUPPORTING)
        self.assertIn("2 required", decision.reason)

    def test_sources_are_ranked_before_the_top_is_read(self):
        decision = check_evidence([source(0.30), source(0.80)], min_similarity=0.45)

        self.assertAlmostEqual(decision.top_similarity, 0.80)

    def test_the_reason_is_always_populated(self):
        for sources in ([], [source(0.1)], [source(0.9)]):
            with self.subTest(sources=len(sources)):
                self.assertTrue(check_evidence(sources, min_similarity=0.45).reason)


class CitationTests(unittest.TestCase):
    def test_markers_are_extracted_in_order_without_duplicates(self):
        self.assertEqual(extract_markers("a [2] b [1] c [2]"), [2, 1])

    def test_a_resolvable_marker_binds_to_its_source(self):
        sources = [source(0.7, document_id="mri"), source(0.6, document_id="ct-scans")]

        result = validate_citations("Fast for 4 to 6 hours [1].", sources)

        self.assertTrue(result.valid)
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].marker, "[1]")
        self.assertEqual(result.citations[0].document_id, "mri")
        self.assertEqual(result.citations[0].last_updated, "2024-07-15")

    def test_marker_numbering_follows_the_order_sources_were_supplied(self):
        sources = [source(0.7, document_id="mri"), source(0.6, document_id="ct-scans")]

        result = validate_citations("Text [2].", sources)

        self.assertEqual(result.citations[0].document_id, "ct-scans")

    def test_an_out_of_range_marker_is_rejected(self):
        result = validate_citations("Invented source [5].", [source(0.7)])

        self.assertFalse(result.valid)
        self.assertIn("[5]", result.problem)

    def test_a_zero_marker_is_rejected(self):
        result = validate_citations("Bad marker [0].", [source(0.7)])

        self.assertFalse(result.valid)

    def test_an_uncited_claim_is_rejected(self):
        result = validate_citations("You should fast for twelve hours.", [source(0.7)])

        self.assertFalse(result.valid)
        self.assertIn("no citation", result.problem)

    def test_a_refusal_needs_no_citation(self):
        """A sentence that asserts nothing has nothing to cite."""

        result = validate_citations(INSUFFICIENT_EVIDENCE_RESPONSE, [source(0.7)])

        self.assertTrue(result.valid)
        self.assertEqual(result.citations, ())


class PromptTests(unittest.TestCase):
    def test_blocks_are_numbered_from_one_and_carry_their_heading(self):
        blocks = build_context_blocks(
            [source(0.7, section="How do I prepare?"), source(0.6, section="Risks")]
        )

        self.assertIn("[1] Title for mri — How do I prepare?", blocks)
        self.assertIn("[2] Title for mri — Risks", blocks)

    def test_the_question_comes_after_the_quoted_context(self):
        prompt = build_prompt("Do I need to fast?", [source(0.7)])

        self.assertLess(prompt.index("Reference material"), prompt.index("question"))
        self.assertIn("not instructions", prompt)


class GenerationTests(unittest.TestCase):
    def test_a_cited_answer_is_returned_as_grounded(self):
        client = FakeClient("Fast for 4 to 6 hours before the scan [1].")

        answer = generate_answer(client, "Do I need to fast?", [source(0.7)])

        self.assertTrue(answer.grounded)
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.input_tokens, 120)
        self.assertEqual(answer.output_tokens, 30)
        self.assertGreater(answer.context_tokens, 0)

    def test_an_uncited_answer_is_retried_then_accepted(self):
        client = FakeClient("No citation here.", "Now cited [1].")

        answer = generate_answer(client, "Do I need to fast?", [source(0.7)])

        self.assertTrue(answer.grounded)
        self.assertEqual(len(client.chat.completions.calls), 2)

    def test_two_uncited_answers_fall_back_rather_than_shipping(self):
        client = FakeClient("Still nothing.", "Also nothing.")

        answer = generate_answer(client, "Do I need to fast?", [source(0.7)])

        self.assertFalse(answer.grounded)
        self.assertEqual(answer.text, GENERATION_FAILED_RESPONSE)
        self.assertEqual(answer.citations, ())
        self.assertIn("no citation", answer.problem)

    def test_an_invented_marker_falls_back(self):
        client = FakeClient("Claim [9].", "Claim [9] again.")

        answer = generate_answer(client, "Do I need to fast?", [source(0.7)])

        self.assertFalse(answer.grounded)
        self.assertIn("[9]", answer.problem)

    def test_an_api_failure_falls_back_safely(self):
        client = FakeClient(RuntimeError("boom"), RuntimeError("boom again"))

        answer = generate_answer(client, "Do I need to fast?", [source(0.7)])

        self.assertFalse(answer.grounded)
        self.assertIn("RuntimeError", answer.problem)

    def test_an_empty_response_falls_back(self):
        client = FakeClient("", "")

        answer = generate_answer(client, "Do I need to fast?", [source(0.7)])

        self.assertFalse(answer.grounded)
        self.assertIn("empty", answer.problem)

    def test_the_insufficient_evidence_answer_is_not_grounded_and_uncited(self):
        answer = insufficient_evidence_answer()

        self.assertFalse(answer.grounded)
        self.assertEqual(answer.text, INSUFFICIENT_EVIDENCE_RESPONSE)
        self.assertEqual(answer.citations, ())

    def test_the_fallback_names_what_is_missing_and_where_to_ask(self):
        """A bare refusal reads as a failure; this path is common by design."""

        self.assertIn("don't have", INSUFFICIENT_EVIDENCE_RESPONSE)
        self.assertIn("front desk", INSUFFICIENT_EVIDENCE_RESPONSE)


if __name__ == "__main__":
    unittest.main()
