"""Unit tests for the route policy and the knowledge branch.

The central assertion is negative: a question that must not be answered from
documents must not reach a retriever. That is proved with a retriever that fails
the test if it is called at all, rather than by inspecting the reply — a reply
can look like a refusal while the search still happened.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guidance import ANAPHYLAXIS_NOTE  # noqa: E402
from rag.evidence import (  # noqa: E402
    EvidenceVerdict,
    apply_answerability,
    check_evidence,
)
from rag.generation import INSUFFICIENT_EVIDENCE_RESPONSE  # noqa: E402
from rag.query import infer_categories, split_question  # noqa: E402
from rag.pipeline import answer_knowledge_question  # noqa: E402
from rag.policy import (  # noqa: E402
    DIAGNOSIS_REFUSAL,
    MEDICATION_CHANGE_REFUSAL,
    NEVER_ROUTE_TOPICS,
    RouteOutcome,
    asks_for_a_diagnosis,
    asks_to_change_medication,
    evaluate,
)
from rag.retrievers import RetrievedSources  # noqa: E402
from rag.store import RetrievedChunk  # noqa: E402


def source(similarity=0.80, document_id="colonoscopy", text="Bowel prep detail.",
           category="endoscopy"):
    return RetrievedChunk(
        node_id=f"{document_id}-{similarity}",
        document_id=document_id,
        title=f"Title for {document_id}",
        category=category,
        section="How do you prepare?",
        page_number=2,
        source_url=f"https://example.invalid/{document_id}",
        last_updated="2024-02-29",
        text=text,
        similarity=similarity,
    )


class ForbiddenRetriever:
    """Fails the test if retrieval happens at all."""

    strategy = "forbidden"
    window_size = None

    def __init__(self, test):
        self._test = test

    def retrieve(self, query, top_k=None, filters=None):
        self._test.fail(f"the retriever was called for: {query!r}")


class StubRetriever:
    strategy = "basic"
    window_size = None

    def __init__(self, sources=()):
        self.sources = list(sources)
        self.calls = []

    def retrieve(self, query, top_k=None, filters=None):
        self.calls.append(query)
        return RetrievedSources(self.sources, latency_ms=12.0)


class StubChatClient:
    def __init__(self, reply="Bowel prep is a clear liquid diet [1]."):
        self.calls = []
        completions = type("C", (), {"create": self._create})()
        self.chat = type("Chat", (), {"completions": completions})()

    def _create(self, model, messages, temperature):
        self.calls.append(messages)
        message = type("M", (), {"content": "Prep is a clear liquid diet [1]."})()
        usage = type("U", (), {"prompt_tokens": 90, "completion_tokens": 20})()
        return type(
            "R", (), {"choices": [type("Ch", (), {"message": message})()], "usage": usage}
        )()


class MedicationChangeTests(unittest.TestCase):
    def test_asking_whether_to_stop_a_medicine_is_detected(self):
        for question in (
            "Should I stop my blood thinner before my colonoscopy?",
            "Can I take my antihistamine the morning of the test?",
            "Should I skip my ibuprofen before the stool test?",
            "Do I still take my metformin before a fasting blood test?",
            "Should I hold my insulin?",
        ):
            with self.subTest(question):
                self.assertTrue(asks_to_change_medication(question))

    def test_a_question_about_food_is_not_a_medication_question(self):
        """Fasting is squarely in scope; refusing it would be a regression."""

        for question in (
            "Should I stop eating before the test?",
            "Can I drink water before my blood test?",
            "Do I need to fast before an MRI?",
            "Should I avoid red drinks before a colonoscopy?",
        ):
            with self.subTest(question):
                self.assertFalse(asks_to_change_medication(question))

    def test_naming_a_medicine_alone_does_not_trigger_a_refusal(self):
        """Recording a medication is intake, not a request to change one."""

        self.assertFalse(asks_to_change_medication("I take warfarin daily."))
        self.assertFalse(
            asks_to_change_medication("What is an allergy blood test for?")
        )


class DiagnosisRequestTests(unittest.TestCase):
    def test_asking_what_a_finding_means_is_detected(self):
        for question in (
            "This mole is asymmetric with a ragged border. Do I have melanoma?",
            "Is this lump serious?",
            "Could this rash be an infection?",
            "Does this mean I have diabetes?",
            "Should I be worried about this mole?",
        ):
            with self.subTest(question):
                self.assertTrue(asks_for_a_diagnosis(question))

    def test_questions_about_the_test_itself_are_not_diagnosis_requests(self):
        """Explaining a rule stays in scope; applying it to the patient does not."""

        for question in (
            "What does the ABCDE rule mean for moles?",
            "Do I have to fast before a skin cancer screening?",
            "What happens during a skin cancer screening?",
            "Is this test painful?",
        ):
            with self.subTest(question):
                self.assertFalse(asks_for_a_diagnosis(question))

    def test_the_refusal_urges_review_rather_than_reassuring(self):
        """Neither 'probably fine' nor 'sounds serious' is safe to say."""

        self.assertIn("examine you", DIAGNOSIS_REFUSAL)
        self.assertIn("have it looked at", DIAGNOSIS_REFUSAL)


class PolicyTests(unittest.TestCase):
    def test_a_diagnosis_request_is_refused_and_never_retrieved(self):
        decision = evaluate("This mole has a ragged border. Do I have melanoma?")

        self.assertIs(decision.outcome, RouteOutcome.CURATED_REFUSAL)
        self.assertFalse(decision.retrieval_allowed)
        self.assertEqual(decision.topic, "diagnosis_request")

    def test_a_medication_change_question_is_refused(self):
        decision = evaluate("Should I stop my blood thinner before my colonoscopy?")

        self.assertIs(decision.outcome, RouteOutcome.CURATED_REFUSAL)
        self.assertFalse(decision.retrieval_allowed)
        self.assertEqual(decision.response, MEDICATION_CHANGE_REFUSAL)

    def test_the_refusal_defers_rather_than_advising(self):
        self.assertIn("prescriber", MEDICATION_CHANGE_REFUSAL)
        self.assertIn("can't advise", MEDICATION_CHANGE_REFUSAL)

    def test_an_interaction_question_is_refused(self):
        decision = evaluate("Does grapefruit interact with my medication?")

        self.assertIs(decision.outcome, RouteOutcome.CURATED_REFUSAL)
        self.assertEqual(decision.topic, "interaction")

    def test_classifying_a_reaction_is_refused(self):
        decision = evaluate("I got a rash after amoxicillin. Is that an allergy?")

        self.assertIs(decision.outcome, RouteOutcome.CURATED_REFUSAL)
        self.assertEqual(decision.topic, "allergy_vs_side_effect")

    def test_whether_a_specialist_is_needed_is_refused(self):
        decision = evaluate("Do I need to see a dermatologist about this?")

        self.assertIs(decision.outcome, RouteOutcome.CURATED_REFUSAL)
        self.assertEqual(decision.topic, "specialist_referral")

    def test_a_described_anaphylactic_reaction_returns_the_safety_note(self):
        decision = evaluate("My throat swelled up and I needed an EpiPen.")

        self.assertIs(decision.outcome, RouteOutcome.ANAPHYLAXIS_NOTE)
        self.assertEqual(decision.response, ANAPHYLAXIS_NOTE)
        self.assertFalse(decision.retrieval_allowed)

    def test_anaphylaxis_outranks_a_medication_phrasing_in_the_same_message(self):
        decision = evaluate(
            "My throat closed up after the pill — should I stop taking it?"
        )

        self.assertIs(decision.outcome, RouteOutcome.ANAPHYLAXIS_NOTE)

    def test_a_covered_topic_allows_retrieval_and_carries_a_fallback(self):
        decision = evaluate("What documents should I bring to my appointment?")

        self.assertTrue(decision.retrieval_allowed)
        self.assertEqual(decision.topic, "documents")
        self.assertTrue(decision.response)

    def test_an_unmatched_question_allows_retrieval_with_no_fallback(self):
        decision = evaluate("What is the bowel prep for a colonoscopy?")

        self.assertTrue(decision.retrieval_allowed)
        self.assertIsNone(decision.response)

    def test_every_never_route_topic_is_a_refusal_in_guidance(self):
        """Guard against a topic being added to the set by mistake."""

        self.assertEqual(
            NEVER_ROUTE_TOPICS,
            frozenset({"interaction", "allergy_vs_side_effect", "specialist_referral"}),
        )


class BranchTests(unittest.TestCase):
    def test_a_refusal_never_reaches_the_retriever(self):
        for question in (
            "Should I stop my blood thinner before my colonoscopy?",
            "Does grapefruit interact with my medication?",
            "Do I need to see a dermatologist about this?",
            "My throat swelled up and I needed an EpiPen.",
            "This mole has a ragged border. Do I have melanoma?",
        ):
            with self.subTest(question):
                answer = answer_knowledge_question(
                    question, ForbiddenRetriever(self), StubChatClient()
                )
                self.assertEqual(answer.source, "policy")
                self.assertEqual(answer.retrieved, 0)

    def test_a_refusal_never_reaches_the_chat_model_either(self):
        client = StubChatClient()

        answer_knowledge_question(
            "Should I stop my blood thinner?", ForbiddenRetriever(self), client
        )

        self.assertEqual(client.calls, [])

    def test_sufficient_evidence_produces_a_grounded_answer(self):
        answer = answer_knowledge_question(
            "What is the bowel prep for a colonoscopy?",
            StubRetriever([source(0.80)]),
            StubChatClient(),
            mode="primary",
        )

        self.assertEqual(answer.source, "rag")
        self.assertEqual(answer.status, "generated")
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.retrieval_latency_ms, 12.0)

    def test_weak_evidence_with_no_curated_answer_falls_back(self):
        answer = answer_knowledge_question(
            "What is a normal blood pressure reading?",
            StubRetriever([source(0.10)]),
            StubChatClient(),
            mode="primary",
        )

        self.assertEqual(answer.source, "fallback")
        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertEqual(answer.text, INSUFFICIENT_EVIDENCE_RESPONSE)

    def test_weak_evidence_with_a_curated_answer_returns_todays_behaviour(self):
        """The reason nothing regresses: the curated answer sits behind RAG."""

        answer = answer_knowledge_question(
            "What documents should I bring to my appointment?",
            StubRetriever([source(0.10)]),
            StubChatClient(),
            mode="primary",
        )

        self.assertEqual(answer.source, "curated")
        self.assertEqual(answer.status, "curated_fallback")
        self.assertIn("photo ID", answer.text)

    def test_shadow_mode_shows_the_curated_answer_but_still_retrieves(self):
        retriever = StubRetriever([source(0.80)])

        answer = answer_knowledge_question(
            "What documents should I bring to my appointment?",
            retriever,
            StubChatClient(),
            mode="shadow",
        )

        self.assertEqual(answer.source, "curated")
        self.assertEqual(answer.citations, ())
        self.assertEqual(len(retriever.calls), 1)
        self.assertTrue(any("shadow" in note for note in answer.notes))

    def test_no_results_falls_back_without_calling_the_model(self):
        client = StubChatClient()

        answer = answer_knowledge_question(
            "Something unrelated entirely.", StubRetriever([]), client, mode="primary"
        )

        self.assertEqual(answer.source, "fallback")
        self.assertEqual(client.calls, [])


class QueryAnalysisTests(unittest.TestCase):
    def test_a_question_is_placed_in_the_categories_it_could_belong_to(self):
        self.assertIn("imaging", infer_categories("Do I need to fast before an MRI?"))
        self.assertIn("endoscopy", infer_categories("What is the bowel prep?"))

    def test_a_question_can_belong_to_more_than_one_category(self):
        """A colonoscopy is filed under both endoscopy and screening."""

        found = infer_categories("How often do I need a colonoscopy screening?")

        self.assertIn("endoscopy", found)
        self.assertIn("screening", found)

    def test_a_question_naming_no_subject_infers_nothing(self):
        """'Cannot tell' must not read as 'mismatch' to the guard."""

        self.assertEqual(infer_categories("How long will it take?"), frozenset())

    def test_a_compound_question_splits(self):
        result = split_question(
            "How long do I fast, and should I take my morning tablet?"
        )

        self.assertTrue(result.is_compound)
        self.assertEqual(len(result.parts), 2)

    def test_a_conjunction_inside_a_list_does_not_split(self):
        result = split_question("Should I bring my ID and insurance card?")

        self.assertFalse(result.is_compound)

    def test_a_split_that_loses_content_is_discarded(self):
        """Answering the whole thing imperfectly beats answering a piece of it."""

        result = split_question("What is a colonoscopy?")

        self.assertFalse(result.is_compound)
        self.assertEqual(result.parts, ("What is a colonoscopy?",))


class GuardTests(unittest.TestCase):
    def test_evidence_from_the_wrong_category_is_rejected(self):
        wrong = source(0.80, document_id="ct-scans", category="imaging")

        decision = check_evidence([wrong], question="What is the bowel prep?")

        self.assertIs(decision.verdict, EvidenceVerdict.WRONG_CATEGORY)
        self.assertEqual(decision.guard, "category_consistency")

    def test_the_category_guard_abstains_when_it_cannot_tell(self):
        decision = check_evidence([source(0.80)], question="How long will it take?")

        self.assertTrue(decision.sufficient)

    def test_a_lone_weak_match_is_rejected(self):
        decision = check_evidence(
            [source(0.57)], question="", isolated_similarity=0.65
        )

        self.assertIs(decision.verdict, EvidenceVerdict.ISOLATED_MATCH)
        self.assertEqual(decision.guard, "score_dispersion")

    def test_the_same_score_is_accepted_when_corroborated(self):
        """A cluster is evidence; one node on its own is a near-miss shape."""

        decision = check_evidence(
            [source(0.57), source(0.56)], question="", isolated_similarity=0.65
        )

        self.assertTrue(decision.sufficient)

    def test_answerability_can_only_remove_an_answer(self):
        sufficient = check_evidence([source(0.80), source(0.70)], question="")
        client = StubChatClient()
        client._create = lambda model, messages, temperature, max_tokens=3: type(
            "R", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": "NO"})()})()]}
        )()
        client.chat.completions.create = client._create

        downgraded = apply_answerability(sufficient, client, "Prepare for a PET scan?")

        self.assertIs(downgraded.verdict, EvidenceVerdict.NOT_ANSWERABLE)
        self.assertEqual(downgraded.guard, "answerability")

    def test_an_answerability_outage_does_not_become_a_refusal(self):
        sufficient = check_evidence([source(0.80), source(0.70)], question="")

        class Broken:
            chat = type("C", (), {"completions": type("X", (), {
                "create": staticmethod(lambda **kw: (_ for _ in ()).throw(RuntimeError()))})()})()

        self.assertTrue(apply_answerability(sufficient, Broken(), "q").sufficient)


if __name__ == "__main__":
    unittest.main()
