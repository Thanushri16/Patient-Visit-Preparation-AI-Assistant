"""Unit and integration tests for summaries, confirmation, and correction loops."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import get_chatbot_response  # noqa: E402
from models import (  # noqa: E402
    ConfirmationAction,
    ConfirmationResult,
    ConversationPhase,
    ConversationState,
    FieldExtractionResult,
    VisitData,
    VisitDataPatch,
    WorkflowType,
)
from summary_workflow import (  # noqa: E402
    begin_summary_review,
    build_summary_text,
    classify_confirmation,
)


class FakeStructuredClient:
    def __init__(self, parsed_results):
        self.parsed_results = list(parsed_results)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(parse=self._parse),
        )

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self.parsed_results.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
        )


def complete_symptom_state():
    return ConversationState(
        session_id="session-123",
        workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
        phase=ConversationPhase.COLLECTING,
        visit_data=VisitData(
            chief_complaint="headache",
            symptom_duration="three days",
            symptom_severity=6,
        ),
    )


class SummaryWorkflowTests(unittest.TestCase):
    def test_summary_uses_only_canonical_provided_data(self):
        summary = build_summary_text(
            VisitData(
                chief_complaint="headache",
                current_medications=[],
            )
        )

        self.assertIn("Chief complaint: headache", summary)
        self.assertIn("Current medications: None reported", summary)
        self.assertNotIn("Date of birth", summary)

    def test_begin_review_stores_summary_and_changes_phase(self):
        state = complete_symptom_state()

        response = begin_summary_review(state)

        self.assertEqual(state.phase, ConversationPhase.AWAITING_CONFIRMATION)
        self.assertEqual(state.summary_text, response.split("\n\nIs this summary")[0])
        self.assertIn("Is this summary correct?", response)

    def test_confirmation_uses_deterministic_common_phrases(self):
        result = classify_confirmation(None, "Yes, correct", "summary")

        self.assertEqual(result.action, ConfirmationAction.CONFIRM)

    def test_correction_text_is_preserved(self):
        result = classify_confirmation(
            None,
            "Actually, change the duration to four days",
            "summary",
        )

        self.assertEqual(result.action, ConfirmationAction.CORRECT)
        self.assertIn("four days", result.correction_text)

    def test_ambiguous_confirmation_uses_typed_classifier(self):
        client = FakeStructuredClient(
            [ConfirmationResult(action=ConfirmationAction.CONFIRM)]
        )

        result = classify_confirmation(client, "Everything seems fine", "summary")

        self.assertEqual(result.action, ConfirmationAction.CONFIRM)
        self.assertIs(client.calls[0]["response_format"], ConfirmationResult)

    def test_collection_completion_immediately_starts_summary_review(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
            phase=ConversationPhase.COLLECTING,
        )
        client = FakeStructuredClient(
            [
                FieldExtractionResult(
                    updates=VisitDataPatch(
                        chief_complaint="headache",
                        symptom_duration="three days",
                        symptom_severity=6,
                    )
                )
            ]
        )

        response = get_chatbot_response([], "I have had a level 6 headache for three days", client, state)

        self.assertEqual(state.phase, ConversationPhase.AWAITING_CONFIRMATION)
        self.assertIn("appointment summary", response.lower())
        self.assertIn("Is this summary correct?", response)

    def test_confirmed_summary_moves_to_completed_without_persisting(self):
        state = complete_symptom_state()
        begin_summary_review(state)

        response = get_chatbot_response([], "yes", client=None, state=state)

        self.assertEqual(state.phase, ConversationPhase.COMPLETED)
        self.assertTrue(state.confirmed)
        self.assertIn("not been submitted", response)

    def test_natural_language_correction_revalidates_and_regenerates_summary(self):
        state = complete_symptom_state()
        begin_summary_review(state)
        client = FakeStructuredClient(
            [
                FieldExtractionResult(
                    corrections=VisitDataPatch(symptom_duration="four days")
                )
            ]
        )

        response = get_chatbot_response(
            [],
            "Actually, change the duration to four days",
            client,
            state,
        )

        self.assertEqual(state.visit_data.symptom_duration, "four days")
        self.assertEqual(state.phase, ConversationPhase.AWAITING_CONFIRMATION)
        self.assertIn("Symptom duration: four days", response)

    def test_view_summary_menu_option_displays_current_state_immediately(self):
        state = ConversationState(
            session_id="session-123",
            visit_data=VisitData(chief_complaint="headache"),
        )

        response = get_chatbot_response([], "7", client=None, state=state)

        self.assertEqual(state.phase, ConversationPhase.AWAITING_CONFIRMATION)
        self.assertIn("Chief complaint: headache", response)


if __name__ == "__main__":
    unittest.main()
