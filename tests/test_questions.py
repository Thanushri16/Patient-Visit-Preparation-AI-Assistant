"""Unit tests for deterministic required-field and conditional question selection."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import ConversationState, VisitData, WorkflowType  # noqa: E402
from questions import AdaptiveQuestionResult, select_next_question  # noqa: E402
from workflow_schemas import refresh_state_completeness  # noqa: E402


class FakeStructuredClient:
    def __init__(self, parsed):
        self.calls = []
        self.parsed = parsed
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self.parsed))]
        )


class QuestionSelectionTests(unittest.TestCase):
    def test_selects_first_required_field_in_schema_order(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
        )
        refresh_state_completeness(state)

        selection = select_next_question(state)

        self.assertEqual(selection.field_path, "patient_name")
        self.assertEqual(selection.reason, "required")
        self.assertEqual(state.requested_field, "patient_name")

    def test_selects_contextual_allergy_reaction_question(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_ALLERGY,
            visit_data=VisitData(
                patient_name="Dana",
                date_of_birth="1984-06-05",
                email="dana@example.com",
                phone="555-0100",
                allergies=[{"allergen": "penicillin"}],
            ),
        )
        refresh_state_completeness(state)

        selection = select_next_question(state)

        self.assertEqual(selection.field_path, "allergies.0.reaction")
        self.assertEqual(selection.reason, "conditional")
        self.assertIn("penicillin", selection.question)

    def test_uses_llm_for_top_level_adaptive_follow_up(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.APPOINTMENT_PREPARATION,
            visit_data=VisitData(
                patient_name="Dana",
                date_of_birth="1984-06-05",
                email="dana@example.com",
                phone="555-0100",
            ),
        )
        refresh_state_completeness(state)
        client = FakeStructuredClient(
            AdaptiveQuestionResult(
                field_path="chief_complaint",
                question="What is the main reason for your visit today?",
            )
        )

        selection = select_next_question(state, client)

        self.assertEqual(selection.field_path, "chief_complaint")
        self.assertEqual(selection.reason, "adaptive")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(state.requested_field, "chief_complaint")

    def test_complete_workflow_clears_requested_field(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
            visit_data=VisitData(
                patient_name="Dana",
                date_of_birth="1984-06-05",
                email="dana@example.com",
                phone="555-0100",
                chief_complaint="headache",
                symptom_duration="three days",
                symptom_severity=5,
            ),
            requested_field="symptom_severity",
        )
        refresh_state_completeness(state)

        selection = select_next_question(state)

        self.assertIsNone(selection)
        self.assertIsNone(state.requested_field)


if __name__ == "__main__":
    unittest.main()
