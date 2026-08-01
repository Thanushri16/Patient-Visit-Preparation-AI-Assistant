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

        # Without a model client the deterministic schema question is used, and
        # the clinical complaint is what a symptom report opens with.
        selection = select_next_question(state)

        self.assertEqual(selection.field_path, "chief_complaint")
        self.assertEqual(selection.reason, "adaptive")
        self.assertEqual(state.requested_field, "chief_complaint")

    def test_selects_contextual_allergy_reaction_question(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_ALLERGY,
            visit_data=VisitData(allergies=[{"allergen": "penicillin"}]),
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
            visit_data=VisitData(patient_name="Dana"),
        )
        refresh_state_completeness(state)
        client = FakeStructuredClient(
            AdaptiveQuestionResult(
                field_path="visit_reason",
                question="What is the main reason for your visit today?",
            )
        )

        selection = select_next_question(state, client)

        self.assertEqual(selection.field_path, "visit_reason")
        self.assertEqual(selection.reason, "adaptive")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(state.requested_field, "visit_reason")

    def test_nested_detail_bypasses_the_adaptive_generator(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.MEDICATION_QUESTION,
            visit_data=VisitData(current_medications=[{"name": "metformin"}]),
        )
        refresh_state_completeness(state)
        client = FakeStructuredClient(
            AdaptiveQuestionResult(field_path="allergies", question="Any allergies?")
        )

        selection = select_next_question(state, client)

        # A named medication missing its dose has one precise question, so the
        # model is never consulted and cannot redirect to an unrelated field.
        self.assertEqual(selection.field_path, "current_medications.0.dosage")
        self.assertEqual(selection.reason, "conditional")
        self.assertIn("metformin", selection.question)
        self.assertEqual(client.calls, [])

    def test_complete_workflow_clears_requested_field(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
            visit_data=VisitData(
                chief_complaint="headache",
                symptom_location="forehead",
                symptom_onset="Monday",
                symptom_duration="three days",
                symptom_severity=5,
                symptom_pattern="comes and goes",
            ),
            requested_field="symptom_severity",
        )
        refresh_state_completeness(state)

        selection = select_next_question(state)

        self.assertIsNone(selection)
        self.assertIsNone(state.requested_field)


if __name__ == "__main__":
    unittest.main()
