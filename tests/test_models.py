"""Unit tests for typed healthcare chatbot domain models."""

import sys
import unittest
from datetime import date
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import (  # noqa: E402
    Allergy,
    ChatMessage,
    ChatSession,
    ConversationPhase,
    ConversationState,
    Measurement,
    Medication,
    VisitData,
    WorkflowType,
)


class VisitDataModelTests(unittest.TestCase):
    def test_defaults_distinguish_unanswered_fields_from_explicit_none(self):
        unanswered = VisitData()
        explicit_none = VisitData(
            medical_conditions=[],
            current_medications=[],
            allergies=[],
        )

        self.assertIsNone(unanswered.medical_conditions)
        self.assertIsNone(unanswered.current_medications)
        self.assertIsNone(unanswered.allergies)
        self.assertEqual(explicit_none.medical_conditions, [])
        self.assertEqual(explicit_none.current_medications, [])
        self.assertEqual(explicit_none.allergies, [])

    def test_accepts_typed_nested_visit_data(self):
        visit_data = VisitData(
            patient_name="  Dana  ",
            date_of_birth="1984-06-05",
            height=Measurement(value=6, unit="ft"),
            weight={"value": 140, "unit": "lb"},
            symptom_severity=6,
            current_medications=[Medication(name="Medicine A", frequency="daily")],
            allergies=[Allergy(allergen="penicillin", reaction="rash")],
        )

        self.assertEqual(visit_data.patient_name, "Dana")
        self.assertEqual(visit_data.date_of_birth, date(1984, 6, 5))
        self.assertEqual(visit_data.weight.value, 140)
        self.assertEqual(visit_data.current_medications[0].name, "Medicine A")

    def test_rejects_out_of_range_severity_and_unknown_fields(self):
        with self.assertRaises(ValidationError):
            VisitData(symptom_severity=11)

        with self.assertRaises(ValidationError):
            VisitData(unsupported_field="value")


class ConversationStateModelTests(unittest.TestCase):
    def test_state_tracks_workflow_progress_outside_message_memory(self):
        state = ConversationState(
            session_id="session-123",
            phase=ConversationPhase.COLLECTING,
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
            visit_data={"chief_complaint": "headache"},
            missing_fields=["symptom_duration", "symptom_severity"],
        )

        self.assertEqual(state.phase, ConversationPhase.COLLECTING)
        self.assertEqual(state.workflow, WorkflowType.REPORT_NEW_SYMPTOMS)
        self.assertEqual(state.visit_data.chief_complaint, "headache")
        self.assertFalse(state.confirmed)

    def test_session_keeps_messages_separate_from_structured_state(self):
        session = ChatSession(
            state=ConversationState(session_id="session-123"),
            messages=[ChatMessage(role="user", content="I have a headache")],
            expires_at=1_900_000_000,
        )

        self.assertEqual(session.messages[0].content, "I have a headache")
        self.assertIsNone(session.state.visit_data.chief_complaint)

    def test_retry_counts_cannot_be_negative(self):
        with self.assertRaises(ValidationError):
            ConversationState(session_id="session-123", extraction_retry_count=-1)


if __name__ == "__main__":
    unittest.main()
