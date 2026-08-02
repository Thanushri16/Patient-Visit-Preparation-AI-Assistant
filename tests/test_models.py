"""Unit tests for typed healthcare chatbot domain models."""

import sys
import unittest
from datetime import date
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import (
    VisitDataPatch,  # noqa: E402
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
            date_of_birth="06/05/1984",
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

    def test_accepts_mm_dd_yyyy_date_patch_values(self):
        visit_data = VisitData(date_of_birth="06/05/1984")

        self.assertEqual(visit_data.date_of_birth, date(1984, 6, 5))

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


class DateOfBirthTests(unittest.TestCase):
    """A birth date is parsed by the application, never reformatted by a model."""

    def test_common_written_forms_are_accepted(self):
        for written in ("06/05/1984", "06-05-1984", "1984-06-05", "June 5, 1984"):
            with self.subTest(written=written):
                self.assertEqual(
                    VisitData(date_of_birth=written).date_of_birth, date(1984, 6, 5)
                )

    def test_a_mangled_date_is_rejected_rather_than_stored(self):
        # The extractor once returned "0605-04-06" for "born 06/05/1984". A wrong
        # birth date looks valid, so it has to be refused and asked for again.
        with self.assertRaises(ValidationError):
            VisitData(date_of_birth="0605-04-06")

    def test_the_extraction_patch_keeps_the_date_exactly_as_written(self):
        patch = VisitDataPatch(date_of_birth="06/05/1984")

        self.assertEqual(patch.date_of_birth, "06/05/1984")
        self.assertEqual(
            VisitData.model_validate({"date_of_birth": patch.date_of_birth}).date_of_birth,
            date(1984, 6, 5),
        )
