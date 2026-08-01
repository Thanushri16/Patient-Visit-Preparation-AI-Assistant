"""Unit and integration tests for structured extraction, validation, and state merging."""

import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import get_chatbot_response  # noqa: E402
from extraction import (  # noqa: E402
    extract_structured_fields,
    process_collection_turn,
    validate_and_merge_extraction,
)
from models import (  # noqa: E402
    ConversationPhase,
    ConversationState,
    FieldExtractionResult,
    VisitData,
    VisitDataPatch,
    WorkflowType,
)


class FakeStructuredClient:
    def __init__(self, parsed):
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(parse=self._parse),
        )
        self.parsed = parsed

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self.parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def symptom_state(visit_data=None):
    return ConversationState(
        session_id="session-123",
        workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
        phase=ConversationPhase.COLLECTING,
        visit_data=visit_data or VisitData(),
    )


class StructuredExtractionTests(unittest.TestCase):
    def test_extractor_uses_typed_response_format_and_relevant_state(self):
        expected = FieldExtractionResult(
            updates=VisitDataPatch(chief_complaint="headache")
        )
        client = FakeStructuredClient(expected)
        state = symptom_state()

        result = extract_structured_fields(client, state, "I have a headache")

        self.assertEqual(result, expected)
        self.assertIs(client.calls[0]["response_format"], FieldExtractionResult)
        self.assertEqual(client.calls[0]["temperature"], 0.0)
        self.assertIn("I have a headache", client.calls[0]["messages"][0]["content"])
        self.assertIn('"requested_field": null', client.calls[0]["messages"][0]["content"])

    def test_valid_updates_are_merged_and_completeness_is_refreshed(self):
        state = symptom_state(
            VisitData(
                patient_name="Dana",
                date_of_birth="1984-06-05",
                email="dana@example.com",
                phone="555-0100",
            )
        )
        extraction = FieldExtractionResult(
            updates=VisitDataPatch(
                chief_complaint="headache",
                symptom_duration="three days",
                symptom_severity=6,
            )
        )

        result = validate_and_merge_extraction(
            state,
            extraction,
            today=date(2026, 7, 26),
        )

        self.assertEqual(
            result.accepted_fields,
            ["chief_complaint", "symptom_duration", "symptom_severity"],
        )
        self.assertEqual(result.missing_fields, [])
        self.assertEqual(state.visit_data.symptom_severity, 6)

    def test_correction_replaces_existing_value(self):
        state = symptom_state(VisitData(chief_complaint="headache"))
        extraction = FieldExtractionResult(
            corrections=VisitDataPatch(chief_complaint="migraine")
        )

        result = validate_and_merge_extraction(state, extraction)

        self.assertEqual(result.errors, {})
        self.assertEqual(state.visit_data.chief_complaint, "migraine")

    def test_update_cannot_silently_overwrite_existing_value(self):
        state = symptom_state(VisitData(chief_complaint="headache"))
        extraction = FieldExtractionResult(
            updates=VisitDataPatch(chief_complaint="migraine")
        )

        result = validate_and_merge_extraction(state, extraction)

        self.assertIn("chief_complaint", result.errors)
        self.assertEqual(state.visit_data.chief_complaint, "headache")

    def test_field_outside_active_workflow_is_rejected(self):
        state = symptom_state()
        extraction = FieldExtractionResult(
            updates=VisitDataPatch(address={"city": "Pittsburgh"})
        )

        result = validate_and_merge_extraction(state, extraction)

        self.assertIn("address", result.rejected_fields)
        self.assertIsNone(state.visit_data.address)

    def test_invalid_email_is_not_merged(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.APPOINTMENT_PREPARATION,
            phase=ConversationPhase.COLLECTING,
        )
        extraction = FieldExtractionResult(
            updates=VisitDataPatch(email="not-an-email")
        )

        result = validate_and_merge_extraction(state, extraction)

        self.assertEqual(result.errors["email"], "Enter a valid email address.")
        self.assertIsNone(state.visit_data.email)

    def test_nested_correction_preserves_existing_address_fields(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.APPOINTMENT_PREPARATION,
            phase=ConversationPhase.COLLECTING,
            visit_data=VisitData(address={"city": "Pittsburgh"}),
        )
        extraction = FieldExtractionResult(
            corrections=VisitDataPatch(address={"street": "100 Main Street"})
        )

        validate_and_merge_extraction(state, extraction)

        self.assertEqual(state.visit_data.address.street, "100 Main Street")
        self.assertEqual(state.visit_data.address.city, "Pittsburgh")

    def test_uncertain_field_requests_clarification_without_merging(self):
        state = symptom_state()
        extraction = FieldExtractionResult(uncertain_fields=["symptom_duration"])

        result = validate_and_merge_extraction(state, extraction)

        self.assertEqual(result.errors["symptom_duration"], "Please clarify this value.")
        self.assertIsNone(state.visit_data.symptom_duration)

    def test_collection_turn_asks_for_next_missing_field(self):
        client = FakeStructuredClient(
            FieldExtractionResult(
                updates=VisitDataPatch(patient_name="Dana")
            )
        )
        state = symptom_state()

        result = process_collection_turn(client, state, "Dana")

        self.assertEqual(state.visit_data.patient_name, "Dana")
        self.assertIn("What is your date of birth?", result.response)

    def test_extraction_failure_does_not_modify_visit_data(self):
        state = symptom_state()

        result = process_collection_turn(None, state, "I have a headache")

        self.assertIsNone(state.visit_data.chief_complaint)
        self.assertEqual(state.extraction_retry_count, 1)
        self.assertIn("couldn't reliably capture", result.response)

    def test_chatbot_collection_path_uses_typed_extractor(self):
        client = FakeStructuredClient(
            FieldExtractionResult(
                updates=VisitDataPatch(patient_name="Dana")
            )
        )
        state = symptom_state()
        messages = []

        response = get_chatbot_response(
            messages,
            "Dana",
            client,
            state=state,
        )

        self.assertIn("What is your date of birth?", response)
        self.assertEqual(state.visit_data.patient_name, "Dana")
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
