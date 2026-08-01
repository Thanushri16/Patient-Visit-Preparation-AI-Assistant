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
        state = symptom_state()
        extraction = FieldExtractionResult(
            updates=VisitDataPatch(
                chief_complaint="headache",
                symptom_location="forehead",
                symptom_onset="Monday",
                symptom_duration="three days",
                symptom_severity=6,
                symptom_pattern="comes and goes",
            )
        )

        result = validate_and_merge_extraction(
            state,
            extraction,
            today=date(2026, 7, 26),
        )

        self.assertEqual(
            result.accepted_fields,
            [
                "chief_complaint",
                "symptom_location",
                "symptom_onset",
                "symptom_duration",
                "symptom_severity",
                "symptom_pattern",
            ],
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

    def test_update_to_an_answered_field_is_treated_as_a_correction(self):
        state = symptom_state(VisitData(chief_complaint="headache"))
        extraction = FieldExtractionResult(
            updates=VisitDataPatch(chief_complaint="migraine")
        )

        result = validate_and_merge_extraction(state, extraction)

        # The latest thing the patient said is the truth, whether or not the
        # extractor labelled it a correction. The overwrite is reported so the
        # reply can confirm it rather than change the record silently.
        self.assertEqual(result.errors, {})
        self.assertEqual(result.corrected_fields, ["chief_complaint"])
        self.assertEqual(state.visit_data.chief_complaint, "migraine")

    def test_repeated_medications_accumulate_and_fill_in_detail(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.MEDICATION_QUESTION,
            phase=ConversationPhase.COLLECTING,
            visit_data=VisitData(current_medications=[{"name": "lisinopril"}]),
        )
        extraction = FieldExtractionResult(
            updates=VisitDataPatch(
                current_medications=[
                    {"name": "lisinopril", "dosage": "10mg"},
                    {"name": "metformin"},
                ]
            )
        )

        validate_and_merge_extraction(state, extraction)

        recorded = state.visit_data.current_medications
        self.assertEqual([item.name for item in recorded], ["lisinopril", "metformin"])
        self.assertEqual(recorded[0].dosage, "10mg")

    def test_field_outside_active_workflow_is_dropped_without_telling_the_user(self):
        state = symptom_state()
        extraction = FieldExtractionResult(
            updates=VisitDataPatch(address={"city": "Pittsburgh"})
        )

        result = validate_and_merge_extraction(state, extraction)

        # It is not written, but it is also not an error the patient should see:
        # which fields a workflow collects is an internal concern, and surfacing
        # it asked people to restate information that was recorded correctly.
        self.assertIsNone(state.visit_data.address)
        self.assertEqual(result.errors, {})
        self.assertEqual(result.ignored_fields, ["address"])

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

    def test_uncertain_field_stays_unanswered_and_keeps_its_question_queued(self):
        state = symptom_state()
        extraction = FieldExtractionResult(uncertain_fields=["symptom_duration"])

        result = validate_and_merge_extraction(state, extraction)

        # An uncertain field is a gap to ask about, not a failed write, so it
        # produces no error and simply stays on the missing list.
        self.assertEqual(result.errors, {})
        self.assertIsNone(state.visit_data.symptom_duration)
        self.assertIn("symptom_duration", result.missing_fields)

    def test_duration_without_a_unit_is_rejected_for_clarification(self):
        state = symptom_state(VisitData(chief_complaint="rash"))
        extraction = FieldExtractionResult(
            updates=VisitDataPatch(symptom_duration="about 3")
        )

        result = validate_and_merge_extraction(state, extraction)

        self.assertIn("hours, days, weeks", result.errors["symptom_duration"])
        self.assertIsNone(state.visit_data.symptom_duration)

    def test_collection_turn_reports_what_it_recorded_and_asks_what_is_next(self):
        client = FakeStructuredClient(
            FieldExtractionResult(
                updates=VisitDataPatch(chief_complaint="headache")
            )
        )
        state = symptom_state()

        result = process_collection_turn(client, state, "I have a headache")

        self.assertEqual(state.visit_data.chief_complaint, "headache")
        self.assertIn("headache", result.response)
        self.assertIn("?", result.response)

    def test_extraction_failure_does_not_modify_visit_data(self):
        state = symptom_state()

        result = process_collection_turn(None, state, "I have a headache")

        self.assertIsNone(state.visit_data.chief_complaint)
        self.assertEqual(state.extraction_retry_count, 1)
        self.assertIn("couldn't reliably capture", result.response)

    def test_chatbot_collection_path_uses_typed_extractor(self):
        client = FakeStructuredClient(
            FieldExtractionResult(
                updates=VisitDataPatch(chief_complaint="headache")
            )
        )
        state = symptom_state()
        messages = []

        response = get_chatbot_response(
            messages,
            "I have a headache",
            client,
            state=state,
        )

        self.assertEqual(state.visit_data.chief_complaint, "headache")
        self.assertIn("headache", response)
        # One structured call extracts the fields, a second picks the follow-up.
        self.assertEqual(len(client.calls), 2)

    def test_placeholder_values_the_user_never_gave_are_discarded(self):
        client = FakeStructuredClient(
            FieldExtractionResult(
                updates=VisitDataPatch(
                    chief_complaint="cough",
                    provider_name="unknown",
                )
            )
        )
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.APPOINTMENT_PREPARATION,
            phase=ConversationPhase.COLLECTING,
        )

        process_collection_turn(client, state, "I have a cough")

        # The message never mentioned a provider, so "unknown" would be
        # fabricated data in the visit record.
        self.assertIsNone(state.visit_data.provider_name)
        self.assertEqual(state.visit_data.chief_complaint, "cough")

    def test_placeholder_is_kept_when_the_user_says_they_do_not_know(self):
        client = FakeStructuredClient(
            FieldExtractionResult(updates=VisitDataPatch(provider_name="unknown"))
        )
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.APPOINTMENT_PREPARATION,
            phase=ConversationPhase.COLLECTING,
        )

        process_collection_turn(client, state, "I'm not sure which doctor I'm seeing")

        self.assertEqual(state.visit_data.provider_name, "unknown")


if __name__ == "__main__":
    unittest.main()
