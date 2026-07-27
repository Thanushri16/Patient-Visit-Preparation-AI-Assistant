"""Unit tests for workflow schemas, completeness checks, and question selection."""

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import ConversationState, VisitData, WorkflowType  # noqa: E402
from workflow_schemas import (  # noqa: E402
    WORKFLOW_SCHEMAS,
    WorkflowSchema,
    get_missing_fields,
    get_conditional_missing_fields,
    get_next_missing_field,
    get_question_for_field,
    is_workflow_complete,
    refresh_state_completeness,
)


class WorkflowSchemaTests(unittest.TestCase):
    def test_every_workflow_has_a_schema(self):
        self.assertEqual(set(WORKFLOW_SCHEMAS), set(WorkflowType))

    def test_schema_rejects_unknown_and_overlapping_fields(self):
        with self.assertRaises(ValidationError):
            WorkflowSchema(
                workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
                required_fields=("not_a_visit_field",),
            )

        with self.assertRaises(ValidationError):
            WorkflowSchema(
                workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
                required_fields=("chief_complaint",),
                optional_fields=("chief_complaint",),
            )

    def test_missing_fields_follow_configured_collection_order(self):
        visit_data = VisitData(chief_complaint="headache")

        missing = get_missing_fields(WorkflowType.REPORT_NEW_SYMPTOMS, visit_data)

        self.assertEqual(missing, ["symptom_duration", "symptom_severity"])
        self.assertEqual(
            get_next_missing_field(WorkflowType.REPORT_NEW_SYMPTOMS, visit_data),
            "symptom_duration",
        )

    def test_appointment_requires_identity_and_contact_fields(self):
        missing = get_missing_fields(
            WorkflowType.APPOINTMENT_PREPARATION,
            VisitData(
                chief_complaint="annual checkup",
                symptom_duration="not applicable",
                symptom_severity=0,
                medical_conditions=[],
                current_medications=[],
                allergies=[],
            ),
        )

        self.assertEqual(missing, ["patient_name", "date_of_birth", "email", "phone"])

    def test_empty_lists_count_as_explicitly_answered(self):
        visit_data = VisitData(
            patient_name="Dana",
            date_of_birth="1984-06-05",
            email="dana@example.com",
            phone="555-0100",
            chief_complaint="annual checkup",
            symptom_duration="not applicable",
            symptom_severity=0,
            medical_conditions=[],
            current_medications=[],
            allergies=[],
        )

        self.assertTrue(
            is_workflow_complete(WorkflowType.APPOINTMENT_PREPARATION, visit_data)
        )

    def test_zero_severity_counts_as_answered(self):
        visit_data = VisitData(
            chief_complaint="follow-up",
            symptom_duration="one week",
            symptom_severity=0,
        )

        self.assertTrue(is_workflow_complete(WorkflowType.REPORT_NEW_SYMPTOMS, visit_data))

    def test_refresh_updates_application_owned_state(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_ALLERGY,
        )

        result = refresh_state_completeness(state)

        self.assertEqual(result, ["allergies"])
        self.assertEqual(state.missing_fields, ["allergies"])

    def test_schema_provides_deterministic_question_for_next_field(self):
        question = get_question_for_field(
            WorkflowType.REPORT_NEW_SYMPTOMS,
            "symptom_duration",
        )

        self.assertEqual(question, "How long have you had these symptoms?")

    def test_partial_address_adds_conditional_missing_fields(self):
        visit_data = VisitData(
            address={"city": "Pittsburgh"},
        )

        missing = get_conditional_missing_fields(
            WorkflowType.APPOINTMENT_PREPARATION,
            visit_data,
        )

        self.assertEqual(
            missing,
            ["address.street", "address.state", "address.postal_code"],
        )

    def test_reported_allergy_requires_reaction(self):
        visit_data = VisitData(allergies=[{"allergen": "penicillin"}])

        missing = get_conditional_missing_fields(
            WorkflowType.REPORT_ALLERGY,
            visit_data,
        )

        self.assertEqual(missing, ["allergies.0.reaction"])


if __name__ == "__main__":
    unittest.main()
