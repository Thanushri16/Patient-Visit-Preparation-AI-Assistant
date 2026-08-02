"""Unit tests for how recorded visit data is put into words for the patient."""

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extraction import describe_field_value  # noqa: E402
from models import VisitData  # noqa: E402
from rendering import render_value  # noqa: E402
from summary_workflow import build_summary_text  # noqa: E402


class BooleanRenderingTests(unittest.TestCase):
    def test_internal_flags_never_surface_as_true_or_false(self):
        insurance = VisitData(
            insurance_info={
                "provider_name": "BCBS",
                "plan_type": "high deductible",
                "has_insurance": True,
                "details_available": False,
            }
        ).insurance_info

        rendered = render_value("insurance_info", insurance)

        # The reported bug: "insurance: bcbs, True, False" told the patient
        # nothing and prompted "what is true and false here?".
        self.assertNotIn("True", rendered)
        self.assertNotIn("False", rendered)
        self.assertIn("BCBS", rendered)
        self.assertIn("insurance on file", rendered)

    def test_a_flag_the_record_contradicts_is_not_shown(self):
        with_details = VisitData(
            insurance_info={
                "provider_name": "BCBS",
                "plan_type": "high deductible",
                "details_available": False,
            }
        ).insurance_info
        without_details = VisitData(
            insurance_info={"provider_name": "BCBS", "details_available": False}
        ).insurance_info

        # Saying "details not to hand" straight after listing the details is
        # worse than saying nothing; with no details it is useful information.
        self.assertNotIn("not to hand", render_value("insurance_info", with_details))
        self.assertIn("not to hand", render_value("insurance_info", without_details))

    def test_nested_values_are_labelled_by_their_part(self):
        insurance = VisitData(
            insurance_info={"provider_name": "Aetna", "member_id": "AE-88213"}
        ).insurance_info

        rendered = render_value("insurance_info", insurance)

        # `provider_name` means the carrier here, not the clinician.
        self.assertIn("carrier Aetna", rendered)
        self.assertIn("member ID AE-88213", rendered)

    def test_a_standalone_boolean_field_reads_as_a_statement(self):
        self.assertEqual(
            render_value("new_patient", True), "first visit with this provider"
        )
        self.assertEqual(render_value("new_patient", False), "returning patient")


class ListRenderingTests(unittest.TestCase):
    def test_an_explicit_denial_reads_as_the_answer_it_is(self):
        self.assertEqual(
            render_value("allergies", []), "no known drug allergies (NKDA)"
        )
        self.assertEqual(render_value("current_medications", []), "no medications")

    def test_medications_render_with_dose_frequency_and_purpose(self):
        data = VisitData(
            current_medications=[
                {"name": "metformin", "dosage": "500mg", "frequency": "twice daily", "purpose": "diabetes"}
            ]
        )

        rendered = render_value("current_medications", data.current_medications)

        self.assertEqual(rendered, "metformin 500mg twice daily for diabetes")

    def test_allergies_render_with_their_reaction(self):
        data = VisitData(allergies=[{"allergen": "penicillin", "reaction": "rash"}])

        self.assertEqual(
            render_value("allergies", data.allergies), "penicillin (rash)"
        )


class ScalarRenderingTests(unittest.TestCase):
    def test_unanswered_fields_render_as_nothing(self):
        self.assertIsNone(render_value("provider_name", None))

    def test_dates_and_severity_use_their_conventional_form(self):
        self.assertEqual(render_value("date_of_birth", date(1984, 6, 5)), "1984-06-05")
        self.assertEqual(render_value("symptom_severity", 7), "7/10")


class SurfaceConsistencyTests(unittest.TestCase):
    """Every user-facing surface must describe a value the same way."""

    def setUp(self):
        self.data = VisitData(
            insurance_info={
                "provider_name": "BCBS",
                "plan_type": "high deductible",
                "has_insurance": True,
                "details_available": False,
            },
            allergies=[],
        )

    def test_the_turn_acknowledgement_is_readable(self):
        described = describe_field_value("insurance_info", self.data.insurance_info)

        self.assertNotIn("True", described)
        self.assertNotIn("False", described)
        self.assertTrue(described.startswith("insurance:"))

    def test_the_summary_is_readable(self):
        summary = build_summary_text(self.data)

        self.assertNotIn("True", summary)
        self.assertNotIn("False", summary)
        self.assertIn("BCBS", summary)

    def test_an_explicit_none_is_not_labelled_twice(self):
        # "allergies: no known drug allergies" would be a stutter.
        self.assertEqual(
            describe_field_value("allergies", []), "no known drug allergies (NKDA)"
        )


if __name__ == "__main__":
    unittest.main()
