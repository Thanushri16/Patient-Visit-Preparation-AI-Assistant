"""Unit tests for versioned prompts used by model-backed chain nodes."""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import VisitData, WorkflowType  # noqa: E402
from prompts import (  # noqa: E402
    CONFIRMATION_PROMPT_VERSION,
    EXTRACTOR_PROMPT_VERSION,
    build_confirmation_prompt,
    build_extractor_prompt,
)
from workflow_schemas import get_workflow_schema  # noqa: E402


class PromptBuilderTests(unittest.TestCase):
    def test_model_backed_prompts_have_independent_versions(self):
        versions = {
            EXTRACTOR_PROMPT_VERSION,
            CONFIRMATION_PROMPT_VERSION,
        }

        self.assertEqual(len(versions), 2)
        # Each prompt is versioned on its own so one can be revised without
        # invalidating the observability history of the others.
        self.assertTrue(
            all(re.search(r"_v\d+$", version) for version in versions),
            versions,
        )

    def test_extractor_receives_schema_state_and_latest_message(self):
        prompt = build_extractor_prompt(
            latest_message="It began three days ago",
            schema=get_workflow_schema(WorkflowType.REPORT_NEW_SYMPTOMS),
            current_data=VisitData(chief_complaint="headache"),
            requested_field="symptom_duration",
        )

        self.assertIn("It began three days ago", prompt)
        self.assertIn("symptom_duration", prompt)
        self.assertIn("headache", prompt)
        self.assertIn("requested_field", prompt)
        self.assertIn('"updates"', prompt)

    def test_confirmation_prompt_defines_one_structured_task(self):
        confirmation = build_confirmation_prompt("Yes, correct", "Chief complaint: headache")

        self.assertIn('"action"', confirmation)


if __name__ == "__main__":
    unittest.main()
