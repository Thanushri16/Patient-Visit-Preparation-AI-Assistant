"""Unit tests for privacy-safe prompt-chain telemetry."""

import json
import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import ConversationState, VisitData, WorkflowType  # noqa: E402
from observability import LOGGER_NAME, emit_chain_event  # noqa: E402


class PromptChainObservabilityTests(unittest.TestCase):
    def test_event_contains_operational_metadata_without_patient_values(self):
        state = ConversationState(
            session_id="raw-private-session",
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
            visit_data=VisitData(patient_name="Sensitive Patient", chief_complaint="headache"),
        )
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.INFO)

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
            event = emit_chain_event(
                state,
                "validation_merge",
                success=True,
                latency_ms=12.5,
                metadata={"accepted_field_count": 2},
            )

        log_payload = captured.output[0]
        self.assertNotIn("raw-private-session", log_payload)
        self.assertNotIn("Sensitive Patient", log_payload)
        self.assertNotIn("headache", log_payload)
        self.assertEqual(len(event.session_reference), 16)
        self.assertEqual(event.metadata["accepted_field_count"], 2)

        json_text = captured.records[0].getMessage()
        self.assertEqual(json.loads(json_text)["node"], "validation_merge")


if __name__ == "__main__":
    unittest.main()
