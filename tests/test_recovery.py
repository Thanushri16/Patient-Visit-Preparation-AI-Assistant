"""Unit tests for bounded extraction, validation, and confirmation recovery."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import get_chatbot_response  # noqa: E402
from extraction import process_collection_turn  # noqa: E402
from models import (  # noqa: E402
    ConversationPhase,
    ConversationState,
    FieldExtractionResult,
    VisitData,
    VisitDataPatch,
    WorkflowType,
)
from summary_workflow import begin_summary_review  # noqa: E402


class FakeStructuredClient:
    def __init__(self, parsed):
        self.parsed = parsed
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self.parsed))]
        )


class BoundedRecoveryTests(unittest.TestCase):
    def test_second_extraction_failure_uses_bounded_fallback(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
            phase=ConversationPhase.COLLECTING,
        )

        process_collection_turn(None, state, "first attempt")
        second = process_collection_turn(None, state, "second attempt")

        self.assertEqual(state.extraction_retry_count, 2)
        self.assertIn("still couldn't reliably capture", second.response)
        self.assertIn("menu or restart", second.response)

    def test_third_validation_failure_uses_direct_fallback(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.APPOINTMENT_PREPARATION,
            phase=ConversationPhase.COLLECTING,
        )
        client = FakeStructuredClient(
            FieldExtractionResult(fields=VisitDataPatch(email="invalid"))
        )

        process_collection_turn(client, state, "invalid")
        process_collection_turn(client, state, "invalid")
        third = process_collection_turn(client, state, "invalid")

        self.assertEqual(state.validation_attempt_count, 3)
        self.assertIn("still couldn't record the email", third.response)
        self.assertIn("menu or restart", third.response)

    def test_third_unclear_confirmation_uses_bounded_fallback(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
            visit_data=VisitData(chief_complaint="headache"),
        )
        begin_summary_review(state)
        history = []

        get_chatbot_response(history, "maybe", None, state)
        get_chatbot_response(history, "perhaps", None, state)
        third = get_chatbot_response(history, "not sure", None, state)

        self.assertEqual(state.confirmation_attempt_count, 3)
        self.assertIn("still couldn't determine", third)
        self.assertEqual(state.phase, ConversationPhase.AWAITING_CONFIRMATION)


if __name__ == "__main__":
    unittest.main()
