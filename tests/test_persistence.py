"""Unit and integration tests for confirmed-visit persistence."""

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import get_chatbot_response  # noqa: E402
from models import ConversationPhase, ConversationState, VisitData, WorkflowType  # noqa: E402
from persistence import JsonVisitRepository  # noqa: E402
from summary_workflow import begin_summary_review  # noqa: E402


def completed_state():
    state = ConversationState(
        session_id="private-session-id",
        workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
        phase=ConversationPhase.COMPLETED,
        confirmed=True,
        visit_data=VisitData(chief_complaint="headache"),
        summary_text="Summary: headache",
    )
    return state


class ConfirmedVisitPersistenceTests(unittest.TestCase):
    def test_only_confirmed_completed_visits_can_be_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonVisitRepository(Path(directory))
            state = completed_state()
            state.confirmed = False

            with self.assertRaises(ValueError):
                repository.save_confirmed(state)

    def test_save_uses_uuid_filename_atomic_record_and_hashed_session(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonVisitRepository(Path(directory))
            state = completed_state()

            path = repository.save_confirmed(state)
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(path.name, f"{state.visit_id}.json")
            self.assertNotIn("headache", path.name)
            self.assertNotEqual(payload["session_reference"], state.session_id)
            self.assertNotIn(state.session_id, path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "confirmed")
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_save_is_idempotent_for_same_confirmed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonVisitRepository(Path(directory))
            state = completed_state()

            first_path = repository.save_confirmed(state)
            second_path = repository.save_confirmed(state)

            self.assertEqual(first_path, second_path)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)

    def test_chatbot_confirmation_persists_when_repository_is_provided(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonVisitRepository(Path(directory))
            state = ConversationState(
                session_id="session-123",
                workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
                visit_data=VisitData(
                    chief_complaint="headache",
                    symptom_duration="three days",
                    symptom_severity=5,
                ),
            )
            begin_summary_review(state)

            response = get_chatbot_response(
                [],
                "yes",
                client=None,
                state=state,
                visit_repository=repository,
            )

            self.assertIsNotNone(state.visit_id)
            self.assertIsNotNone(state.persisted_at)
            self.assertIn("saved locally", response)
            self.assertTrue((Path(directory) / f"{state.visit_id}.json").exists())


if __name__ == "__main__":
    unittest.main()
