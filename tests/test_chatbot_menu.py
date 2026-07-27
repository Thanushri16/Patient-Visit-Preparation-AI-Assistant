import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import classify_intent, get_chatbot_response
from models import ConversationPhase, ConversationState, WorkflowType


class FakeIntentClient:
    def __init__(self, label: str):
        self.label = label
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.label)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class MenuRequestTests(unittest.TestCase):
    def test_unknown_intent_returns_unknown(self):
        client = FakeIntentClient("unknown")
        result = classify_intent("I have a general question about my visit", client)

        self.assertEqual(result["intent"], "unknown")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(len(client.calls), 1)

    def test_classifies_directly_with_model(self):
        client = FakeIntentClient("report_new_symptoms")
        result = classify_intent("I want to report new symptoms", client)

        self.assertEqual(result["intent"], "report_new_symptoms")
        self.assertGreater(result["confidence"], 0.0)
        self.assertEqual(result["status"], "model")
        self.assertEqual(len(client.calls), 1)

    def test_missing_client_does_not_use_keyword_matching(self):
        result = classify_intent("I want to report new symptoms", client=None)

        self.assertEqual(result["intent"], "unknown")
        self.assertEqual(result["status"], "unknown")

    def test_chatbot_routes_menu_message_through_model_classifier(self):
        client = FakeIntentClient("report_new_symptoms")
        state = ConversationState(session_id="session-123")

        response = get_chatbot_response([], "I need help with a new issue", client, state)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(state.workflow, WorkflowType.REPORT_NEW_SYMPTOMS)
        self.assertEqual(state.phase, ConversationPhase.COLLECTING)
        self.assertIn("describe your symptoms", response)


if __name__ == "__main__":
    unittest.main()
