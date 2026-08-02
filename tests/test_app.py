"""HTTP-level tests for the FastAPI chatbot UI."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app as app_module  # noqa: E402

app = app_module.app


class ChatbotUiTests(unittest.TestCase):
    def test_landing_page_embeds_initial_greeting(self):
        client = TestClient(app)

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("healthcare assistant", response.text)
        self.assertIn("What can I help you with today?", response.text)

    def test_chat_response_exposes_benchmark_contract(self):
        client = TestClient(app)
        with (
            patch.object(app_module, "client", object()),
            patch.object(app_module, "get_chatbot_response", return_value="How can I help?"),
        ):
            response = client.post(
                "/chat",
                json={"message": "Hello", "session_id": "benchmark-contract"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["reply"], "How can I help?")
        self.assertEqual(payload["intent"], "unknown")
        self.assertIn("state", payload)
        self.assertFalse(payload["is_emergency"])
        self.assertFalse(payload["safety_triggered"])

    def test_empty_message_preserves_benchmark_contract(self):
        client = TestClient(app)

        response = client.post(
            "/chat",
            json={"message": "", "session_id": "empty-message-contract"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {"reply", "intent", "state", "is_emergency", "safety_triggered"},
        )


if __name__ == "__main__":
    unittest.main()
