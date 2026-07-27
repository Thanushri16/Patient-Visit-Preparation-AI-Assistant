"""Unit tests for typed in-memory session state."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import (  # noqa: E402
    SESSION_TTL_SECONDS,
    conversation_histories,
    create_chat_session,
    get_or_create_chat_session,
    prune_expired_sessions,
)
from models import ChatMessage, ConversationPhase  # noqa: E402


class SessionStateTests(unittest.TestCase):
    def setUp(self):
        conversation_histories.clear()

    def tearDown(self):
        conversation_histories.clear()

    def test_new_session_contains_separate_state_and_message_memory(self):
        session = create_chat_session("session-123", now=1_000)

        self.assertEqual(session.state.session_id, "session-123")
        self.assertEqual(session.state.phase, ConversationPhase.MENU)
        self.assertEqual(session.messages, [])
        self.assertEqual(session.expires_at, 1_000 + SESSION_TTL_SECONDS)

    def test_get_or_create_returns_existing_session(self):
        original = get_or_create_chat_session("session-123", now=1_000)
        original.messages.append(ChatMessage(role="user", content="Hello"))

        retrieved = get_or_create_chat_session("session-123", now=1_100)

        self.assertIs(retrieved, original)
        self.assertEqual(retrieved.messages[0].content, "Hello")

    def test_prune_removes_only_expired_sessions(self):
        expired = create_chat_session("expired", now=1_000)
        active = create_chat_session("active", now=2_000)
        conversation_histories["expired"] = expired
        conversation_histories["active"] = active

        prune_expired_sessions(now=1_000 + SESSION_TTL_SECONDS)

        self.assertNotIn("expired", conversation_histories)
        self.assertIn("active", conversation_histories)

if __name__ == "__main__":
    unittest.main()
