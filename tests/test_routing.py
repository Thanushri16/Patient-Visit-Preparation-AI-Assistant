"""Unit tests for state-aware routing, global commands, and route transitions."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import get_chatbot_response  # noqa: E402
from models import ConversationPhase, ConversationState, VisitData, WorkflowType  # noqa: E402
from routing import RouteAction, route_message  # noqa: E402


class StateAwareRoutingTests(unittest.TestCase):
    def test_menu_option_starts_workflow_and_calculates_missing_fields(self):
        state = ConversationState(session_id="session-123")

        decision = route_message(state, "2")

        self.assertEqual(decision.action, RouteAction.START_WORKFLOW)
        self.assertEqual(state.workflow, WorkflowType.REPORT_NEW_SYMPTOMS)
        self.assertEqual(state.phase, ConversationPhase.COLLECTING)
        self.assertEqual(
            state.missing_fields,
            [
                "chief_complaint",
                "symptom_location",
                "symptom_severity",
                "symptom_duration",
                "symptom_onset",
                "symptom_pattern",
                "patient_name",
                "date_of_birth",
                "email",
                "phone",
            ],
        )

    def test_active_workflow_continues_without_reclassifying_answer(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
            phase=ConversationPhase.COLLECTING,
        )
        classifier_called = False

        def classifier(_text):
            nonlocal classifier_called
            classifier_called = True
            return {"intent": "medication_question", "confidence": 1.0}

        decision = route_message(state, "It started after taking medication", classifier)

        self.assertEqual(decision.action, RouteAction.CONTINUE)
        self.assertFalse(decision.handled)
        self.assertFalse(classifier_called)
        self.assertEqual(state.workflow, WorkflowType.REPORT_NEW_SYMPTOMS)

    def test_restart_clears_collected_data_and_returns_to_menu(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.APPOINTMENT_PREPARATION,
            phase=ConversationPhase.COLLECTING,
            visit_data=VisitData(patient_name="Dana"),
        )

        decision = route_message(state, "start over")

        self.assertEqual(decision.action, RouteAction.RESTART)
        self.assertEqual(state.phase, ConversationPhase.MENU)
        self.assertIsNone(state.workflow)
        self.assertIsNone(state.visit_data.patient_name)

    def test_cancel_preserves_collected_data(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.APPOINTMENT_PREPARATION,
            phase=ConversationPhase.COLLECTING,
            visit_data=VisitData(patient_name="Dana"),
        )

        route_message(state, "cancel")

        self.assertEqual(state.phase, ConversationPhase.MENU)
        self.assertIsNone(state.workflow)
        self.assertEqual(state.visit_data.patient_name, "Dana")

    def test_low_confidence_route_falls_back_to_menu(self):
        state = ConversationState(session_id="session-123")

        decision = route_message(
            state,
            "Something else",
            lambda _text: {"intent": "report_allergy", "confidence": 0.1},
        )

        self.assertEqual(decision.action, RouteAction.FALLBACK)
        self.assertEqual(state.phase, ConversationPhase.MENU)
        self.assertIn("rephrase", decision.response)

    def test_change_answer_returns_confirmation_state_to_collection(self):
        state = ConversationState(
            session_id="session-123",
            workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
            phase=ConversationPhase.AWAITING_CONFIRMATION,
            confirmed=True,
        )

        decision = route_message(state, "change my answer")

        self.assertEqual(decision.action, RouteAction.CHANGE_ANSWER)
        self.assertEqual(state.phase, ConversationPhase.COLLECTING)
        self.assertFalse(state.confirmed)

    def test_chatbot_applies_emergency_transition_before_routing(self):
        state = ConversationState(session_id="session-123")
        messages = []

        response = get_chatbot_response(
            messages,
            "I am having severe chest pain.",
            client=None,
            state=state,
        )

        self.assertIn("call 911", response.lower())
        self.assertEqual(state.phase, ConversationPhase.ESCALATED)
        self.assertEqual(state.workflow, WorkflowType.EMERGENCY_SUPPORT)

    def test_menu_command_cannot_bypass_emergency_state(self):
        state = ConversationState(
            session_id="session-123",
            phase=ConversationPhase.ESCALATED,
            workflow=WorkflowType.EMERGENCY_SUPPORT,
            emergency_detected=True,
        )

        decision = route_message(state, "show menu")

        self.assertEqual(decision.source, "emergency_state")
        self.assertEqual(state.phase, ConversationPhase.ESCALATED)
        self.assertIn("call 911", decision.response.lower())

    def test_chatbot_persists_routed_menu_exchange_in_memory(self):
        state = ConversationState(session_id="session-123")
        messages = []

        response = get_chatbot_response(messages, "1", client=None, state=state)

        self.assertIn("preparing for your appointment", response)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-2].role, "user")
        self.assertEqual(messages[-1].role, "assistant")


if __name__ == "__main__":
    unittest.main()
