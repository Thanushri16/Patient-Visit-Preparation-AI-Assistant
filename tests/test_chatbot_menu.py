import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatbot import classify_intent, handle_menu_request


class MenuRequestTests(unittest.TestCase):
    def test_menu_request_returns_healthcare_menu(self):
        response = handle_menu_request("show me the menu")
        self.assertIn("Appointment Preparation", response)
        self.assertIn("Report New Symptoms", response)
        self.assertIn("Emergency Support", response)

    def test_start_preparation_option_returns_guidance(self):
        response = handle_menu_request("1")
        self.assertIn("Let’s start preparing for your appointment", response)
        self.assertIn("What brings you in today", response)

    def test_fallback_intent_for_general_question(self):
        response = handle_menu_request("I have a general question")
        self.assertIsNone(response)

    def test_unknown_intent_returns_unknown(self):
        result = classify_intent("I have a general question about my visit")
        self.assertEqual(result["intent"], "unknown")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["status"], "unknown")

    def test_classify_intent_report_symptoms(self):
        result = classify_intent("I want to report new symptoms")
        self.assertEqual(result["intent"], "report_new_symptoms")
        self.assertGreater(result["confidence"], 0.0)
        self.assertEqual(result["status"], "confident")


if __name__ == "__main__":
    unittest.main()
