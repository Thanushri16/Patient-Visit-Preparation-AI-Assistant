"""Unit tests for resolving a scenario's stated precondition into setup turns."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluators.benchmarks.preconditions import (  # noqa: E402
    resolve_scenario_messages,
    split_precondition,
)


class SplitPreconditionTests(unittest.TestCase):
    def test_leading_parenthetical_is_separated_from_the_scored_message(self):
        precondition, remainder = split_precondition(
            "(Summary shown) 'Yes, everything looks correct'"
        )

        self.assertEqual(precondition, "Summary shown")
        self.assertEqual(remainder, "'Yes, everything looks correct'")

    def test_an_ordinary_message_is_left_alone(self):
        precondition, remainder = split_precondition("I have a headache")

        self.assertEqual(precondition, "")
        self.assertEqual(remainder, "I have a headache")


class ResolveScenarioMessagesTests(unittest.TestCase):
    def test_shown_summary_is_actually_shown_before_it_is_confirmed(self):
        setup, scored = resolve_scenario_messages(
            ("(Summary shown) 'Yes, everything looks correct'",)
        )

        # Without this the scored turn confirms a summary that never existed.
        self.assertTrue(setup)
        self.assertIn("summary", setup[-1].lower())
        self.assertEqual(scored, ("Yes, everything looks correct",))

    def test_named_content_in_the_precondition_is_replayed(self):
        setup, scored = resolve_scenario_messages(
            ("(After stating they take lisinopril) Yes, once a day",)
        )

        self.assertEqual(setup, ("I take lisinopril.",))
        self.assertEqual(scored, ("Yes, once a day",))

    def test_capitalisation_of_quoted_content_is_preserved(self):
        setup, _ = resolve_scenario_messages(
            ("(User corrected provider from Smith to Jones) Generate summary",)
        )

        self.assertEqual(setup, ("My provider is Dr. Smith.", "Actually it's Dr. Jones."))

    def test_a_message_described_by_length_is_generated_to_that_length(self):
        _, scored = resolve_scenario_messages(("(2000+ character message with mixed topics)",))

        self.assertGreaterEqual(len(scored[0]), 2000)
        self.assertIn("appointment", scored[0])

    def test_an_ordinary_scenario_gains_no_setup(self):
        setup, scored = resolve_scenario_messages(("I have a cough",))

        self.assertEqual(setup, ())
        self.assertEqual(scored, ("I have a cough",))

    def test_scored_turns_are_never_empty(self):
        setup, scored = resolve_scenario_messages(("(Internal)",))

        self.assertTrue(scored)
        self.assertNotIn(scored[0], setup)

    def test_multi_turn_scenarios_keep_every_scored_turn(self):
        setup, scored = resolve_scenario_messages(
            ("I take metformin 500mg twice daily.", "What medications do I have listed?")
        )

        self.assertEqual(setup, ())
        self.assertEqual(len(scored), 2)


if __name__ == "__main__":
    unittest.main()
