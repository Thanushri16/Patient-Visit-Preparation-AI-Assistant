"""Unit tests for the workflow menu and intent metadata catalog."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import WorkflowType  # noqa: E402
from workflow_catalog import (  # noqa: E402
    INTENT_LABELS,
    MENU_OPTION_TO_WORKFLOW,
    MENU_PROMPT_RESPONSE,
    SHOW_MENU_COMMANDS,
    SHOW_MENU_INTENT,
    WORKFLOW_CATALOG,
    build_intent_classifier_prompt,
)


class WorkflowCatalogTests(unittest.TestCase):
    def test_catalog_covers_every_workflow_once(self):
        self.assertEqual(set(WORKFLOW_CATALOG), set(WorkflowType))
        self.assertEqual(len(MENU_OPTION_TO_WORKFLOW), len(WorkflowType))

    def test_menu_and_classifier_are_generated_from_catalog(self):
        classifier_prompt = build_intent_classifier_prompt("I need help")

        for workflow, definition in WORKFLOW_CATALOG.items():
            self.assertIn(
                f"{definition.menu_option}. {definition.menu_label}",
                MENU_PROMPT_RESPONSE,
            )
            self.assertIn(workflow.value, INTENT_LABELS)
            self.assertIn(workflow.value, classifier_prompt)
        self.assertIn(SHOW_MENU_INTENT, classifier_prompt)

    def test_help_alias_uses_the_show_menu_intent(self):
        self.assertIn("help", SHOW_MENU_COMMANDS)
        self.assertIn(SHOW_MENU_INTENT, INTENT_LABELS)


if __name__ == "__main__":
    unittest.main()
