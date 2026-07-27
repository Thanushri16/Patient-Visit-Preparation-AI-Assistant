"""Prompt builder for classifying summary confirmation and correction requests."""

import json

from .common import HEALTHCARE_ROLE_AND_SAFETY_PROMPT

CONFIRMATION_PROMPT_VERSION = "confirmation_classifier_v1"


def build_confirmation_prompt(latest_message: str, displayed_summary: str) -> str:
    context = {
        "displayed_summary": displayed_summary,
        "latest_message": latest_message,
    }
    return f"""{HEALTHCARE_ROLE_AND_SAFETY_PROMPT}

Task: Classify whether the user confirms the summary, requests corrections, or is unclear.
Context: {json.dumps(context)}

Return JSON only:
{{"action": "confirm|correct|unclear", "correction_text": null}}
Include correction_text only when action is "correct".
""".strip()
