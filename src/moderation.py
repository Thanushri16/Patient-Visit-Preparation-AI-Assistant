import re
from dataclasses import dataclass

try:
    from .chatbot_content import (
        BLOCKED_RESPONSE,
        EMERGENCY_ESCALATION_RESPONSE,
        EMERGENCY_SYMPTOM_PATTERNS,
        HIGH_RISK_CRISIS_PATTERNS,
        INPUT_BLOCK_LIST,
        MEDICAL_POLICY_VIOLATION_PATTERNS,
        OUTPUT_BLOCK_LIST,
        OUTPUT_SAFETY_REDIRECT_RESPONSE,
        PROMPT_INJECTION_PATTERNS,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from chatbot_content import (
        BLOCKED_RESPONSE,
        EMERGENCY_ESCALATION_RESPONSE,
        EMERGENCY_SYMPTOM_PATTERNS,
        HIGH_RISK_CRISIS_PATTERNS,
        INPUT_BLOCK_LIST,
        MEDICAL_POLICY_VIOLATION_PATTERNS,
        OUTPUT_BLOCK_LIST,
        OUTPUT_SAFETY_REDIRECT_RESPONSE,
        PROMPT_INJECTION_PATTERNS,
    )


@dataclass
class ModerationDecision:
    stage: str
    action: str
    risk_level: str
    risk_score: int
    reasons: list[str]
    response: str | None = None


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _regex_matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _substring_matches_any(phrases: list[str], text: str) -> bool:
    return any(normalize_text(phrase) in text for phrase in phrases)


def _risk_level(score: int) -> str:
    if score >= 90:
        return "critical"
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def moderate_text(text: str, stage: str) -> ModerationDecision:
    normalized = normalize_text(text)
    reasons: list[str] = []
    risk_score = 0

    has_prompt_injection = _regex_matches_any(PROMPT_INJECTION_PATTERNS, normalized)
    has_input_blocklist_phrase = _substring_matches_any(INPUT_BLOCK_LIST, normalized)
    has_output_blocklist_phrase = _substring_matches_any(OUTPUT_BLOCK_LIST, normalized)
    has_emergency_symptom = _regex_matches_any(EMERGENCY_SYMPTOM_PATTERNS, normalized)
    has_crisis_signal = _regex_matches_any(HIGH_RISK_CRISIS_PATTERNS, normalized)
    has_medical_policy_violation = _regex_matches_any(MEDICAL_POLICY_VIOLATION_PATTERNS, normalized)

    if has_prompt_injection:
        reasons.append("prompt_injection_pattern")
        risk_score = max(risk_score, 80)
    if stage == "input" and has_input_blocklist_phrase:
        reasons.append("input_block_list_phrase")
        risk_score = max(risk_score, 70)
    if stage == "output" and has_output_blocklist_phrase:
        reasons.append("output_block_list_phrase")
        risk_score = max(risk_score, 80)
    if stage == "input" and has_emergency_symptom:
        reasons.append("emergency_symptom_detected")
        risk_score = max(risk_score, 95)
    if stage == "input" and has_crisis_signal:
        reasons.append("high_risk_crisis_detected")
        risk_score = max(risk_score, 99)
    if has_medical_policy_violation:
        reasons.append("medical_policy_violation")
        risk_score = max(risk_score, 75 if stage == "output" else 65)

    risk_level = _risk_level(risk_score)
    action = "allow"
    response = None

    if stage == "input" and ("emergency_symptom_detected" in reasons or "high_risk_crisis_detected" in reasons):
        action = "escalate"
        response = EMERGENCY_ESCALATION_RESPONSE
    elif stage == "output" and "medical_policy_violation" in reasons and not {
        "prompt_injection_pattern",
        "output_block_list_phrase",
    }.intersection(reasons):
        action = "sanitize"
        response = OUTPUT_SAFETY_REDIRECT_RESPONSE
    elif risk_score >= 70:
        action = "block"
        response = BLOCKED_RESPONSE

    return ModerationDecision(
        stage=stage,
        action=action,
        risk_level=risk_level,
        risk_score=risk_score,
        reasons=reasons,
        response=response,
    )
