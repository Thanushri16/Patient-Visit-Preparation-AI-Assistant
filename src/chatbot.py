import json
import os
import re
from pathlib import Path

from openai import OpenAI

try:
    from .moderation import moderate_text
    from .chatbot_content import (
        APPOINTMENT_SUMMARY_FIELDS,
        APPOINTMENT_SUMMARY_HEADER,
        API_KEY_ERROR_MESSAGE,
        BLOCKED_RESPONSE,
        BOOTSTRAP_PROMPT,
        DEFAULT_MODEL,
        FEW_SHOT_EXAMPLES,
        FEW_SHOT_FALLBACK_THRESHOLD,
        GENERIC_ASSISTANT_PROMPT,
        INTAKE_SYSTEM_PROMPT,
        INTENT_CLASSIFIER_PROMPT_PREFIX_LINES,
        INTENT_CLASSIFIER_PROMPT_SUFFIX_LABEL,
        INPUT_BLOCK_LIST,
        INTENT_PATTERNS,
        INTENT_TO_MENU_OPTION,
        MENU_OPTIONS,
        MENU_PROMPT_RESPONSE,
        NOT_PROVIDED_TEXT,
        OUTPUT_BLOCK_LIST,
        PROMPT_INJECTION_PATTERNS,
        USE_FEW_SHOT_INTENT_CLASSIFIER,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from moderation import moderate_text
    from chatbot_content import (
        APPOINTMENT_SUMMARY_FIELDS,
        APPOINTMENT_SUMMARY_HEADER,
        API_KEY_ERROR_MESSAGE,
        BLOCKED_RESPONSE,
        BOOTSTRAP_PROMPT,
        DEFAULT_MODEL,
        FEW_SHOT_EXAMPLES,
        FEW_SHOT_FALLBACK_THRESHOLD,
        GENERIC_ASSISTANT_PROMPT,
        INTAKE_SYSTEM_PROMPT,
        INTENT_CLASSIFIER_PROMPT_PREFIX_LINES,
        INTENT_CLASSIFIER_PROMPT_SUFFIX_LABEL,
        INPUT_BLOCK_LIST,
        INTENT_PATTERNS,
        INTENT_TO_MENU_OPTION,
        MENU_OPTIONS,
        MENU_PROMPT_RESPONSE,
        NOT_PROVIDED_TEXT,
        OUTPUT_BLOCK_LIST,
        PROMPT_INJECTION_PATTERNS,
        USE_FEW_SHOT_INTENT_CLASSIFIER,
    )


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def is_blocked_text(text: str, block_list: list[str]) -> bool:
    if not isinstance(text, str):
        return False
    normalized_text = normalize_text(text)
    normalized_block_list = [normalize_text(block_phrase) for block_phrase in block_list]
    return any(block_phrase in normalized_text for block_phrase in normalized_block_list)


def classify_intent_rule(text: str) -> dict[str, object]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return {"intent": "unknown", "confidence": 0.0, "top_score": 0.0, "margin": 0.0, "status": "unknown"}

    scored_intents = []
    for intent, patterns in INTENT_PATTERNS.items():
        matches = sum(1 for keyword in patterns if keyword in normalized_text)
        score = matches / len(patterns) if patterns else 0.0
        scored_intents.append((intent, score))

    scored_intents.sort(key=lambda item: item[1], reverse=True)
    best_intent, top_score = scored_intents[0]

    if top_score == 0.0 or top_score < FEW_SHOT_FALLBACK_THRESHOLD:
        return {"intent": "unknown", "confidence": 0.0, "top_score": round(top_score, 2), "margin": 0.0, "status": "unknown"}

    confidence = round(top_score, 2)
    return {
        "intent": best_intent,
        "confidence": confidence,
        "top_score": round(top_score, 2),
        "margin": 0.0,
        "status": "confident",
    }


def classify_intent_few_shot(text: str, client: OpenAI | None) -> dict[str, object]:
    if client is None:
        return {"intent": "unknown", "confidence": 0.0, "top_score": 0.0, "margin": 0.0, "status": "unknown"}

    prompt_lines = list(INTENT_CLASSIFIER_PROMPT_PREFIX_LINES)

    for example in FEW_SHOT_EXAMPLES:
        prompt_lines.append(f"User: {example['text']}")
        prompt_lines.append(f"Intent: {example['intent']}")
        prompt_lines.append("")

    prompt_lines.append(f"User: {text}")
    prompt_lines.append(INTENT_CLASSIFIER_PROMPT_SUFFIX_LABEL)
    prompt = "\n".join(prompt_lines)

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    label = response.choices[0].message.content.strip().lower().splitlines()[0]
    if label not in INTENT_PATTERNS:
        return {"intent": "unknown", "confidence": 0.0, "top_score": 0.0, "margin": 0.0, "status": "unknown"}

    return {"intent": label, "confidence": 0.85, "top_score": 0.85, "margin": 0.0, "status": "few_shot"}


def classify_intent(text: str, client: OpenAI | None = None) -> dict[str, object]:
    rule_result = classify_intent_rule(text)
    if rule_result["status"] != "unknown" or not USE_FEW_SHOT_INTENT_CLASSIFIER:
        return rule_result
    if client is None:
        return rule_result

    few_shot_result = classify_intent_few_shot(text, client)
    if few_shot_result["status"] == "few_shot":
        return few_shot_result
    return rule_result


def contains_prompt_injection_attempt(text: str) -> bool:
    if not isinstance(text, str):
        return False
    normalized_text = normalize_text(text)
    return any(re.search(pattern, normalized_text) for pattern in PROMPT_INJECTION_PATTERNS)


def handle_menu_request(text: str, client: OpenAI | None = None):
    if not isinstance(text, str):
        return None

    normalized_text = normalize_text(text)
    if normalized_text in MENU_OPTIONS:
        return MENU_OPTIONS[normalized_text]["response"]

    intent_result = classify_intent(text, client)
    intent = intent_result["intent"]

    if intent == "show_menu":
        return MENU_PROMPT_RESPONSE

    if intent in INTENT_TO_MENU_OPTION:
        option_key = INTENT_TO_MENU_OPTION[intent]
        return MENU_OPTIONS[option_key]["response"]

    return None


def load_api_key():
    project_src_root = Path(__file__).resolve().parent
    key_file = project_src_root / "openaikey.txt"

    if key_file.exists():
        return key_file.read_text().strip()

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    raise RuntimeError(
        API_KEY_ERROR_MESSAGE
    )


def extract_json_payload(response_text):
    if not isinstance(response_text, str) or not response_text.strip():
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1)
    else:
        candidate = response_text

    brace_match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not brace_match:
        return None

    try:
        payload = json.loads(brace_match.group(0))
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload

    return None


def save_summary_to_db(payload):
    project_root = Path(__file__).resolve().parent.parent
    db_dir = project_root / "db"
    db_dir.mkdir(exist_ok=True)

    if isinstance(payload, str):
        payload = extract_json_payload(payload)

    if not isinstance(payload, dict):
        return None

    patient_name = payload.get("patient_name") or payload.get("name") or "unknown_patient"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(patient_name)).strip("._-") or "unknown_patient"
    file_path = db_dir / f"{safe_name}.json"

    existing_payload = {}
    if file_path.exists():
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                existing_payload = json.load(handle)
        except json.JSONDecodeError:
            existing_payload = {}

    if not isinstance(existing_payload, dict):
        existing_payload = {}

    merged_payload = {**existing_payload, **payload}
    if existing_payload.get("notes") and payload.get("notes"):
        merged_payload["notes"] = f"{existing_payload.get('notes')}\n{payload.get('notes')}"

    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_payload, handle, indent=2)

    return str(file_path)


def build_aesthetic_summary(payload):
    if not isinstance(payload, dict):
        return None

    lines = [APPOINTMENT_SUMMARY_HEADER, ""]
    for field_key, field_label in APPOINTMENT_SUMMARY_FIELDS:
        lines.append(f"{field_label}: {payload.get(field_key, NOT_PROVIDED_TEXT)}")
    return "\n".join(lines)


def get_response(prompt, client):
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": GENERIC_ASSISTANT_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

def get_chatbot_response(message_list, prompt, client):
    if not message_list:
        message_list.append(
            {
                "role": "system",
                "content": INTAKE_SYSTEM_PROMPT,
            }
        )

    input_moderation = moderate_text(prompt, stage="input")
    if input_moderation.action in {"block", "escalate"}:
        safe_reply = input_moderation.response or BLOCKED_RESPONSE
        message_list.append({"content": prompt, "role": "user"})
        message_list.append({"content": safe_reply, "role": "assistant"})
        return safe_reply

    menu_response = handle_menu_request(prompt, client)
    if menu_response:
        return menu_response

    if is_blocked_text(prompt, INPUT_BLOCK_LIST) or contains_prompt_injection_attempt(prompt):
        return BLOCKED_RESPONSE

    message_list.append({"content": prompt, "role": "user"})
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=message_list,
    )
    response_text = response.choices[0].message.content

    output_moderation = moderate_text(response_text, stage="output")
    if output_moderation.action in {"block", "sanitize"}:
        safe_reply = output_moderation.response or BLOCKED_RESPONSE
        message_list.append({"content": safe_reply, "role": "assistant"})
        return safe_reply

    message_list.append({"content": response_text, "role": response.choices[0].message.role})

    if is_blocked_text(response_text, OUTPUT_BLOCK_LIST) or contains_prompt_injection_attempt(response_text):
        return BLOCKED_RESPONSE
    payload = extract_json_payload(response_text)
    if payload is not None:
        save_summary_to_db(payload)
        return build_aesthetic_summary(payload)

    save_summary_to_db(response_text)
    return response_text

def main():
    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        print(exc)
        return 1

    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()
    message_list = []
    print(get_chatbot_response(message_list, BOOTSTRAP_PROMPT, client))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
