"""Conversational layers that sit alongside structured intake.

Intake alone answers only half of what patients actually say. A single message
often carries a question ("do I need to fast?"), a feeling ("I'm really nervous
about this"), a greeting, or a request to read back what the assistant already
holds. Each is handled here and composed into the same reply as the extraction
result, so the bot responds to the whole message instead of only the parts that
map onto fields.

All educational content is deliberately general and non-prescriptive: it
describes what is typical and routes specifics back to the patient's clinician.
"""

import re

try:
    from .models import ConversationState, VisitData
except ImportError:  # pragma: no cover - allows running as a script
    from models import ConversationState, VisitData


# (name, trigger patterns, answer). Order matters: the first match wins.
EDUCATIONAL_TOPICS = (
    (
        "documents",
        (r"what (documents|papers|paperwork) .{0,20}bring", r"what should i bring", r"\bbring .{0,20}(appointment|visit)\b"),
        "Generally, visits go smoothly when you bring a photo ID, your insurance "
        "card, a list of your current medications with doses, any referral letter "
        "or recent test results, and your list of questions. Some clinics ask for "
        "more, so it is worth checking the reminder they sent you.",
    ),
    (
        "forms",
        (r"\bforms?\b.{0,30}(fill|ahead|advance|before)", r"\bpaperwork\b.{0,20}(ahead|advance|before)"),
        "Many practices post new-patient and health-history forms on their portal "
        "so you can complete them beforehand, which usually shortens the wait. If "
        "you haven't received a link, the front desk can tell you what they need.",
    ),
    (
        "fasting",
        (r"\bfast(ing)?\b", r"\b(eat|drink|food)\b.{0,25}\bbefore\b.{0,25}(test|blood|lab|appointment)"),
        "Fasting requirements depend on the specific test — some blood panels ask "
        "for 8 to 12 hours with water only, while many tests need no fasting at "
        "all. Because it changes your results, please follow the instructions the "
        "ordering clinic gave you, and call them if you did not get any.",
    ),
    (
        "telehealth",
        (r"\b(telehealth|virtual|video)\b", r"in person or virtual"),
        "Appointments can be in person or by video, and the clinic's confirmation "
        "message normally says which. For a video visit it helps to test your "
        "camera and microphone beforehand and to have a quiet, well-lit spot.",
    ),
    (
        "transportation",
        (r"\btransport(ation)?\b", r"\b(ride|get there|getting there)\b.{0,20}\bappointment\b"),
        "Many clinics and insurance plans, including a lot of Medicaid plans, offer "
        "non-emergency medical transportation or can point you to local services. "
        "The practice's front desk or your plan's member services line is the "
        "fastest way to arrange it.",
    ),
    (
        "specialist_referral",
        (r"\b(dermatologist|specialist|cardiologist|neurologist)\b.{0,30}\?", r"do i need to see a\b"),
        "Whether a specialist is needed is a judgement your clinician makes after "
        "examining you, and some plans also require a referral from your primary "
        "care provider first. Describing when it started and how it has changed "
        "gives them what they need to decide.",
    ),
    (
        "latex",
        (r"\blatex\b",),
        "Latex-free supplies are standard practice now, and clinics routinely "
        "switch to nitrile gloves and latex-free equipment for patients with a "
        "latex allergy. Telling the front desk when you check in makes sure it is "
        "flagged before anyone examines you.",
    ),
    (
        "interaction",
        (
            r"\b(interact|interaction|don'?t mix|doesn'?t mix|mix with)\b",
            r"\bgrapefruit\b",
        ),
        "Food and drug interactions are real and worth raising — but whether one "
        "applies to your prescription, and what to do about it, is your "
        "clinician's or pharmacist's call, and I can't tell you to change how "
        "you take anything. Your pharmacist can usually answer this the same "
        "day. I'll note the concern so you can raise it at your visit.",
    ),
    (
        "allergy_vs_side_effect",
        (r"\bis that an allerg", r"\ballergy or (a )?side effect\b"),
        "That distinction matters clinically — a true allergy is an immune "
        "response, while nausea or an upset stomach is more often a side effect "
        "— but telling them apart is your clinician's call. Either way it is "
        "worth having on your record, so I've noted the reaction as you "
        "described it.",
    ),
    (
        "new_patient",
        (r"\bfirst time\b.{0,30}(seeing|visiting)", r"\bnew patient\b"),
        "For a first visit, practices typically ask you to arrive a little early "
        "and to bring your ID, insurance card, medication list, and any records "
        "from previous providers.",
    ),
)

EMOTIONAL_PATTERNS = (
    r"\b(nervous|anxious|anxiety|scared|afraid|worried|frightened|terrified|dreading)\b",
    r"\bstressed( out)?\b",
)

EMOTIONAL_ACKNOWLEDGEMENT = (
    "It's completely understandable to feel that way, and a lot of people do — "
    "being prepared often takes some of the edge off. We can go at your pace, and "
    "you can write down anything you want to raise with your clinician."
)

GREETING_PATTERNS = (
    r"^(hi|hey|hello|good morning|good afternoon|good evening|greetings)\b",
)

FAREWELL_PATTERNS = (
    r"\b(bye|goodbye|see you|take care)\b",
    r"\bthat'?s (everything|all|it)\b",
    r"\bbefore i go\b",
    r"\bthanks,? that'?s (all|it)\b",
    r"\ball i need\b",
)

FAREWELL_RESPONSE = (
    "Thanks for going through this with me, and good luck at your appointment. "
    "Before you go, I can show you a summary of everything recorded so far — "
    "just say summary. Take care."
)

# Requests that are outside a healthcare assistant's remit. Declining these by
# name reads as an answer; the bare option list reads as not having listened.
OFF_TOPIC_PATTERNS = (
    r"\b(weather|forecast|temperature outside)\b",
    r"\b(poem|joke|story|song|recipe)\b",
    r"\b(sports|football|score|stock|stocks|crypto|bitcoin)\b",
    r"\bwho (won|is the president)\b",
    r"\btranslate this\b",
    r"\bwrite (me )?(a|an)\b",
)

OFF_TOPIC_RESPONSE = (
    "That one is outside what I can help with — I'm here specifically to help you "
    "get ready for a healthcare appointment. I can capture your symptoms, "
    "medications, allergies, and visit details, and turn them into a summary to "
    "bring with you. Would you like to start there?"
)

# Messages that name no topic at all and cannot be routed on their own.
AMBIGUOUS_PATTERNS = (
    r"^i need (some )?(information|help|assistance)\.?$",
    r"^(help me|i need something)\.?$",
    r"^i have a question\.?$",
    r"^(medicine|medication|doctor|appointment|symptom|health)\.?$",
    r"^my (arm|leg|back|head|stomach|chest|neck|shoulder|knee|foot|hand)\.?$",
)

AMBIGUOUS_RESPONSE = (
    "I want to make sure I point you the right way — could you tell me a little "
    "more about what you need? For example, are you preparing for an upcoming "
    "appointment, reporting a symptom, asking about a medication, or recording an "
    "allergy?"
)

# Questions asking the assistant to read state back, mapped to the fields to report.
STATE_QUERY_TOPICS = (
    ("current_medications", (r"\b(medications?|meds|drugs)\b.{0,30}\b(do i have|have i|listed|so far|mentioned|taking)\b", r"what (medications?|meds) do i")),
    ("allergies", (r"\ballerg\w*\b.{0,30}\b(do i have|have i|listed|so far|mentioned)\b",)),
    ("chief_complaint", (r"\bsymptoms?\b.{0,30}\b(have i|did i|listed|so far|mentioned)\b",)),
    ("visit_reason", (r"\b(reason|why)\b.{0,30}\b(visit|appointment)\b.{0,20}\b(was|is|said|again)\b", r"remind me .{0,30}reason")),
    ("provider_name", (r"\b(who|which doctor)\b.{0,30}\b(am i|i'?m)\b.{0,15}seeing",)),
    ("appointment_date", (r"\bwhen\b.{0,20}\bmy appointment\b",)),
)

# A state query has to actually be a question. "I'm not sure which doctor I'm
# seeing" mentions a field but is an answer, and treating it as a question meant
# replying "I don't have a provider recorded" instead of recording that.
STATE_QUERY_TRIGGER = re.compile(
    r"(\?\s*$)|^(what|which|who|when|do i|have i|can you (show|tell)|show me|"
    r"remind me|list )",
    flags=re.IGNORECASE,
)

# "I'm not sure which doctor I'm seeing" names a field but supplies an answer.
# The trigger above already rejects it for not being in question form; this
# catches the same shape when it does end in a question mark.
STATE_QUERY_DISQUALIFIERS = re.compile(
    r"\b(not sure|unsure|don'?t know|do not know|no idea)\b",
    flags=re.IGNORECASE,
)

NON_ASCII_WORD = re.compile(r"[À-ɏ]")
COMMON_NON_ENGLISH_WORDS = {
    "tengo", "dolor", "cabeza", "estoy", "necesito", "gracias", "por", "favor",
    "je", "suis", "bonjour", "merci", "ich", "habe", "bitte", "danke",
    "ho", "sono", "grazie", "eu", "tenho", "obrigado",
}


def _matches(patterns, text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def detect_educational_topic(message: str) -> tuple[str, str] | None:
    """Return the (topic, answer) for a general preparation question, if any."""

    for topic, patterns, answer in EDUCATIONAL_TOPICS:
        if _matches(patterns, message):
            return topic, answer
    return None


def detect_emotional_content(message: str) -> bool:
    return _matches(EMOTIONAL_PATTERNS, message)


def detect_greeting(message: str) -> bool:
    return _matches(GREETING_PATTERNS, message.strip())


def detect_farewell(message: str) -> bool:
    return _matches(FAREWELL_PATTERNS, message)


def detect_off_topic(message: str) -> bool:
    return _matches(OFF_TOPIC_PATTERNS, message)


def detect_ambiguous(message: str) -> bool:
    """Flag a message that names no topic the assistant could route on."""

    normalized = re.sub(r"\s+", " ", message).strip().lower()
    return _matches(AMBIGUOUS_PATTERNS, normalized)


def looks_non_english(message: str) -> bool:
    """Flag likely non-English input from accented characters or common words."""

    words = {word.strip(".,!?¿¡").lower() for word in message.split()}
    return bool(NON_ASCII_WORD.search(message)) or len(words & COMMON_NON_ENGLISH_WORDS) >= 2


def is_low_information(message: str) -> bool:
    """Detect gibberish or punctuation-only input that carries no request."""

    stripped = message.strip()
    if not stripped:
        return True
    letters = re.sub(r"[^a-z]", "", stripped.lower())
    if not letters:
        return True
    if len(stripped) < 40 and not re.search(r"[aeiou]", letters):
        return True
    # Long runs of consonant-only "words" are keyboard mash rather than language.
    tokens = [token for token in re.findall(r"[a-z]+", stripped.lower()) if len(token) > 3]
    if tokens and all(len(re.sub(r"[^aeiou]", "", token)) / len(token) < 0.2 for token in tokens):
        return True
    return False


def _describe(field_name: str, value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return "none recorded"
        parts = []
        for item in value:
            if hasattr(item, "name"):
                parts.append(
                    " ".join(filter(None, [item.name, item.dosage or "", item.frequency or ""])).strip()
                )
            elif hasattr(item, "allergen"):
                reaction = f" (reaction: {item.reaction})" if item.reaction else ""
                parts.append(f"{item.allergen}{reaction}")
            else:
                parts.append(str(item))
        return ", ".join(parts)
    return str(value)


def answer_state_query(message: str, visit_data: VisitData) -> str | None:
    """Answer a "what have I told you" question from recorded state only.

    Returns ``None`` when the message is not such a question, so the caller can
    fall through to normal intake.
    """

    normalized = message.strip()
    if not STATE_QUERY_TRIGGER.search(normalized):
        return None
    if STATE_QUERY_DISQUALIFIERS.search(normalized):
        return None
    for field_name, patterns in STATE_QUERY_TOPICS:
        if not _matches(patterns, message):
            continue
        described = _describe(field_name, getattr(visit_data, field_name, None))
        readable = field_name.replace("_", " ")
        if described is None:
            return f"I don't have any {readable} recorded yet for this visit."
        return f"Here is the {readable} I have recorded: {described}."
    return None


def build_supplementary_response(state: ConversationState, message: str) -> str:
    """Build the non-intake part of a reply: greeting, empathy, and guidance.

    The pieces are returned as one string so the caller can place them ahead of
    the extraction acknowledgement and the next intake question.
    """

    segments: list[str] = []
    if detect_greeting(message):
        segments.append("Hello, and thanks for reaching out.")
    if detect_emotional_content(message):
        segments.append(EMOTIONAL_ACKNOWLEDGEMENT)
    topic = detect_educational_topic(message)
    if topic is not None:
        segments.append(topic[1])
    return " ".join(segments)
