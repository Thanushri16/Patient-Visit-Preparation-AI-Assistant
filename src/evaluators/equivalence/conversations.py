"""The conversations Part B's equivalence run replays, and how strictly.

Each conversation declares a mode. STRICT means the path contains no model
call, so a perfect reimplementation must reproduce it byte for byte. STRUCTURAL
means a model decides something on the path, so only the decisions it drives are
compared.

The declaration is a claim about the code, and `calibrate()` checks it: every
STRICT conversation is replayed chain-against-chain, where any divergence means
the declaration is wrong rather than the migration. Without that check this file
would be a place to quietly downgrade anything inconvenient.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .harness import Mode
except ImportError:  # pragma: no cover
    from harness import Mode


@dataclass(frozen=True)
class ConversationSpec:
    messages: tuple[str, ...]
    mode: Mode
    why: str


CONVERSATIONS: dict[str, ConversationSpec] = {
    # ---- STRICT: no model call decides anything on these paths -------------
    "emergency": ConversationSpec(
        ("I have crushing chest pain and my left arm is numb",),
        Mode.STRICT,
        "moderation is regex; escalation text is a constant",
    ),
    "menu": ConversationSpec(
        ("menu", "2"),
        Mode.STRICT,
        "menu options route without the intent classifier",
    ),
    "never_route_medication": ConversationSpec(
        ("Should I stop my blood thinner before my colonoscopy?",),
        Mode.STRICT,
        "the policy ladder refuses before retrieval; refusal is a constant",
    ),
    "never_route_diagnosis": ConversationSpec(
        ("This mole has a ragged border. Do I have melanoma?",),
        Mode.STRICT,
        "same ladder, diagnosis branch",
    ),
    "anaphylaxis": ConversationSpec(
        ("My throat swelled up and I needed an EpiPen last month",),
        Mode.STRICT,
        "detector is regex; the safety note is a constant",
    ),
    "off_topic": ConversationSpec(
        ("What's the weather like today?",),
        Mode.STRICT,
        "pre-check declines before any model call",
    ),
    "state_recall": ConversationSpec(
        ("menu", "What medications have I told you about?"),
        Mode.STRICT,
        "answered from VisitData, never from a model",
    ),
    # ---- STRUCTURAL: a model decides something -----------------------------
    "intake": ConversationSpec(
        ("2", "I have a headache that started yesterday", "it's a 6 out of 10"),
        Mode.STRUCTURAL,
        "extraction and follow-up wording are model-backed",
    ),
    "knowledge": ConversationSpec(
        ("What is the bowel prep for a colonoscopy?",),
        Mode.STRUCTURAL,
        "grounded generation writes the sentence",
    ),
    "knowledge_fallback": ConversationSpec(
        ("What does my A1C result mean?",),
        Mode.STRUCTURAL,
        "the fallback text is constant but extraction still runs on the turn",
    ),
    "compound": ConversationSpec(
        ("Is a hearing test painful, and where do I park?",),
        Mode.STRUCTURAL,
        "partial answering composes generated and constant segments",
    ),
    "summary": ConversationSpec(
        ("1", "My name is Dana", "show me my summary"),
        Mode.STRUCTURAL,
        "extraction populates the record the summary renders",
    ),
}


def calibrate(run, repeats: int = 2) -> dict[str, list[str]]:
    """Replay every STRICT conversation against itself and report any that move.

    A conversation that diverges here is misdeclared: the path it takes reaches
    a model somewhere. Fix the declaration or the path — do not widen the
    tolerance, which is the failure mode this function exists to prevent.
    """

    try:
        from .harness import compare_conversation
    except ImportError:  # pragma: no cover
        from harness import compare_conversation

    unstable: dict[str, list[str]] = {}
    for name, spec in CONVERSATIONS.items():
        if spec.mode is not Mode.STRICT:
            continue
        for attempt in range(repeats):
            result = compare_conversation(
                f"{name}-cal{attempt}", spec.messages, run, run, Mode.STRICT
            )
            if not result.equivalent:
                unstable.setdefault(name, []).extend(
                    d.describe() for d in result.divergences
                )
    return unstable
