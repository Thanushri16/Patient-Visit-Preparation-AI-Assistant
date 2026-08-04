"""What may be answered by retrieval, and what must never be.

This is the precedence ladder from section 3.6 of the plan. It runs BEFORE any
retrieval, and its most important job is to say no.

The failure it exists to prevent is specific. Some questions are not requests
for information at all; they are requests for clinical judgement, and the
assistant's answer to them is a refusal. "Should I stop my blood thinner before
my colonoscopy?" is one. The danger is that the corpus contains text that reads
like an answer — preparation documents discussing which medicines may need
stopping — so retrieval would find something, the model would ground a fluent
answer in it, the citation would resolve, and every downstream check would pass.
Faithfulness included: the answer really would be faithful to the retrieved
passage. Faithful to a document is not the same as safe, and nothing further
down the pipeline measures the difference.

So the guard runs on the question, before the retriever is called at all. A
decision reached after retrieval is a decision made with the wrong answer
already in hand.

Detectors are imported from src/guidance.py rather than restated here. Copying
those regexes would create a second source of truth that drifts from the first,
which is exactly the defect the manifest's cleaning rules turned out to be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

try:
    from ..guidance import (
        ANAPHYLAXIS_NOTE,
        detect_anaphylaxis_risk,
        detect_educational_topic,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from guidance import (
        ANAPHYLAXIS_NOTE,
        detect_anaphylaxis_risk,
        detect_educational_topic,
    )


class RouteOutcome(StrEnum):
    """What should happen to this message. Mirrors the benchmark's outcomes."""

    RETRIEVE = "retrieve"
    CURATED_REFUSAL = "curated_refusal"
    ANAPHYLAXIS_NOTE = "anaphylaxis_note"
    CURATED_ANSWER = "curated_answer"


@dataclass(frozen=True)
class RouteDecision:
    """Whether retrieval may run, and what to say if it may not."""

    outcome: RouteOutcome
    response: str | None = None
    topic: str | None = None
    reason: str = ""

    @property
    def retrieval_allowed(self) -> bool:
        return self.outcome is RouteOutcome.RETRIEVE


# ---------------------------------------------------------------------------
# Never route: refusals to exercise clinical judgement
# ---------------------------------------------------------------------------

# Topics in guidance.EDUCATIONAL_TOPICS that are refusals rather than answers.
# Retrieval must not be able to turn a refusal into an answer.
NEVER_ROUTE_TOPICS = frozenset(
    {
        "interaction",              # food/drug interactions -> pharmacist
        "allergy_vs_side_effect",   # classifying a reaction -> clinician
        "specialist_referral",      # whether a specialist is needed -> clinician
    }
)

# Asking whether to start, stop, hold or change a medication.
#
# The existing topic table has no entry for this, because it was written when
# the assistant answered clinic logistics — what to bring, where to park. This
# corpus is procedure preparation, and that changes which questions patients
# will ask. Every preparation document in it discusses medicines: the lab-prep
# page lists medicines that alter results, the colonoscopy and colorectal pages
# both say "you may need to stop taking some of your medicines", the allergy
# page names antihistamines. A patient reading about a procedure asks the
# obvious next question — "so should I stop mine?" — and the assistant now holds
# documents that appear to answer it.
#
# They do not. Every one of those passages defers the decision to the
# prescriber, and the deferral is the substance of the guidance, not a caveat
# attached to it. An answer that relayed the restriction without the deferral
# would invert the source's meaning while remaining faithful to its words.
# Refusing before retrieval is the only place that distinction can be enforced.
MEDICATION_CHANGE_PATTERNS = (
    r"\b(should|can|do|must)\s+i\s+(stop|skip|hold|pause|continue|keep taking|take)\b",
    r"\b(stop|skip|hold|pause)\s+(taking\s+)?(my|the|his|her|their)\b",
    r"\bdo i (still )?(need to )?take\b",
    r"\b(before|prior to)\b.{0,40}\b(should i|can i)\b.{0,20}\b(take|stop|skip)\b",
)

MEDICATION_CHANGE_REFUSAL = (
    "Whether to stop, hold or keep taking a medication before a test is your "
    "prescriber's or pharmacist's decision, and I can't advise you to change how "
    "you take anything — stopping some medicines suddenly carries its own risk. "
    "The clinic that ordered the test can tell you, and your pharmacist can "
    "usually answer the same day. I'll note the question so you can raise it."
)

_MEDICATION_CHANGE = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in MEDICATION_CHANGE_PATTERNS
)

# A medication-change phrasing only counts when a medicine is actually in view.
# Without this, "should I stop eating before the test" — a legitimate fasting
# question the corpus answers well — would be refused.
MEDICATION_SUBJECT = re.compile(
    r"\b(medication|medicine|medicines|meds|drug|drugs|pill|pills|tablet|tablets|"
    r"dose|doses|prescription|blood thinner|anticoagulant|warfarin|aspirin|"
    r"ibuprofen|naproxen|antihistamine|insulin|metformin|statin|supplement|"
    r"vitamin|inhaler|injection)\b",
    re.IGNORECASE,
)


def asks_to_change_medication(message: str) -> bool:
    """Report whether the message asks whether to alter a medication."""

    if not MEDICATION_SUBJECT.search(message):
        return False
    return any(pattern.search(message) for pattern in _MEDICATION_CHANGE)


# Asking the assistant to say what is wrong with you.
#
# "The assistant does not diagnose" is an invariant this project already holds
# (src/moderation.py enforces it on output). What changes with a corpus of
# diagnostic-test documents is that the assistant now holds the criteria. The
# skin-cancer page states the ABCDE rule for moles; the colorectal page lists
# what abnormal results can mean. A patient who describes their own mole in
# ABCDE terms and asks "do I have melanoma?" would retrieve exactly the passage
# that appears to answer it.
#
# Explaining what the rule is remains in scope and is well covered. Applying it
# to the patient's own body is diagnosis, and the difference is not something a
# grounded-answer prompt can be trusted to hold: the model would be reasoning
# from a real source, on topic, with a resolvable citation.
DIAGNOSIS_REQUEST_PATTERNS = (
    r"\bdo i have\b(?!\s+to\b)",
    r"\b(is|could)\s+(this|it|that)\s+(be\s+)?(a\s+|an\s+)?",
    r"\bdoes (this|that|it) mean i (have|might have)\b",
    r"\bam i (having|getting)\b",
    r"\bwhat'?s wrong with me\b",
    r"\bshould i be worried (about|that)\b",
)

# The phrasings above are common in harmless questions too ("is this the right
# form?"), so a diagnosis reading also requires a condition or a body finding to
# be in view.
DIAGNOSIS_SUBJECT = re.compile(
    r"\b(cancer|melanoma|carcinoma|tumou?r|infection|diabetes|disease|"
    r"condition|serious|malignant|benign|mole|lump|lesion|growth|rash|"
    r"bleeding|symptom|symptoms)\b",
    re.IGNORECASE,
)

DIAGNOSIS_REFUSAL = (
    "I can't tell you what a symptom or a mark on your body means — that needs "
    "someone who can examine you, and getting it wrong in either direction "
    "causes harm. Please have it looked at rather than waiting; if it is "
    "changing, bleeding, or new, say so when you book. I'll note what you've "
    "described so it is on the record for your visit."
)

_DIAGNOSIS_REQUEST = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in DIAGNOSIS_REQUEST_PATTERNS
)


def asks_for_a_diagnosis(message: str) -> bool:
    """Report whether the message asks what is wrong with the patient."""

    if not DIAGNOSIS_SUBJECT.search(message):
        return False
    return any(pattern.search(message) for pattern in _DIAGNOSIS_REQUEST)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def evaluate(message: str) -> RouteDecision:
    """Decide what may answer this message, in the order of section 3.6.

    Steps 1 and 2 of the published ladder — input moderation and state recall —
    run in the chat layer before this is reached, because both concern the whole
    turn rather than the knowledge branch. What is here is steps 3 to 5: the
    never-route set, then retrieval, with any curated answer carried alongside as
    the fallback.
    """

    # 3a. A described anaphylactic reaction outranks everything else here. It is
    # not a question to answer, it is a safety notice to deliver.
    if detect_anaphylaxis_risk(message):
        return RouteDecision(
            outcome=RouteOutcome.ANAPHYLAXIS_NOTE,
            response=ANAPHYLAXIS_NOTE,
            reason="described reaction reads as anaphylaxis",
        )

    topic = detect_educational_topic(message)

    # 3b. Curated topics that are refusals. Matched on the topic NAME, not on
    # whether retrieval later finds anything: the point is that it never runs.
    #
    # These are checked before the broader rules below because their wording is
    # specific to the question asked — the allergy-versus-side-effect answer
    # explains the distinction and says the reaction has been recorded, where a
    # general refusal could only decline. Specific before general, so the better
    # answer wins whenever both apply.
    if topic is not None and topic[0] in NEVER_ROUTE_TOPICS:
        name, answer = topic
        return RouteDecision(
            outcome=RouteOutcome.CURATED_REFUSAL,
            response=answer,
            topic=name,
            reason=f"{name} is a refusal to give clinical judgement",
        )

    # 3c. Asking what is wrong with them. The topic table has no entry for this,
    # and retrieval holds the criteria that would appear to answer it.
    if asks_for_a_diagnosis(message):
        return RouteDecision(
            outcome=RouteOutcome.CURATED_REFUSAL,
            response=DIAGNOSIS_REFUSAL,
            topic="diagnosis_request",
            reason="asks the assistant to interpret the patient's own finding",
        )

    # 3d. Asking whether to alter a medication. No entry in the table either.
    if asks_to_change_medication(message):
        return RouteDecision(
            outcome=RouteOutcome.CURATED_REFUSAL,
            response=MEDICATION_CHANGE_REFUSAL,
            topic="medication_change",
            reason="asks whether to alter a medication",
        )

    # 4/5. A curated answer exists but is not a refusal, so retrieval may run
    # and this is the fallback if it finds nothing.
    if topic is not None:
        name, answer = topic
        return RouteDecision(
            outcome=RouteOutcome.RETRIEVE,
            response=answer,
            topic=name,
            reason=f"curated answer for {name} available as fallback",
        )

    return RouteDecision(outcome=RouteOutcome.RETRIEVE, reason="no curated topic")
