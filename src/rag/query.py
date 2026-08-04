"""Query analysis: what a question is about, and how many questions it is.

Both answers are used before generation and neither is allowed to be a model's
opinion on its own. Category inference feeds a veto (A.4.2) and sub-question
splitting feeds partial answering (A.4.1); a wrong answer from either would
either suppress a good answer or fragment one, so both are deterministic and
both fail open — when the analysis cannot tell, it says so and the caller
proceeds as if it had not run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Vocabulary that places a question in one of the corpus categories. Derived
# from the documents themselves rather than invented: these are the words the
# sources use for the things they describe.
#
# A question can plausibly belong to more than one category — a colonoscopy is
# both an endoscopy and a colorectal screening, and the corpus files it under
# both. Inference therefore returns a set, and the guard that uses it only
# objects when retrieval agrees with none of them. A narrower reading would
# reject correct answers for being filed under the neighbouring heading.
CATEGORY_VOCABULARY: dict[str, tuple[str, ...]] = {
    "imaging": (
        r"\bmri\b", r"\bct\b", r"\bcat scan\b", r"\bscan\b", r"\bscanner\b",
        r"\bcontrast\b", r"\bradiation\b", r"\bx-?ray\b", r"\bimaging\b",
        r"\bmagnet", r"\bgadolinium\b", r"\bpet\b", r"\bmammogram\b",
        r"\bultrasound\b", r"\bradiologist\b",
    ),
    "endoscopy": (
        r"\bcolonoscop", r"\bendoscop", r"\bsigmoidoscop", r"\bscope\b",
        r"\bbowel prep\b", r"\begd\b", r"\bgastroscop", r"\bbronchoscop",
        r"\bcystoscop", r"\blaparoscop", r"\bpolyp",
    ),
    "lab_test": (
        r"\bblood (test|draw|sample|panel)\b", r"\blab (test|work|results?)\b",
        r"\burine\b", r"\burinalysis\b", r"\bfasting\b", r"\bige\b",
        r"\ballergy (blood )?test\b", r"\brapid test\b", r"\bswab\b",
        r"\bcholesterol\b", r"\bglucose\b", r"\btriglyceride", r"\bspecimen\b",
    ),
    "screening": (
        r"\bscreening\b", r"\bcolorectal\b", r"\bskin cancer\b", r"\bmole\b",
        r"\bmelanoma\b", r"\bstool (test|sample)\b", r"\bfobt\b", r"\bfit\b",
        r"\bocc?ult blood\b", r"\babcde\b",
    ),
    "exam": (
        r"\bhearing (test|loss)\b", r"\baudiometry\b", r"\baudiologist\b",
        r"\btympanometry\b", r"\bneurolog", r"\breflex", r"\btuning fork\b",
        r"\bcranial nerve", r"\bphysical exam\b",
    ),
}

_COMPILED = {
    category: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for category, patterns in CATEGORY_VOCABULARY.items()
}


def infer_categories(question: str) -> frozenset[str]:
    """Return the corpus categories a question plausibly belongs to.

    An empty set means "cannot tell", which is a normal outcome — a question
    like "how long will it take?" names no subject at all. Callers must treat
    that as no information rather than as a mismatch.
    """

    return frozenset(
        category
        for category, patterns in _COMPILED.items()
        if any(pattern.search(question) for pattern in patterns)
    )


# ---------------------------------------------------------------------------
# Compound questions
# ---------------------------------------------------------------------------

# Where one question becomes two. Splitting on "and" alone would cut "bring your
# ID and insurance card" in half, so a split point must join two clauses that
# each look like a question.
_SPLIT = re.compile(
    r"\s*(?:,\s*)?\band\s+(?=(?:also\s+)?"
    r"(?:can|could|should|do|does|will|would|is|are|am|what|when|where|how|why|who|"
    r"i\s|my\s|the\s))",
    re.IGNORECASE,
)

# A clause has to look like a question to be worth answering separately.
_INTERROGATIVE = re.compile(
    r"\b(can|could|should|do|does|did|will|would|is|are|am|what|when|where|how|"
    r"why|who|which)\b",
    re.IGNORECASE,
)

MAX_RECONSTRUCTION_LOSS = 0.15


@dataclass(frozen=True)
class QuestionSplit:
    """The parts of a compound question, and whether the split is trustworthy."""

    parts: tuple[str, ...]
    split: bool
    reason: str

    @property
    def is_compound(self) -> bool:
        return self.split and len(self.parts) > 1


def split_question(question: str, max_parts: int = 3) -> QuestionSplit:
    """Split a compound question, or decline to.

    The guard matters more than the split. A split that drops or distorts part
    of the question would produce an answer that looks complete while silently
    omitting what the patient asked, so the parts are checked against the
    original: if reconstructing them loses more than a small fraction of the
    content words, the split is discarded and the turn is treated as one
    question. Answering the whole thing imperfectly beats answering a piece of
    it confidently.
    """

    text = question.strip()
    if not text:
        return QuestionSplit(parts=(text,), split=False, reason="empty question")

    # Sentence-final question marks first, then conjunctions inside a sentence.
    sentences = [part.strip() for part in re.split(r"(?<=\?)\s+", text) if part.strip()]
    parts: list[str] = []
    for sentence in sentences:
        parts.extend(
            piece.strip() for piece in _SPLIT.split(sentence) if piece.strip()
        )

    parts = [part for part in parts if _INTERROGATIVE.search(part)]

    if len(parts) < 2:
        return QuestionSplit(parts=(text,), split=False, reason="not compound")
    if len(parts) > max_parts:
        return QuestionSplit(
            parts=(text,), split=False, reason=f"more than {max_parts} parts"
        )

    lost = _content_words(text) - _content_words(" ".join(parts))
    if len(lost) / max(len(_content_words(text)), 1) > MAX_RECONSTRUCTION_LOSS:
        return QuestionSplit(
            parts=(text,),
            split=False,
            reason=f"split lost content: {sorted(lost)[:4]}",
        )

    return QuestionSplit(parts=tuple(parts), split=True, reason=f"{len(parts)} parts")


# Words a split consumes by definition. Counting them as lost content made the
# guard reject its own correct splits: "is a hearing test painful, and where do
# I park?" loses exactly one word — "and", the conjunction it split on — and one
# word out of six is enough to trip the 15% threshold, collapsing a two-part
# question back into one and answering neither half well.
_STRUCTURAL = frozenset({"and", "also", "but", "plus"})


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9']+", text.lower())
        if len(word) > 2 and word not in _STRUCTURAL
    }


# ---------------------------------------------------------------------------
# Is this a question at all?
# ---------------------------------------------------------------------------

_ASKS = re.compile(
    r"(\?\s*$)|^\s*(what|when|where|how|why|who|which|can|could|should|do|does|"
    r"did|is|are|will|would|must|tell me|explain)\b",
    re.IGNORECASE,
)


def is_information_request(message: str) -> bool:
    """Report whether a message asks for information rather than supplying it.

    The knowledge branch must stand down on everything else. "I have a headache
    that started yesterday" is content for the intake workflow, and answering it
    with "I don't have documentation covering that" would replace a turn of
    progress with an apology for a question the patient never asked.

    Deliberately narrow: a question mark, or an interrogative opener. A
    statement that happens to contain "what" mid-sentence is not a request.
    """

    text = message.strip()
    if not text:
        return False
    return bool(_ASKS.search(text))
