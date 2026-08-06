"""Load the corpus manifest and turn each PDF into clean, sectioned text.

This is the "Cleaning & Metadata Extraction" stage the plan places between the
LlamaIndex document loader and chunking. It ends by emitting LlamaIndex
`Document` objects — one per section — which the node parser then splits.

One `Document` per section rather than per PDF is what keeps a chunk from
spanning two sections. These MedlinePlus pages are organised as literal patient
questions, so a section boundary is a topic boundary, and a node straddling two
of them mixes the answers to different questions. A node parser cannot know
that; the document boundaries it is given are the only way to tell it.

The PDFs are the source of truth and are never converted on disk. This module
extracts their text at ingest time, strips the boilerplate that each MedlinePlus
page shape carries, and splits what remains into the sections the manifest
declares.

Sections are matched against the manifest rather than guessed. That makes
section detection auditable, and it means a re-fetched PDF whose headings have
changed fails loudly at ingest instead of silently producing chunks with the
wrong citation metadata.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pypdf import PdfReader

try:
    from .config import CORPUS_DIR, MANIFEST_PATH, PIPELINE_VERSION
except ImportError:  # pragma: no cover - allows running as a script
    from config import CORPUS_DIR, MANIFEST_PATH, PIPELINE_VERSION

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core.schema import Document


@dataclass(frozen=True)
class ManifestDocument:
    """One indexed entry from clinical_docs/manifest.yaml."""

    document_id: str
    file: str
    title: str
    category: str
    page_shape: str
    source_url: str | None
    last_updated: date | None
    page_count: int
    sections: tuple[str, ...]

    @property
    def path(self) -> Path:
        return CORPUS_DIR / self.file


@dataclass
class Section:
    """A named span of document text, with the page it starts on."""

    heading: str | None
    page_number: int
    text: str


@dataclass
class LoadedDocument:
    """A cleaned, sectioned document ready for chunking."""

    manifest: ManifestDocument
    sections: list[Section]
    content_hash: str
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(section.text for section in self.sections)

    @property
    def fingerprint(self) -> str:
        """Identify the stored form of this document.

        The PDF's hash plus the pipeline version: cleaning and chunking are
        code, so a change to either must force a re-ingest even when the source
        file is untouched.
        """

        return f"{self.content_hash}:v{PIPELINE_VERSION}"


class ManifestError(RuntimeError):
    """The manifest and the files on disk disagree."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_manifest(path: Path | None = None) -> tuple[list[ManifestDocument], dict]:
    """Return the indexed documents and the raw manifest.

    A PDF present on disk but absent from the manifest is an error, not a file
    to index quietly: the manifest is where a document's category, citation
    metadata and safety notes live, and indexing without them would put
    uncurated content behind a citation.
    """

    manifest_path = path or MANIFEST_PATH
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    shapes = raw.get("page_shapes", {})

    documents: list[ManifestDocument] = []
    declared_files: set[str] = set()
    for entry in raw.get("documents", []):
        declared_files.add(entry["file"])
        if not entry.get("indexed"):
            continue
        shape = entry["page_shape"]
        if shape not in shapes:
            raise ManifestError(
                f"{entry['document_id']}: page_shape {shape!r} is not declared "
                "in the manifest's page_shapes block"
            )
        # The manifest names a shape; this module owns what that name does. The
        # check keeps the two from drifting apart, which is the failure the
        # deleted rules block in the manifest used to invite: it looked like
        # configuration, was never read, and could disagree with the code
        # silently.
        if shape not in CONTENT_END_MARKERS:
            raise ManifestError(
                f"{entry['document_id']}: page_shape {shape!r} has no cleaning "
                "rules in CONTENT_END_MARKERS (src/rag/documents.py)"
            )
        documents.append(
            ManifestDocument(
                document_id=entry["document_id"],
                file=entry["file"],
                title=entry["title"],
                category=entry["category"],
                page_shape=shape,
                source_url=entry.get("source_url"),
                last_updated=_as_date(entry.get("last_updated")),
                page_count=int(entry.get("pages", 0)),
                sections=tuple(entry.get("sections", ())),
            )
        )

    on_disk = {p.name for p in manifest_path.parent.glob("*.pdf")}
    undeclared = on_disk - declared_files
    if undeclared:
        raise ManifestError(
            "PDFs on disk with no manifest entry: " + ", ".join(sorted(undeclared))
        )
    missing = {d.file for d in documents} - on_disk
    if missing:
        raise ManifestError(
            "Manifest entries with no PDF on disk: " + ", ".join(sorted(missing))
        )
    return documents, raw


def _as_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        return date.fromisoformat(value.strip())
    return None


# ---------------------------------------------------------------------------
# Extraction and cleaning
# ---------------------------------------------------------------------------

# Boilerplate every MedlinePlus print view carries, regardless of page shape.
BOILERPLATE_PATTERNS = (
    re.compile(r"^\d+/\d+/\d+,\s.*$", re.MULTILINE),          # print timestamp
    re.compile(r"^https?://\S*\s+\d+/\d+$", re.MULTILINE),     # page-n-of-m footer
    re.compile(r"^https?://\S+\?utm_source=\S*\s+\d+/\d+$", re.MULTILINE),
    re.compile(r"^An off?icial website.*$", re.MULTILINE),
    re.compile(r"^National Institutes of Health\s*/.*$", re.MULTILINE),
    re.compile(r"^Home\s*→.*$", re.MULTILINE),
    re.compile(r"^URL of this page:.*$", re.MULTILINE),
)

# Inline "[https://...]" link brackets left behind by the print stylesheet.
INLINE_LINK = re.compile(r"\s*\[https?://[^\]]*\]")

# Where the content of each page shape ends. Everything after this marker is
# apparatus: references, related-topic link farms, and site chrome.
#
# The health_topic cut matters most. On the CT page the link farm after
# "Start Here" is roughly 60% of the extracted text, and indexing it would fill
# the store with nodes that are lists of article titles — high lexical overlap
# with any test question and no answer inside them.
CONTENT_END_MARKERS = {
    "medical_test": (r"^References\s*$",),
    "encyclopedia": (r"^References\s*$", r"^Review Date\s",),
    "health_topic": (r"^Start Here\s*$", r"^References\s*$"),
}


def normalise_typography(text: str) -> str:
    """Expand PDF ligatures and normalise quotes and dashes.

    These PDFs render "fl" and "fi" as the single glyphs U+FB02 and U+FB01, so
    the raw extraction contains "ﬂexible" and "speciﬁc". Left alone those words
    are unsearchable, tokenise into nonsense, and never match a manifest
    heading — the colonoscopy page's sections all failed on exactly this.
    NFKC decomposes the ligatures; the rest maps typographic punctuation onto
    the ASCII a patient would actually type.
    """

    text = unicodedata.normalize("NFKC", text)
    for source, target in (
        ("‘", "'"), ("’", "'"),
        ("“", '"'), ("”", '"'),
        ("–", "-"), ("—", "-"),
        (" ", " "),
    ):
        text = text.replace(source, target)
    return text


@lru_cache(maxsize=32)
def _extract_pages_cached(path: str, fingerprint: tuple[int, int]) -> tuple[str, ...]:
    reader = PdfReader(path)
    return tuple((page.extract_text() or "") for page in reader.pages)


def extract_pages(path: Path) -> list[str]:
    """Return the raw text of each page, one string per page.

    Extraction is the slow step, and ingestion reads every document twice — once
    to plan, once to store. The cache is keyed on size and mtime as well as the
    path, so editing a PDF invalidates it.
    """

    stat = path.stat()
    return list(_extract_pages_cached(str(path), (stat.st_size, stat.st_mtime_ns)))


def clean_pages(pages: list[str], page_shape: str) -> list[tuple[int, str]]:
    """Strip boilerplate and cut apparatus, keeping the page number of each line.

    Returns (page_number, line) pairs so that a chunk can still cite the page it
    came from after the text has been reflowed.
    """

    markers = [
        re.compile(pattern, re.MULTILINE)
        for pattern in CONTENT_END_MARKERS.get(page_shape, (r"^References\s*$",))
    ]

    numbered: list[tuple[int, str]] = []
    stop = False
    for page_number, raw in enumerate(pages, start=1):
        if stop:
            break
        text = normalise_typography(raw)
        text = INLINE_LINK.sub("", text)
        for pattern in BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        for marker in markers:
            found = marker.search(text)
            if found:
                text = text[: found.start()]
                stop = True
                break
        for line in text.splitlines():
            entry = line.strip()
            if entry:
                numbered.append((page_number, entry))
    return numbered


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Compare headings ignoring case, punctuation and whitespace runs."""

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def split_sections(
    numbered_lines: list[tuple[int, str]],
    declared_sections: tuple[str, ...],
) -> tuple[list[Section], list[str]]:
    """Split cleaned lines into the sections the manifest declares.

    Any text before the first declared heading becomes a leading section with no
    heading — on these pages that is the title and the opening summary, which is
    worth keeping and worth labelling honestly as unsectioned.

    A heading longer than the PDF's text column is extracted as two or three
    lines, so matching is done over a small lookahead window as well as over
    single lines. The colonoscopy page is the reason: every one of its headings
    names three procedures and every one of them wraps.
    """

    wanted = {_normalise(heading): heading for heading in declared_sections}
    longest = max((len(_normalise(h).split()) for h in declared_sections), default=0)
    max_lookahead = 1 if longest <= 6 else 3

    sections: list[Section] = []
    current_heading: str | None = None
    current_page = numbered_lines[0][0] if numbered_lines else 1
    buffer: list[str] = []
    seen: set[str] = set()

    def flush() -> None:
        if buffer:
            sections.append(
                Section(
                    heading=current_heading,
                    page_number=current_page,
                    text="\n".join(buffer).strip(),
                )
            )

    index = 0
    while index < len(numbered_lines):
        page_number, line = numbered_lines[index]

        # Prefer the longest match, so a heading that is a prefix of a longer
        # one cannot claim the shorter span.
        matched_key: str | None = None
        matched_span = 0
        for span in range(min(max_lookahead, len(numbered_lines) - index), 0, -1):
            candidate = " ".join(text for _, text in numbered_lines[index : index + span])
            key = _normalise(candidate)
            if key in wanted and key not in seen:
                matched_key, matched_span = key, span
                break

        if matched_key is not None:
            flush()
            seen.add(matched_key)
            current_heading = wanted[matched_key]
            current_page = page_number
            buffer = []
            index += matched_span
            continue

        buffer.append(line)
        index += 1
    flush()

    warnings = [
        f"declared section not found in the PDF: {heading!r}"
        for key, heading in wanted.items()
        if key not in seen
    ]
    return [s for s in sections if s.text], warnings


def load_document(manifest: ManifestDocument) -> LoadedDocument:
    """Extract, clean and section one document."""

    pages = extract_pages(manifest.path)
    numbered = clean_pages(pages, manifest.page_shape)
    sections, warnings = split_sections(numbered, manifest.sections)

    if not sections:
        warnings.append("no text survived cleaning")
    if manifest.page_count and len(pages) != manifest.page_count:
        warnings.append(
            f"manifest says {manifest.page_count} pages, the PDF has {len(pages)}"
        )

    return LoadedDocument(
        manifest=manifest,
        sections=sections,
        content_hash=content_hash(manifest.path),
        warnings=warnings,
    )


def _normalise_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def strip_echoes(text: str, title: str, heading: str | None) -> str:
    """Drop lines that merely repeat the document title or the section heading.

    The print view of every page opens with its own title, so the unsectioned
    lead of most documents extracts as "Colonoscopy\\nColonoscopy". Left in, that
    becomes a node whose entire content is the word a patient is most likely to
    type — a strong similarity match holding no answer, which is exactly the
    near-miss failure the guards in the plan exist to prevent.

    This filters on content, not length. Several correct answers in this corpus
    are one short sentence: "You don't need any special preparations for a
    hearing test" is the whole answer to a benchmark question, and a
    minimum-size rule would have discarded it.
    """

    echoes = {_normalise_heading(title)}
    if heading:
        echoes.add(_normalise_heading(heading))
    kept = [
        line
        for line in text.splitlines()
        if line.strip() and _normalise_heading(line) not in echoes
    ]
    return "\n".join(kept)


def to_llamaindex_documents(document: LoadedDocument) -> list["Document"]:
    """Emit one LlamaIndex Document per section, carrying citation metadata.

    Metadata is split into what the embedder sees and what it does not. The
    title and section heading are embedded with the text: a node from the middle
    of a section is otherwise anonymous prose, and the heading — a literal
    patient question here — states its topic. Everything else (source URL, page
    number, content hash) is citation bookkeeping that would only add noise to
    the vector, so it is excluded from embedding while remaining queryable.
    """

    from llama_index.core.schema import Document

    manifest = document.manifest
    documents: list[Document] = []
    for index, section in enumerate(document.sections):
        text = strip_echoes(section.text, manifest.title, section.heading)
        if not text.strip():
            continue
        documents.append(
            Document(
                id_=f"{manifest.document_id}::{index}",
                text=text,
                metadata={
                    # NOT "document_id": that is a reserved LlamaIndex metadata
                    # key. node_to_metadata_dict() overwrites it with the node's
                    # ref_doc_id, so our value would never reach the queryable
                    # column -- it would survive only inside the _node_content
                    # blob, where filters cannot see it. Silent, and it would
                    # make every re-ingest append a duplicate corpus because the
                    # delete-by-filter matched nothing.
                    "corpus_document_id": manifest.document_id,
                    "title": manifest.title,
                    "section": section.heading or "",
                    "category": manifest.category,
                    "page_number": section.page_number,
                    "source_url": manifest.source_url or "",
                    # A citation to health guidance is worth little without a
                    # date: "MedlinePlus, reviewed July 2024" is checkable,
                    # "MedlinePlus" is not.
                    "last_updated": (
                        manifest.last_updated.isoformat()
                        if manifest.last_updated
                        else ""
                    ),
                    "content_fingerprint": document.fingerprint,
                },
                excluded_embed_metadata_keys=EXCLUDED_FROM_EMBEDDING,
                excluded_llm_metadata_keys=EXCLUDED_FROM_PROMPT,
            )
        )
    return documents


# Embedded with the chunk text, because they state what the chunk is about.
EMBEDDED_METADATA_KEYS = ("title", "section")

EXCLUDED_FROM_EMBEDDING = [
    "corpus_document_id",
    "category",
    "page_number",
    "source_url",
    "last_updated",
    "content_fingerprint",
]

# The generation prompt gets the citation-relevant fields and nothing else.
EXCLUDED_FROM_PROMPT = ["content_fingerprint", "category"]

# Everything a citation needs, for the retrieval layer to read back.
CITATION_METADATA_KEYS = (
    "corpus_document_id",
    "title",
    "section",
    "page_number",
    "source_url",
    "last_updated",
)


def content_hash(path: Path) -> str:
    """Hash the PDF bytes.

    Hashing the source file rather than the extracted text means a re-run is
    skipped only when the document itself is unchanged. Extraction and cleaning
    are code, and a change to either must force a re-ingest — which it does,
    because the ingest CLI also compares the pipeline version.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()
