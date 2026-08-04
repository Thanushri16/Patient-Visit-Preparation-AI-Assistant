"""Configuration for the RAG knowledge branch.

Every value the plan in `documentation/rag_architecture.md` names as tunable
lives here, so a tuning decision is a diff in one file rather than a constant
buried in a call site. Nothing here reads the network or the database.
"""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "clinical_docs"
MANIFEST_PATH = CORPUS_DIR / "manifest.yaml"


def env_value(name: str) -> str | None:
    """Read a setting from the environment, falling back to the .env file.

    Mirrors the loader already used for OPENAI_API_KEY in src/chatbot.py rather
    than adding a settings library for three values.
    """

    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[len("export "):].strip()
        key, separator, raw = entry.partition("=")
        if separator and key.strip() == name:
            resolved = raw.strip().strip('"').strip("'")
            return resolved or None
    return None

# ---------------------------------------------------------------------------
# Embedding model
#
# The vector width is a property of the chosen model, so it is declared here and
# substituted into migrations/001 at apply time rather than hard-coded in SQL.
# Switching model therefore means a schema rebuild, which src/rag/store.py
# detects and explains instead of letting pgvector fail on the first insert.
#
# `embed_query` and `embed_documents` are separate calls in embeddings.py even
# though OpenAI embedding models are symmetric. Asymmetric models -- BGE, E5 and
# most other local retrievers -- require an instruction prefix on the query and
# not on the passage, and a single `embed()` would make the wrong spelling the
# default if one is ever added.
# ---------------------------------------------------------------------------


class EmbeddingProfile(BaseModel):
    """Everything about an embedding model that the rest of the code needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str                      # "openai"
    model: str
    dimensions: int
    query_instruction: str = ""
    token_encoding: str = "cl100k_base"
    max_input_tokens: int = 8191


EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    "openai-small": EmbeddingProfile(
        backend="openai",
        model="text-embedding-3-small",
        dimensions=1536,
    ),
    "openai-large": EmbeddingProfile(
        backend="openai",
        model="text-embedding-3-large",
        dimensions=3072,
    ),
}

DEFAULT_EMBEDDING_PROFILE = "openai-small"


def embedding_profile() -> EmbeddingProfile:
    """Return the configured profile, defaulting to text-embedding-3-small."""

    key = (env_value("EMBEDDING_PROFILE") or DEFAULT_EMBEDDING_PROFILE).strip()
    if key not in EMBEDDING_PROFILES:
        raise RuntimeError(
            f"Unknown EMBEDDING_PROFILE {key!r}. "
            f"Choose one of: {', '.join(sorted(EMBEDDING_PROFILES))}"
        )
    return EMBEDDING_PROFILES[key]


EMBEDDING = embedding_profile()
EMBEDDING_MODEL = EMBEDDING.model
EMBEDDING_DIMENSIONS = EMBEDDING.dimensions
EMBEDDING_ENCODING = EMBEDDING.token_encoding

# Guards against a chunk size that would be silently truncated by the model.
MAX_MODEL_INPUT_TOKENS = EMBEDDING.max_input_tokens

# Bump when cleaning, sectioning or chunking changes in a way that alters the
# stored text. Recorded with each document's content hash, so a re-ingest is
# triggered by a code change as well as by an edited PDF -- hashing the source
# file alone would miss the former entirely.
PIPELINE_VERSION = 2

# The LlamaIndex table name. PGVectorStore prefixes it, so the physical table is
# data_knowledge_chunk.
VECTOR_TABLE_NAME = "knowledge_chunk"


class RagSettings(BaseModel):
    """Chunking, retrieval, and generation settings for the knowledge branch."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Rollout stage. See rag_architecture.md section 3.7.
    #   shadow    - run RAG, log it, return the curated answer
    #   preferred - RAG when evidence is sufficient, else the curated answer
    #   primary   - RAG, with the explicit fallback when evidence is missing
    mode: str = "shadow"

    # Ingestion
    chunk_size_tokens: int = Field(default=400, gt=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    embed_batch_size: int = Field(default=64, gt=0)

    # Retrieval
    top_k: int = Field(default=4, gt=0)
    max_context_tokens: int = Field(default=1600, gt=0)

    # Evidence sufficiency (A.4). Tuned against the 68-question benchmark after
    # the guards were implemented, not before.
    #
    # A sweep from 0.30 to 0.55 with all three A.4.2 guards active held
    # near-miss resistance at 10/10 throughout, so the floor is not what
    # separates a wrong answer from a right one — guard 3 is. What the floor
    # costs is recall on real questions: 25/35 answerable at 0.55, 29/35 at
    # 0.45. It is therefore set as low as it can go without admitting the
    # obviously unrelated, and the near-miss work is left to the guards.
    #
    # Lowered again to 0.35 after the full run: the sweep held near-miss
    # resistance at 10/10 all the way down, and 0.45 was still rejecting real
    # questions outright ("what does the ABCDE rule mean" tops out at 0.344,
    # because an acronym embeds poorly against prose that spells it out).
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)
    min_supporting_nodes: int = Field(default=1, gt=0)

    # Partial coverage (A.4.1)
    split_compound_questions: bool = True
    max_subquestions: int = Field(default=3, gt=0)

    # Near-miss guards (A.4.2).
    #
    # Guard 3 is ON, which the plan left conditional on evidence. The evidence:
    # with guards 1 and 2 alone, near-miss resistance is 7/10 and the three that
    # leak are all same-category misses — an upper-endoscopy question answered
    # from the colonoscopy page, an "MRI radiation dose" question answered from
    # the CT page, a PET question answered from CT preparation. Guards 1 and 2
    # structurally cannot see those: the category is right and the retrieval
    # clusters. Adding guard 3 takes resistance to 10/10 and costs nothing in
    # answerable recall. One extra short model call per knowledge question is
    # cheap next to a confident wrong answer about a medical procedure.
    enforce_category_consistency: bool = True
    isolated_node_similarity: float = Field(default=0.50, ge=0.0, le=1.0)
    answerability_check: bool = True

    # Generation
    generation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    branch_timeout_ms: int = Field(default=3000, gt=0)


SETTINGS = RagSettings()


def database_url() -> str | None:
    """Return the configured database URL, or None when RAG is not provisioned.

    Returning None rather than raising is deliberate: the knowledge branch must
    degrade to the curated answers in guidance.py when the store is absent, and
    the intake chain must not depend on a database at all.
    """

    return env_value("DATABASE_URL")


def require_database_url() -> str:
    """Return the database URL, or explain how to start one."""

    url = database_url()
    if url is None:
        raise RuntimeError(
            "DATABASE_URL is not set. Start the knowledge store with "
            "`docker compose up -d`, then copy the DATABASE_URL line from "
            ".env.example into .env."
        )
    return url
