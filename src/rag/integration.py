"""The seam between the knowledge branch and the chat application.

`src/chatbot.py` must not import a vector store, a retriever or a node parser.
It takes one callable — built here, injected at startup like `client` and
`visit_repository` — and everything behind it stays inside `src/rag/`.

The other half of this module's job is degradation. The knowledge store is a new
runtime dependency for an application that had none, and a database that will not
start must not stop a patient recording their symptoms. Every failure path here
returns None, which the caller reads as "no knowledge answer this turn" and
answers from the curated content exactly as it did before.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .config import SETTINGS, database_url
    from .embeddings import build_embed_model
    from .pipeline import KnowledgeAnswer, answer_knowledge_question
    from .query import is_information_request
    from .retrievers import BasicChunkRetriever
    from .store import KnowledgeStore
except ImportError:  # pragma: no cover - allows running as a script
    from config import SETTINGS, database_url
    from embeddings import build_embed_model
    from pipeline import KnowledgeAnswer, answer_knowledge_question
    from query import is_information_request
    from retrievers import BasicChunkRetriever
    from store import KnowledgeStore


@dataclass
class KnowledgeBranch:
    """Answers knowledge questions, or returns None so the caller carries on."""

    retriever: object
    chat_client: object

    def answer(self, question: str) -> tuple[str, dict] | None:
        """Return the reply text and the turn record, or None to stand down.

        None means "this turn produced nothing to say", which happens when the
        branch has no answer and no curated fallback of its own. It is not an
        error, and the caller must not surface it as one.
        """

        # Stand down on anything that is not a request for information. A
        # symptom the patient is reporting is content for the intake workflow;
        # answering it with "I don't have documentation covering that" trades a
        # turn of progress for an apology about a question nobody asked.
        if not is_information_request(question):
            return None

        try:
            result = answer_knowledge_question(
                question, self.retriever, self.chat_client
            )
        except Exception:  # noqa: BLE001 - a broken branch must not break intake
            return None

        if not result.text.strip():
            return None
        return result.text, _to_record(question, result)


def _to_record(question: str, result: KnowledgeAnswer) -> dict:
    """Project the branch result onto plain data for the chat layer to type.

    Deliberately a dict rather than the `RagTurn` model. This package is
    imported both as `rag` and as `src.rag` depending on entry point, and the
    repo's import shim means `models` and `src.models` can both be loaded as
    separate modules with separate classes. A model instance built here would
    then fail validation when assigned to state built there — which it did, once.
    Plain data crosses that boundary safely; the chat layer owns the typing.
    """

    return dict(
        question=question,
        status=result.status,
        source=result.source,
        citations=[
            dict(
                marker=citation.marker,
                document_id=citation.document_id,
                title=citation.title,
                section=citation.section,
                page_number=citation.page_number,
                source_url=citation.source_url,
                last_updated=citation.last_updated,
            )
            for citation in result.citations
        ],
        retrieved=result.retrieved,
        uncovered=list(result.uncovered),
        context_tokens=result.context_tokens,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        retrieval_latency_ms=result.retrieval_latency_ms,
        total_latency_ms=result.total_latency_ms,
    )


def build_knowledge_branch(chat_client) -> KnowledgeBranch | None:
    """Construct the branch, or return None when RAG is not provisioned.

    Called once at startup. Returning None rather than raising is the whole
    point: an application without a database keeps working, minus one feature.
    """

    if database_url() is None:
        return None
    try:
        store = KnowledgeStore()
        store.healthcheck()
        retriever = BasicChunkRetriever(store, build_embed_model())
    except Exception:  # noqa: BLE001 - reported by the caller, not fatal
        return None
    return KnowledgeBranch(retriever=retriever, chat_client=chat_client)


def is_enabled() -> bool:
    return database_url() is not None and SETTINGS.mode in {
        "shadow",
        "preferred",
        "primary",
    }
