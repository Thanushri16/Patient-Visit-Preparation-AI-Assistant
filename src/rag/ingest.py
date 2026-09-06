"""Ingestion CLI for the knowledge corpus.

    uv run python -m src.rag.ingest plan      # what would change, no API calls
    uv run python -m src.rag.ingest ingest    # embed and store what changed
    uv run python -m src.rag.ingest ingest --force
    uv run python -m src.rag.ingest status    # what is currently stored

The pipeline is the one the plan specifies: load, clean, chunk with the
LlamaIndex node parser, embed with the LlamaIndex embedding model, store in
PGVectorStore.

Idempotency is by content fingerprint — the PDF's hash plus PIPELINE_VERSION.
Hashing the PDF alone would be wrong: cleaning, sectioning and chunking are code,
and a change to any of them must force a re-ingest even though the source file is
untouched. Bumping PIPELINE_VERSION in config.py is how that is declared.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

try:
    from .chunking import nodes_from_loaded_document
    from .config import (
        EMBEDDING,
        PIPELINE_VERSION,
        SENTENCE_TABLE_NAME,
        SETTINGS,
        VECTOR_TABLE_NAME,
    )
    from .documents import ManifestDocument, load_document, load_manifest
    from .embeddings import build_embed_model, embed_nodes
    from .sentence_window import MAX_WINDOW, sentence_nodes_from_loaded_document
    from .store import KnowledgeStore
except ImportError:  # pragma: no cover - allows running as a script
    from chunking import nodes_from_loaded_document
    from config import (
        EMBEDDING,
        PIPELINE_VERSION,
        SENTENCE_TABLE_NAME,
        SETTINGS,
        VECTOR_TABLE_NAME,
    )
    from documents import ManifestDocument, load_document, load_manifest
    from embeddings import build_embed_model, embed_nodes
    from sentence_window import MAX_WINDOW, sentence_nodes_from_loaded_document
    from store import KnowledgeStore


@dataclass
class DocumentPlan:
    """What ingestion intends to do with one document, and why."""

    manifest: ManifestDocument
    action: str          # "ingest" | "reingest" | "skip"
    reason: str
    nodes: int
    warnings: list[str]


def build_plan(
    store: KnowledgeStore | None,
    force: bool = False,
    build_nodes=nodes_from_loaded_document,
) -> list[DocumentPlan]:
    """Decide what to do with each indexed document, without embedding anything.

    `build_nodes` selects the retrieval unit: 400-token chunks by default,
    sentences for Part C. The fingerprint logic is deliberately shared -- both
    tables key idempotency on the same content hash plus PIPELINE_VERSION, so a
    change to cleaning or sectioning re-ingests both rather than leaving one
    silently built from older text than the other.
    """

    documents, _ = load_manifest()
    plans: list[DocumentPlan] = []

    for manifest in documents:
        loaded = load_document(manifest)
        nodes = build_nodes(loaded)

        action, reason = "ingest", "not stored yet"
        if store is not None:
            stored = store.stored_fingerprint(manifest.document_id)
            if stored is not None:
                if force:
                    action, reason = "reingest", "forced"
                elif stored == loaded.fingerprint:
                    action, reason = "skip", "unchanged"
                elif stored.split(":")[0] == loaded.content_hash:
                    action, reason = "reingest", "pipeline version changed"
                else:
                    action, reason = "reingest", "source PDF changed"

        plans.append(
            DocumentPlan(
                manifest=manifest,
                action=action,
                reason=reason,
                nodes=len(nodes),
                warnings=loaded.warnings,
            )
        )
    return plans


def prune(store: KnowledgeStore) -> list[str]:
    """Remove stored documents the manifest no longer indexes.

    Un-indexing a document in the manifest does not, on its own, take it out of
    the store: `build_plan` only walks documents the manifest still lists, so an
    excluded one keeps its nodes and stays retrievable. That is the wrong
    behaviour for a content decision and a dangerous one for a licence decision —
    a document removed because indexing it is prohibited must actually stop being
    indexed.
    """

    indexed = {manifest.document_id for manifest in load_manifest()[0]}
    stored = {row["document_id"] for row in store.corpus_status()}
    removed = sorted(stored - indexed)
    for document_id in removed:
        store.delete_document(document_id)
    return removed


def ingest(
    store: KnowledgeStore,
    embed_model,
    force: bool = False,
    build_nodes=nodes_from_loaded_document,
) -> list[DocumentPlan]:
    """Embed and store every document whose content or pipeline has changed."""

    prune(store)
    plans = build_plan(store, force=force, build_nodes=build_nodes)
    for plan in plans:
        if plan.action == "skip":
            continue

        loaded = load_document(plan.manifest)
        nodes = build_nodes(loaded)
        if not nodes:
            plan.action, plan.reason = "skip", "no nodes produced"
            continue

        embed_nodes(embed_model, nodes)
        store.replace_document(plan.manifest.document_id, nodes)
    return plans


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_plan(plans: list[DocumentPlan]) -> None:
    width = max(len(plan.manifest.document_id) for plan in plans)
    for plan in plans:
        print(
            f"{plan.action:8s} {plan.manifest.document_id:<{width}}  "
            f"nodes={plan.nodes:3d}  ({plan.reason})"
        )
        for warning in plan.warnings:
            print(f"         ! {warning}")

    counts: dict[str, int] = {}
    for plan in plans:
        counts[plan.action] = counts.get(plan.action, 0) + 1
    summary = ", ".join(f"{action} {count}" for action, count in sorted(counts.items()))
    print(f"\n{len(plans)} documents: {summary} | {sum(p.nodes for p in plans)} nodes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Knowledge corpus ingestion.")
    parser.add_argument("command", choices=("plan", "ingest", "status"))
    parser.add_argument(
        "--force", action="store_true", help="re-ingest even when unchanged"
    )
    parser.add_argument(
        "--strategy",
        choices=("basic", "sentence_window", "both"),
        default="basic",
        help="which retrieval unit to build: 400-token chunks, sentences, or both",
    )
    args = parser.parse_args(argv)

    # (table, node builder) per strategy. Part C compares the two, so `both`
    # exists to keep the tables built from the same corpus revision -- ingesting
    # them separately invites comparing a strategy against a stale opponent.
    targets = {
        "basic": ((None, nodes_from_loaded_document),),
        "sentence_window": ((SENTENCE_TABLE_NAME, sentence_nodes_from_loaded_document),),
        "both": (
            (None, nodes_from_loaded_document),
            (SENTENCE_TABLE_NAME, sentence_nodes_from_loaded_document),
        ),
    }[args.strategy]

    if args.command == "plan":
        # Works without a database: reports what would be built from the corpus.
        for table, build_nodes in targets:
            store = _open_store(optional=True, table_name=table)
            print(f"--- {table or VECTOR_TABLE_NAME} ---")
            _print_plan(build_plan(store, build_nodes=build_nodes))
            if store is None:
                print("\nNo DATABASE_URL, so nothing was compared against the store.")
        return 0

    embed_model = None
    for table, build_nodes in targets:
        label = table or VECTOR_TABLE_NAME
        store = _open_store(optional=False, table_name=table)
        if store is None:
            return 1

        if args.command == "status":
            rows = store.corpus_status()
            print(f"--- {label} ---")
            if not rows:
                print("Empty. Run `ingest` first.")
                continue
            for row in rows:
                print(
                    f"{row['document_id']:32s} {row['category']:12s} "
                    f"nodes={row['nodes']:3d}"
                )
            print(f"{len(rows)} documents, {sum(r['nodes'] for r in rows)} nodes.\n")
            continue

        mismatch = store.dimension_mismatch()
        if mismatch:
            print(mismatch, file=sys.stderr)
            return 1

        dropped = prune(store)
        if dropped:
            print(f"Removed from {label}, no longer indexed: {', '.join(dropped)}\n")

        unit = (
            f"chunk_size={SETTINGS.chunk_size_tokens}, "
            f"overlap={SETTINGS.chunk_overlap_tokens}"
            if build_nodes is nodes_from_loaded_document
            else f"sentences, window +/-{MAX_WINDOW} stored"
        )
        print(
            f"--- {label} ---\n"
            f"Embedding with {EMBEDDING.model} ({EMBEDDING.dimensions}d), "
            f"{unit}, pipeline v{PIPELINE_VERSION}\n"
        )
        # Built once and shared: the two tables must be embedded by the same
        # model instance, or a profile change between them would go unnoticed.
        embed_model = embed_model or build_embed_model()
        _print_plan(ingest(store, embed_model, force=args.force, build_nodes=build_nodes))
    return 0


def _open_store(optional: bool, table_name: str | None = None) -> KnowledgeStore | None:
    try:
        store = KnowledgeStore(**({"table_name": table_name} if table_name else {}))
        store.healthcheck()
        return store
    except Exception as error:  # noqa: BLE001 - the CLI reports, it does not crash
        if optional:
            return None
        print(f"Cannot reach the knowledge store: {error}", file=sys.stderr)
        print(
            "Start it with `docker compose up -d`, then set DATABASE_URL in .env.",
            file=sys.stderr,
        )
        return None


if __name__ == "__main__":
    raise SystemExit(main())
