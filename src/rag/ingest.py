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
    from .config import EMBEDDING, PIPELINE_VERSION, SETTINGS
    from .documents import ManifestDocument, load_document, load_manifest
    from .embeddings import build_embed_model, embed_nodes
    from .store import KnowledgeStore
except ImportError:  # pragma: no cover - allows running as a script
    from chunking import nodes_from_loaded_document
    from config import EMBEDDING, PIPELINE_VERSION, SETTINGS
    from documents import ManifestDocument, load_document, load_manifest
    from embeddings import build_embed_model, embed_nodes
    from store import KnowledgeStore


@dataclass
class DocumentPlan:
    """What ingestion intends to do with one document, and why."""

    manifest: ManifestDocument
    action: str          # "ingest" | "reingest" | "skip"
    reason: str
    nodes: int
    warnings: list[str]


def build_plan(store: KnowledgeStore | None, force: bool = False) -> list[DocumentPlan]:
    """Decide what to do with each indexed document, without embedding anything."""

    documents, _ = load_manifest()
    plans: list[DocumentPlan] = []

    for manifest in documents:
        loaded = load_document(manifest)
        nodes = nodes_from_loaded_document(loaded)

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


def ingest(
    store: KnowledgeStore,
    embed_model,
    force: bool = False,
) -> list[DocumentPlan]:
    """Embed and store every document whose content or pipeline has changed."""

    plans = build_plan(store, force=force)
    for plan in plans:
        if plan.action == "skip":
            continue

        loaded = load_document(plan.manifest)
        nodes = nodes_from_loaded_document(loaded)
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
    args = parser.parse_args(argv)

    if args.command == "plan":
        # Works without a database: reports what would be built from the corpus.
        store = _open_store(optional=True)
        _print_plan(build_plan(store))
        if store is None:
            print("\nNo DATABASE_URL, so nothing was compared against the store.")
        return 0

    store = _open_store(optional=False)
    if store is None:
        return 1

    if args.command == "status":
        rows = store.corpus_status()
        if not rows:
            print("The knowledge store is empty. Run `ingest` first.")
            return 0
        for row in rows:
            print(
                f"{row['document_id']:32s} {row['category']:12s} nodes={row['nodes']:3d}"
            )
        print(f"\n{len(rows)} documents, {sum(r['nodes'] for r in rows)} nodes.")
        return 0

    mismatch = store.dimension_mismatch()
    if mismatch:
        print(mismatch, file=sys.stderr)
        return 1

    print(
        f"Embedding with {EMBEDDING.model} ({EMBEDDING.dimensions}d), "
        f"chunk_size={SETTINGS.chunk_size_tokens}, "
        f"overlap={SETTINGS.chunk_overlap_tokens}, pipeline v{PIPELINE_VERSION}\n"
    )
    _print_plan(ingest(store, build_embed_model(), force=args.force))
    return 0


def _open_store(optional: bool) -> KnowledgeStore | None:
    try:
        store = KnowledgeStore()
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
