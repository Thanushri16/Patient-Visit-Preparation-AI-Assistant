"""Per-turn dependencies, kept out of graph state.

A checkpointer serialises state. An OpenAI client or a database pool in there
would either fail to serialise or, worse, be restored from a checkpoint as a
stale handle. Dependencies are therefore passed through `RunnableConfig` and
read once per node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TurnDeps:
    """Everything a turn needs that is not part of the conversation."""

    client: Any = None
    visit_repository: Any = None
    knowledge_branch: Any = None


def deps_from(config: dict | None) -> TurnDeps:
    """Read dependencies from a RunnableConfig, tolerating their absence."""

    configurable = (config or {}).get("configurable", {}) or {}
    found = configurable.get("deps")
    return found if isinstance(found, TurnDeps) else TurnDeps()
