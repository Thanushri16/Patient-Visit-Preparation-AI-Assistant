"""Privacy-safe structured telemetry for prompt-chain node execution."""

import hashlib
import json
import logging
from datetime import datetime, timezone

from pydantic import Field

try:
    from .models import ConversationState, DomainModel
except ImportError:  # pragma: no cover - allows running as a script
    from models import ConversationState, DomainModel


LOGGER_NAME = "healthcare_chatbot.prompt_chain"
logger = logging.getLogger(LOGGER_NAME)
SAFE_METADATA_KEYS = {
    "accepted_field_count",
    "action",
    "file_extension",
    "handled",
    "missing_field_count",
    "reason_count",
    "rejected_field_count",
    "risk_level",
    "source",
}


class ChainEvent(DomainModel):
    """Non-clinical metadata emitted for one prompt-chain node execution."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node: str
    session_reference: str
    workflow: str | None = None
    phase_before: str
    phase_after: str
    success: bool
    latency_ms: float = Field(ge=0)
    prompt_version: str | None = None
    retry_count: int = Field(default=0, ge=0)
    error_category: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


def anonymize_session_id(session_id: str) -> str:
    """Create a stable, non-reversible reference without logging a raw session ID."""

    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def emit_chain_event(
    state: ConversationState,
    node: str,
    *,
    success: bool,
    latency_ms: float = 0.0,
    prompt_version: str | None = None,
    retry_count: int = 0,
    error_category: str | None = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
    phase_before: str | None = None,
) -> ChainEvent:
    """Log operational metadata without messages or collected patient values."""

    event = ChainEvent(
        node=node,
        session_reference=anonymize_session_id(state.session_id),
        workflow=state.workflow.value if state.workflow else None,
        phase_before=phase_before or state.phase.value,
        phase_after=state.phase.value,
        success=success,
        latency_ms=round(latency_ms, 3),
        prompt_version=prompt_version,
        retry_count=retry_count,
        error_category=error_category,
        metadata={
            key: value
            for key, value in (metadata or {}).items()
            if key in SAFE_METADATA_KEYS
        },
    )
    logger.info(json.dumps(event.model_dump(mode="json"), sort_keys=True))
    return event
