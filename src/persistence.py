"""Atomic UUID-based persistence for explicitly confirmed visit summaries."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

try:
    from .models import ConversationPhase, ConversationState, DomainModel, VisitData
    from .observability import anonymize_session_id
except ImportError:  # pragma: no cover - allows running as a script
    from models import ConversationPhase, ConversationState, DomainModel, VisitData
    from observability import anonymize_session_id


VISIT_SCHEMA_VERSION = "1.0"


class ConfirmedVisitRecord(DomainModel):
    """Versioned storage record created only after explicit user confirmation."""

    visit_id: str
    session_reference: str
    schema_version: str = VISIT_SCHEMA_VERSION
    status: Literal["confirmed"] = "confirmed"
    created_at: datetime
    summary_text: str
    visit_data: VisitData


class JsonVisitRepository:
    """Persist each confirmed visit to a separate, atomically written JSON file."""

    def __init__(self, directory: Path):
        self.directory = directory

    def _path_for(self, visit_id: str) -> Path:
        return self.directory / f"{visit_id}.json"

    def save_confirmed(self, state: ConversationState) -> Path:
        if state.phase is not ConversationPhase.COMPLETED or not state.confirmed:
            raise ValueError("Only completed and confirmed visits may be persisted.")
        if not state.summary_text:
            raise ValueError("A confirmed visit must have a summary.")

        if state.visit_id:
            existing_path = self._path_for(state.visit_id)
            if existing_path.exists():
                return existing_path

        visit_id = str(uuid4())
        created_at = datetime.now(timezone.utc)
        record = ConfirmedVisitRecord(
            visit_id=visit_id,
            session_reference=anonymize_session_id(state.session_id),
            created_at=created_at,
            summary_text=state.summary_text,
            visit_data=state.visit_data,
        )

        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        final_path = self._path_for(visit_id)
        temporary_path = self.directory / f".{visit_id}.{uuid4().hex}.tmp"
        try:
            with temporary_path.open("x", encoding="utf-8") as handle:
                json.dump(record.model_dump(mode="json"), handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, final_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        state.visit_id = visit_id
        state.persisted_at = created_at
        state.persistence_error = None
        return final_path

    def load(self, visit_id: str) -> ConfirmedVisitRecord:
        payload = json.loads(self._path_for(visit_id).read_text(encoding="utf-8"))
        return ConfirmedVisitRecord.model_validate(payload)
