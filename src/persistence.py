"""Email-indexed persistence for explicitly confirmed visit summaries."""

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
PERSON_INDEX_FILENAME = "person_index.json"


class ConfirmedVisitRecord(DomainModel):
    """Versioned storage record created only after explicit user confirmation."""

    visit_id: str
    # The lookup key for matching repeat visits. ``None`` when the patient never
    # gave an address; the record is still valid, it simply is not indexed.
    email: str | None = None
    session_reference: str
    schema_version: str = VISIT_SCHEMA_VERSION
    status: Literal["confirmed"] = "confirmed"
    created_at: datetime
    updated_at: datetime
    summary_text: str
    visit_data: VisitData


class JsonVisitRepository:
    """Persist each confirmed person record to a stable email-indexed JSON file."""

    def __init__(self, directory: Path):
        self.directory = directory

    def _path_for(self, visit_id: str) -> Path:
        return self.directory / f"{visit_id}.json"

    def _index_path(self) -> Path:
        return self.directory / PERSON_INDEX_FILENAME

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().lower()

    def _load_index(self) -> dict[str, str]:
        try:
            payload = json.loads(self._index_path().read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        index: dict[str, str] = {}
        for email, visit_id in payload.items():
            if isinstance(email, str) and isinstance(visit_id, str):
                index[self._normalize_email(email)] = visit_id
        return index

    def _write_atomic_json(self, path: Path, payload: dict[str, object]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        temporary_path = self.directory / f".{path.stem}.{uuid4().hex}.tmp"
        try:
            with temporary_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _write_index(self, index: dict[str, str]) -> None:
        self._write_atomic_json(self._index_path(), index)

    def _load_record_by_visit_id(self, visit_id: str) -> ConfirmedVisitRecord:
        payload = json.loads(self._path_for(visit_id).read_text(encoding="utf-8"))
        return ConfirmedVisitRecord.model_validate(payload)

    @staticmethod
    def _merge_visit_data(existing: VisitData, updated: VisitData) -> VisitData:
        return VisitData.model_validate(
            {
                **existing.model_dump(exclude_none=True),
                **updated.model_dump(exclude_none=True),
            }
            )

    def _find_record_by_email(self, email: str) -> ConfirmedVisitRecord | None:
        normalized_email = self._normalize_email(email)
        for path in self.directory.glob("*.json"):
            if path.name == PERSON_INDEX_FILENAME:
                continue
            try:
                record = self._load_record_by_visit_id(path.stem)
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                continue
            record_email = self._normalize_email(record.email or record.visit_data.email or "")
            if record_email == normalized_email:
                return record
        return None

    def load_by_email(self, email: str) -> ConfirmedVisitRecord | None:
        normalized_email = self._normalize_email(email)
        visit_id = self._load_index().get(normalized_email)
        if visit_id:
            path = self._path_for(visit_id)
            if path.exists():
                return self._load_record_by_visit_id(visit_id)

        return self._find_record_by_email(normalized_email)

    def save_confirmed(self, state: ConversationState) -> Path:
        if state.phase is not ConversationPhase.COMPLETED or not state.confirmed:
            raise ValueError("Only completed and confirmed visits may be persisted.")
        if not state.summary_text:
            raise ValueError("A confirmed visit must have a summary.")

        # An email is how repeat visits are matched to an existing record, not a
        # condition of being a valid one. Clinical detail is collected before
        # contact details, so a patient can reasonably confirm a summary without
        # ever giving an address; that visit is still saved, just unindexed.
        normalized_email = self._normalize_email(state.visit_data.email or "")
        index = self._load_index()
        existing_record = None
        # Without an email there is nothing to match on, and matching every
        # anonymous visit to every other would merge unrelated patients.
        if normalized_email:
            existing_visit_id = index.get(normalized_email)
            if existing_visit_id:
                existing_path = self._path_for(existing_visit_id)
                if existing_path.exists():
                    existing_record = self._load_record_by_visit_id(existing_visit_id)
            if existing_record is None:
                existing_record = self._find_record_by_email(normalized_email)

        now = datetime.now(timezone.utc)
        if existing_record is not None:
            visit_id = existing_record.visit_id
            created_at = existing_record.created_at
            merged_visit_data = self._merge_visit_data(
                existing_record.visit_data,
                state.visit_data,
            )
            summary_text = state.summary_text or existing_record.summary_text
        else:
            visit_id = str(uuid4())
            created_at = now
            merged_visit_data = state.visit_data
            summary_text = state.summary_text

        merged_visit_data.email = normalized_email or None

        record = ConfirmedVisitRecord(
            visit_id=visit_id,
            email=normalized_email or None,
            session_reference=anonymize_session_id(state.session_id),
            created_at=created_at,
            updated_at=now,
            summary_text=summary_text,
            visit_data=merged_visit_data,
        )

        final_path = self._path_for(visit_id)
        self._write_atomic_json(final_path, record.model_dump(mode="json"))
        if normalized_email:
            index[normalized_email] = visit_id
            self._write_index(index)

        state.visit_id = visit_id
        state.persisted_at = now
        state.persistence_error = None
        return final_path

    def load(self, visit_id: str) -> ConfirmedVisitRecord:
        return self._load_record_by_visit_id(visit_id)
