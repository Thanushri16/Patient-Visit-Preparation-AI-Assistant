"""Typed domain objects for visit data, workflow state, messages, and chat sessions."""

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _parse_date_of_birth(value: object) -> date | object:
    if value is None or isinstance(value, date):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    for format_name in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, format_name).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).date()
    except ValueError as exc:
        raise ValueError("Date of birth must use MM/DD/YYYY format.") from exc


class DomainModel(BaseModel):
    """Forbid extra fields, strip strings, and validate values assigned after creation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ConversationPhase(StrEnum):
    MENU = "menu"
    COLLECTING = "collecting"
    REVIEWING = "reviewing"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class WorkflowType(StrEnum):
    APPOINTMENT_PREPARATION = "appointment_preparation"
    REPORT_NEW_SYMPTOMS = "report_new_symptoms"
    REVIEW_HEALTH_NOTES = "review_health_notes"
    REPORT_ALLERGY = "report_allergy"
    MEDICATION_QUESTION = "medication_question"
    EMERGENCY_SUPPORT = "emergency_support"
    VIEW_SUMMARY = "view_summary"


class ConfirmationAction(StrEnum):
    CONFIRM = "confirm"
    CORRECT = "correct"
    UNCLEAR = "unclear"


class Address(DomainModel):
    street: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None


class InsuranceInfo(DomainModel):
    provider_name: str | None = None
    policy_number: str | None = None
    group_number: str | None = None


class Measurement(DomainModel):
    value: float = Field(gt=0)
    unit: str = Field(min_length=1, max_length=20)


class Medication(DomainModel):
    name: str = Field(min_length=1, max_length=200)
    dosage: str | None = Field(default=None, max_length=100)
    frequency: str | None = Field(default=None, max_length=100)
    purpose: str | None = Field(default=None, max_length=300)


class Allergy(DomainModel):
    allergen: str = Field(min_length=1, max_length=200)
    reaction: str | None = Field(default=None, max_length=500)
    severity: str | None = Field(default=None, max_length=100)


class VisitData(DomainModel):
    """Canonical structured data collected for appointment preparation.

    A value of ``None`` means that the user has not answered the field. For
    collection fields, an empty list means that the user explicitly reported
    none.
    """

    patient_name: str | None = Field(default=None, max_length=200)
    date_of_birth: date | None = None
    gender: str | None = Field(default=None, max_length=100)
    height: Measurement | None = None
    weight: Measurement | None = None
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=50)
    address: Address | None = None
    insurance_info: InsuranceInfo | None = None
    chief_complaint: str | None = Field(default=None, max_length=1_000)
    symptom_duration: str | None = Field(default=None, max_length=200)
    symptom_severity: int | None = Field(default=None, ge=0, le=10)
    medical_conditions: list[str] | None = None
    current_medications: list[Medication] | None = None
    allergies: list[Allergy] | None = None
    lifestyle_info: str | None = Field(default=None, max_length=1_000)
    emergency_symptoms: list[str] | None = None
    notes: list[str] | None = None

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def validate_date_of_birth(cls, value):
        return _parse_date_of_birth(value)


class VisitDataPatch(DomainModel):
    """Optional visit fields proposed by one extraction call."""

    patient_name: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    height: Measurement | None = None
    weight: Measurement | None = None
    email: str | None = None
    phone: str | None = None
    address: Address | None = None
    insurance_info: InsuranceInfo | None = None
    chief_complaint: str | None = None
    symptom_duration: str | None = None
    symptom_severity: int | None = Field(default=None, ge=0, le=10)
    medical_conditions: list[str] | None = None
    current_medications: list[Medication] | None = None
    allergies: list[Allergy] | None = None
    lifestyle_info: str | None = None
    emergency_symptoms: list[str] | None = None
    notes: list[str] | None = None

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def validate_date_of_birth(cls, value):
        return _parse_date_of_birth(value)


class FieldExtractionResult(DomainModel):
    """Typed output returned by the visit-field extractor prompt."""

    updates: VisitDataPatch = Field(default_factory=VisitDataPatch)
    corrections: VisitDataPatch = Field(default_factory=VisitDataPatch)
    uncertain_fields: list[str] = Field(default_factory=list)


class ConfirmationResult(DomainModel):
    """Typed classification of a summary confirmation response."""

    action: ConfirmationAction
    correction_text: str | None = None


class ConversationState(DomainModel):
    """Source of truth for workflow phase, type, progress, and collected data."""

    session_id: str = Field(min_length=1, max_length=200)
    phase: ConversationPhase = ConversationPhase.MENU
    workflow: WorkflowType | None = None
    visit_data: VisitData = Field(default_factory=VisitData)
    missing_fields: list[str] = Field(default_factory=list)
    requested_field: str | None = None
    validation_errors: dict[str, str] = Field(default_factory=dict)
    extraction_retry_count: int = Field(default=0, ge=0)
    validation_attempt_count: int = Field(default=0, ge=0)
    confirmation_attempt_count: int = Field(default=0, ge=0)
    emergency_detected: bool = False
    confirmed: bool = False
    summary_text: str | None = None
    visit_id: str | None = None
    persisted_at: datetime | None = None
    persistence_error: str | None = None


class ChatMessage(DomainModel):
    """Contains a single message in a conversation, including the role of the sender and the content of the message."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatSession(DomainModel):
    """Contains the state of a conversation and the messages exchanged in that conversation."""

    state: ConversationState
    messages: list[ChatMessage] = Field(default_factory=list)
    expires_at: float = Field(gt=0)
