"""Typed domain objects for visit data, workflow state, messages, and chat sessions."""

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Dates people actually type. Parsing is the application's job precisely so that
# no model has to reformat one — a date silently rewritten is a wrong date in a
# clinical record, and unlike a missing field it looks perfectly valid.
DATE_OF_BIRTH_FORMATS = (
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m.%d.%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %B %Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%b %d, %Y",
)

EARLIEST_PLAUSIBLE_BIRTH_YEAR = 1900


def _parse_date_of_birth(value: object) -> date | object:
    if value is None:
        return value
    if isinstance(value, date):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        parsed = None
        for format_name in DATE_OF_BIRTH_FORMATS:
            try:
                parsed = datetime.strptime(text, format_name).date()
                break
            except ValueError:
                continue
        if parsed is None:
            try:
                parsed = datetime.fromisoformat(text).date()
            except ValueError as exc:
                raise ValueError(
                    "Date of birth must use MM/DD/YYYY format."
                ) from exc
    else:
        return value

    # A year outside living memory means the value was mistyped or mangled in
    # transit. Rejecting it forces the question to be asked again, which is far
    # better than storing a birth date of 0605.
    if parsed.year < EARLIEST_PLAUSIBLE_BIRTH_YEAR:
        raise ValueError("Date of birth must use MM/DD/YYYY format.")
    return parsed


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
    plan_type: str | None = None
    member_id: str | None = None
    # ``False`` records an explicit "I have no insurance"; ``None`` means unasked.
    has_insurance: bool | None = None
    details_available: bool | None = None


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
    none — for example, an empty ``allergies`` list is a recorded "no known
    drug allergies" rather than an unanswered question.
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

    # Appointment logistics.
    visit_reason: str | None = Field(default=None, max_length=1_000)
    provider_name: str | None = Field(default=None, max_length=200)
    appointment_date: str | None = Field(default=None, max_length=200)
    appointment_time: str | None = Field(default=None, max_length=100)
    visit_type: str | None = Field(default=None, max_length=200)
    referral_source: str | None = Field(default=None, max_length=300)
    new_patient: bool | None = None
    patient_context: str | None = Field(default=None, max_length=500)

    # Pre-visit preparation details.
    documents_status: str | None = Field(default=None, max_length=500)
    fasting_status: str | None = Field(default=None, max_length=300)
    accessibility_needs: list[str] | None = None
    special_instructions: list[str] | None = None
    transportation_needs: str | None = Field(default=None, max_length=500)
    companion: str | None = Field(default=None, max_length=300)

    # Clinical detail.
    chief_complaint: str | None = Field(default=None, max_length=1_000)
    symptom_location: str | None = Field(default=None, max_length=300)
    symptom_onset: str | None = Field(default=None, max_length=200)
    symptom_duration: str | None = Field(default=None, max_length=200)
    symptom_severity: int | None = Field(default=None, ge=0, le=10)
    symptom_pattern: str | None = Field(default=None, max_length=300)
    symptom_progression: str | None = Field(default=None, max_length=300)
    aggravating_factors: str | None = Field(default=None, max_length=300)
    relieving_factors: str | None = Field(default=None, max_length=300)
    associated_symptoms: list[str] | None = None
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


class VisitDataPatch(VisitData):
    """Optional visit fields proposed by one extraction call.

    Every field is already optional on ``VisitData``, so a patch is structurally
    the same document. Deriving it keeps the extractor's response schema from
    drifting away from the canonical model when fields are added.
    """

    # Taken as free text, not as a typed date. Asking the model for an ISO date
    # made it reformat what the patient said, and it got that wrong — "06/05/1984"
    # came back as "0605-04-06". The application parses the date it was given.
    date_of_birth: str | None = None

    # Shares the parent's validator name so it replaces rather than compounds it;
    # the patch keeps the date exactly as written and never parses it.
    @field_validator("date_of_birth", mode="before")
    @classmethod
    def validate_date_of_birth(cls, value):
        if value is None:
            return None
        return value.isoformat() if isinstance(value, date) else str(value)


class FieldExtractionResult(DomainModel):
    """Typed output returned by the visit-field extractor prompt.

    One patch, not two. Splitting the output into "updates" and "corrections"
    asked the model to classify intent as well as extract, and it reliably got
    the classification backwards — echoing the whole existing record as updates
    and putting the *superseded* value under corrections, which then overwrote
    the new one. Whether a value is new or replaces an earlier answer is a
    question the application can settle by comparing against its own state, so
    the model is only asked what the message says.
    """

    fields: VisitDataPatch = Field(default_factory=VisitDataPatch)
    uncertain_fields: list[str] = Field(default_factory=list)
    # Fields the patient is taking back — "I never said I had nausea", "the
    # medications are all wrong, let me redo them". Retracting is not the same
    # as correcting: there may be no replacement value, and leaving the old one
    # in place would keep something in the record they have disowned.
    cleared_fields: list[str] = Field(default_factory=list)
    # Individual entries to drop, as "field:value" — "chief_complaint:nausea",
    # "current_medications:metformin". Clearing a whole field is too blunt when
    # a patient disowns one item out of several: "I never said I had nausea"
    # must not also erase the headache they did report.
    removed_items: list[str] = Field(default_factory=list)


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
