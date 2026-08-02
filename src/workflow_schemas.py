"""Workflow field contracts and deterministic completeness helpers."""

from pydantic import ConfigDict, Field, model_validator

try:
    from .models import ConversationState, DomainModel, VisitData, WorkflowType
except ImportError:  # pragma: no cover - allows running as a script
    from models import ConversationState, DomainModel, VisitData, WorkflowType


class WorkflowSchema(DomainModel):
    """Deterministic collection contract for a single chatbot workflow."""

    # Workflow definitions are application configuration and must not be
    # reassigned at runtime. Other validation settings inherit from DomainModel.
    model_config = ConfigDict(frozen=True)

    workflow: WorkflowType
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    question_by_field: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_field_configuration(self):
        # Validate that required and optional fields are individually unique.
        required = set(self.required_fields)
        optional = set(self.optional_fields)
        configured = required | optional
        known_visit_fields = set(VisitData.model_fields)

        if len(required) != len(self.required_fields):
            raise ValueError("Required workflow fields must be unique.")
        if len(optional) != len(self.optional_fields):
            raise ValueError("Optional workflow fields must be unique.")
        # A field cannot be both required and optional in the same workflow.
        if required & optional:
            raise ValueError("A workflow field cannot be both required and optional.")
        # Every configured field must exist on the canonical VisitData model.
        if unknown_fields := configured - known_visit_fields:
            raise ValueError(f"Unknown VisitData fields: {sorted(unknown_fields)}")
        # Questions may only target fields collected by this workflow.
        if unknown_question_fields := set(self.question_by_field) - configured:
            raise ValueError(
                f"Questions reference fields outside the workflow: {sorted(unknown_question_fields)}"
            )
        return self


INTAKE_OPTIONAL_FIELDS = (
    "gender",
    "height",
    "weight",
    "address",
    "referral_source",
    "new_patient",
    "patient_context",
    "documents_status",
    "fasting_status",
    "accessibility_needs",
    "special_instructions",
    "transportation_needs",
    "companion",
    "symptom_location",
    "symptom_onset",
    "symptom_pattern",
    "symptom_progression",
    "aggravating_factors",
    "relieving_factors",
    "associated_symptoms",
    "lifestyle_info",
    "emergency_symptoms",
    "notes",
)

SHARED_IDENTITY_CONTACT_FIELDS = (
    "patient_name",
    "date_of_birth",
    "email",
    "phone",
)

SHARED_IDENTITY_CONTACT_QUESTIONS = {
    "patient_name": "What name should I use for this visit summary?",
    "date_of_birth": "What is your date of birth? Please use MM/DD/YYYY.",
    "email": "What email address should I include for this visit?",
    "phone": "What phone number should I include for this visit?",
}


def _shared_identity_contact_questions() -> dict[str, str]:
    return dict(SHARED_IDENTITY_CONTACT_QUESTIONS)


def _shared_required_fields(*workflow_fields: str) -> tuple[str, ...]:
    """Order the workflow's own fields ahead of identity and contact details.

    Follow-up questions come from the head of the missing-field list, so a
    patient who reports a cough is asked how long they have had it before
    anything administrative. Identity still has to be collected, though: a
    confirmed visit summary with no name, date of birth or contact number is
    not a usable clinical record, so these are required and simply come last.
    """

    return workflow_fields + SHARED_IDENTITY_CONTACT_FIELDS


def _all_visit_fields_except_shared_identity() -> tuple[str, ...]:
    return tuple(
        field_name
        for field_name in VisitData.model_fields
        if field_name not in SHARED_IDENTITY_CONTACT_FIELDS
    )


# Whole-body symptoms have no location to ask about, so the question is skipped
# rather than asking a dizzy patient where the dizziness is.
NON_LOCALIZED_COMPLAINTS = (
    "dizz", "nausea", "nauseous", "fever", "chills", "fatigue", "tired",
    "insomnia", "sleep", "anxiety", "weakness", "vomit", "appetite", "malaise",
)

CLINICAL_QUESTIONS = {
    "chief_complaint": "What symptoms are you experiencing?",
    "symptom_location": "Whereabouts are you feeling it?",
    "symptom_onset": "When did it first start, and how often does it happen?",
    "symptom_duration": (
        "How long have you had these symptoms? If you only know a number, please "
        "include the unit, like hours, days, weeks, months, or years."
    ),
    "symptom_severity": "On a scale from 0 to 10, how severe is it?",
    "symptom_pattern": "Is it constant, or does it come and go?",
    "medical_conditions": "Do you have any existing medical conditions? You can say none.",
    "current_medications": "What medications are you currently taking? You can say none.",
    "allergies": "Do you have any medication or other allergies? You can say none.",
}

APPOINTMENT_QUESTIONS = {
    "visit_reason": "What is the reason for your visit?",
    "provider_name": "Which doctor are you seeing?",
    "appointment_date": "When is your appointment?",
    "appointment_time": "What time is your appointment?",
    "visit_type": "Is the appointment in person or a telehealth visit?",
    "insurance_info": "Which insurance do you have? You can say none.",
    "accessibility_needs": "What accommodations do you need?",
    "documents_status": "Do you already have your ID and insurance card ready?",
}


WORKFLOW_SCHEMAS: dict[WorkflowType, WorkflowSchema] = {
    WorkflowType.APPOINTMENT_PREPARATION: WorkflowSchema(
        workflow=WorkflowType.APPOINTMENT_PREPARATION,
        # A visit record is not usable without the time or the format of the
        # appointment, so both are collected rather than left to chance.
        required_fields=_shared_required_fields(
            "visit_reason",
            "appointment_date",
            "appointment_time",
            "provider_name",
            "visit_type",
            "insurance_info",
        ),
        optional_fields=INTAKE_OPTIONAL_FIELDS
        + (
            "chief_complaint",
            "symptom_duration",
            "symptom_severity",
            "medical_conditions",
            "current_medications",
            "allergies",
        ),
        question_by_field={
            **_shared_identity_contact_questions(),
            **CLINICAL_QUESTIONS,
            **APPOINTMENT_QUESTIONS,
        },
    ),
    WorkflowType.REPORT_NEW_SYMPTOMS: WorkflowSchema(
        workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
        # A symptom is only usable by a clinician with its location, onset,
        # duration, severity, and pattern, so all five are collected. The
        # adaptive follow-up picks whichever of them the message left open.
        # Ordered the way a clinician takes a history: what and where, then how
        # bad, then how long, then when and in what pattern.
        required_fields=_shared_required_fields(
            "chief_complaint",
            "symptom_location",
            "symptom_severity",
            "symptom_duration",
            "symptom_onset",
            "symptom_pattern",
        ),
        optional_fields=(
            "symptom_progression",
            "aggravating_factors",
            "relieving_factors",
            "associated_symptoms",
            "medical_conditions",
            "current_medications",
            "allergies",
            "emergency_symptoms",
            "notes",
        ),
        question_by_field={
            **_shared_identity_contact_questions(),
            **CLINICAL_QUESTIONS,
            "symptom_severity": "On a scale from 0 to 10, how severe are the symptoms?",
        },
    ),
    WorkflowType.REPORT_ALLERGY: WorkflowSchema(
        workflow=WorkflowType.REPORT_ALLERGY,
        required_fields=_shared_required_fields("allergies"),
        # A reported allergy usually arrives with its symptom and often with the
        # medication taken instead, so those neighbouring fields are collected
        # here rather than discarded for belonging to another workflow.
        optional_fields=(
            "current_medications",
            "medical_conditions",
            "chief_complaint",
            "emergency_symptoms",
            "notes",
        ),
        question_by_field={
            **_shared_identity_contact_questions(),
            "allergies": "What are you allergic to, and what reaction do you experience?",
            "current_medications": "What medications are you currently taking? You can say none.",
            "medical_conditions": "Do you have any existing medical conditions? You can say none.",
            "chief_complaint": "What symptoms are you experiencing?",
        },
    ),
    WorkflowType.MEDICATION_QUESTION: WorkflowSchema(
        workflow=WorkflowType.MEDICATION_QUESTION,
        required_fields=_shared_required_fields("current_medications"),
        optional_fields=("allergies", "medical_conditions", "chief_complaint", "notes"),
        question_by_field={
            **_shared_identity_contact_questions(),
            "current_medications": "Which medication would you like to discuss?",
            "allergies": "Do you have any medication allergies? You can say none.",
            "medical_conditions": "Do you have any existing medical conditions? You can say none.",
            "chief_complaint": "What symptoms are you experiencing?",
        },
    ),
    WorkflowType.REVIEW_HEALTH_NOTES: WorkflowSchema(
        workflow=WorkflowType.REVIEW_HEALTH_NOTES,
        optional_fields=SHARED_IDENTITY_CONTACT_FIELDS
        + _all_visit_fields_except_shared_identity(),
        question_by_field=_shared_identity_contact_questions(),
    ),
    WorkflowType.VIEW_SUMMARY: WorkflowSchema(
        workflow=WorkflowType.VIEW_SUMMARY,
        optional_fields=SHARED_IDENTITY_CONTACT_FIELDS
        + _all_visit_fields_except_shared_identity(),
        question_by_field=_shared_identity_contact_questions(),
    ),
    WorkflowType.EMERGENCY_SUPPORT: WorkflowSchema(
        workflow=WorkflowType.EMERGENCY_SUPPORT,
        optional_fields=("emergency_symptoms", "notes"),
    ),
}


def get_workflow_schema(workflow: WorkflowType) -> WorkflowSchema:
    return WORKFLOW_SCHEMAS[workflow]


def get_missing_fields(
    workflow: WorkflowType,
    visit_data: VisitData,
    already_asked: frozenset[str] = frozenset(),
) -> list[str]:
    """Return unanswered required fields in their configured collection order."""

    schema = get_workflow_schema(workflow)
    # Conditional detail comes first: once a patient names a medication, the
    # natural next question is its dose, not the next unrelated intake field.
    missing_fields = get_conditional_missing_fields(workflow, visit_data)
    # "My leg hurts" records a location but not a usable one; which leg matters
    # clinically. `already_asked` stops that becoming a loop — see
    # `refresh_state_completeness`.
    if (
        needs_laterality(visit_data)
        and "symptom_location" not in already_asked
        and "symptom_location" in get_workflow_schema(workflow).required_fields
    ):
        missing_fields.append("symptom_location")
    skipped = _inapplicable_fields(visit_data)
    missing_fields.extend(
        field_name
        for field_name in schema.required_fields
        if getattr(visit_data, field_name) is None and field_name not in skipped
    )
    return missing_fields


def _inapplicable_fields(visit_data: VisitData) -> set[str]:
    """Return required fields that this particular complaint cannot answer."""

    complaint = (visit_data.chief_complaint or "").lower()
    if complaint and any(token in complaint for token in NON_LOCALIZED_COMPLAINTS):
        return {"symptom_location"}
    return set()


# Body parts a patient has two of. "My leg hurts" is not yet a usable location.
PAIRED_BODY_PARTS = (
    "leg", "arm", "hand", "foot", "knee", "shoulder", "hip", "ankle", "wrist",
    "elbow", "ear", "eye", "side", "breast", "thigh", "calf", "ankle",
)
LATERALITY_WORDS = ("left", "right", "both", "either", "bilateral")


def needs_laterality(visit_data: VisitData) -> bool:
    """Report whether a recorded location names a paired part without a side."""

    location = (visit_data.symptom_location or "").lower()
    if not location:
        return False
    if any(word in location for word in LATERALITY_WORDS):
        return False
    return any(part in location for part in PAIRED_BODY_PARTS)


def get_conditional_missing_fields(
    workflow: WorkflowType,
    visit_data: VisitData,
) -> list[str]:
    """Return nested detail that becomes relevant only once a section is started.

    These follow-ups are what turn "I take metformin" into a question about the
    dose, so they are ordered ahead of the workflow's remaining top-level fields
    by `get_missing_fields`.
    """

    missing_fields: list[str] = []

    # Which carrier the patient is with is what preparation needs. The plan type
    # and policy number are worth asking for once, but demanding them blocked
    # every appointment record from completing over detail most people do not
    # have to hand.
    if (
        visit_data.insurance_info is not None
        and visit_data.insurance_info.has_insurance is not False
        and visit_data.insurance_info.provider_name is None
    ):
        missing_fields.append("insurance_info.provider_name")

    if visit_data.current_medications:
        for index, medication in enumerate(visit_data.current_medications):
            for field_name in ("dosage", "frequency"):
                if getattr(medication, field_name) is None:
                    missing_fields.append(f"current_medications.{index}.{field_name}")

    if visit_data.allergies:
        for index, allergy in enumerate(visit_data.allergies):
            if allergy.reaction is None:
                missing_fields.append(f"allergies.{index}.reaction")

    if workflow is WorkflowType.APPOINTMENT_PREPARATION and visit_data.address is not None:
        for field_name in ("street", "city", "state", "postal_code"):
            if getattr(visit_data.address, field_name) is None:
                missing_fields.append(f"address.{field_name}")

    return missing_fields


def is_workflow_complete(workflow: WorkflowType, visit_data: VisitData) -> bool:
    return not get_missing_fields(workflow, visit_data)


def get_next_missing_field(workflow: WorkflowType, visit_data: VisitData) -> str | None:
    missing_fields = get_missing_fields(workflow, visit_data)
    return missing_fields[0] if missing_fields else None


def get_question_for_field(workflow: WorkflowType, field_name: str) -> str | None:
    return get_workflow_schema(workflow).question_by_field.get(field_name)


def refresh_state_completeness(state: ConversationState) -> list[str]:
    """Recalculate and store missing fields for the state's selected workflow.

    A clarification is asked once. If the patient answers something else — or
    simply does not know which leg — the coarse value stands and the
    conversation moves on, because re-asking the same unanswered question
    forever is worse than an imprecise record.
    """

    if state.workflow is None:
        state.missing_fields = []
    else:
        already_asked = (
            frozenset({state.requested_field}) if state.requested_field else frozenset()
        )
        state.missing_fields = get_missing_fields(
            state.workflow, state.visit_data, already_asked
        )
    return state.missing_fields
