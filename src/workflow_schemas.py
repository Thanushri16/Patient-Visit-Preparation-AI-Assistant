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
    "insurance_info",
    "lifestyle_info",
    "emergency_symptoms",
    "notes",
)


WORKFLOW_SCHEMAS: dict[WorkflowType, WorkflowSchema] = {
    WorkflowType.APPOINTMENT_PREPARATION: WorkflowSchema(
        workflow=WorkflowType.APPOINTMENT_PREPARATION,
        required_fields=(
            "chief_complaint",
            "symptom_duration",
            "symptom_severity",
            "medical_conditions",
            "current_medications",
            "allergies",
            "patient_name",
            "date_of_birth",
            "email",
            "phone",
        ),
        optional_fields=INTAKE_OPTIONAL_FIELDS,
        question_by_field={
            "chief_complaint": "What is the main concern you want to discuss with your clinician?",
            "symptom_duration": "How long have you been experiencing this concern?",
            "symptom_severity": "On a scale from 0 to 10, how severe is it?",
            "medical_conditions": "Do you have any existing medical conditions? You can say none.",
            "current_medications": "What medications are you currently taking? You can say none.",
            "allergies": "Do you have any medication or other allergies? You can say none.",
            "patient_name": "What name would you like included in the appointment summary?",
            "date_of_birth": "What is your date of birth?",
            "email": "What email address should be included in your visit information?",
            "phone": "What phone number should be included in your visit information?",
        },
    ),
    WorkflowType.REPORT_NEW_SYMPTOMS: WorkflowSchema(
        workflow=WorkflowType.REPORT_NEW_SYMPTOMS,
        required_fields=("chief_complaint", "symptom_duration", "symptom_severity"),
        optional_fields=("emergency_symptoms", "notes"),
        question_by_field={
            "chief_complaint": "What symptoms are you experiencing?",
            "symptom_duration": "How long have you had these symptoms?",
            "symptom_severity": "On a scale from 0 to 10, how severe are the symptoms?",
        },
    ),
    WorkflowType.REPORT_ALLERGY: WorkflowSchema(
        workflow=WorkflowType.REPORT_ALLERGY,
        required_fields=("allergies",),
        optional_fields=("emergency_symptoms", "notes"),
        question_by_field={
            "allergies": "What are you allergic to, and what reaction do you experience?",
        },
    ),
    WorkflowType.MEDICATION_QUESTION: WorkflowSchema(
        workflow=WorkflowType.MEDICATION_QUESTION,
        required_fields=("current_medications",),
        optional_fields=("allergies", "medical_conditions", "notes"),
        question_by_field={
            "current_medications": "Which medication would you like to discuss?",
        },
    ),
    WorkflowType.REVIEW_HEALTH_NOTES: WorkflowSchema(
        workflow=WorkflowType.REVIEW_HEALTH_NOTES,
        optional_fields=tuple(VisitData.model_fields),
    ),
    WorkflowType.VIEW_SUMMARY: WorkflowSchema(
        workflow=WorkflowType.VIEW_SUMMARY,
        optional_fields=tuple(VisitData.model_fields),
    ),
    WorkflowType.EMERGENCY_SUPPORT: WorkflowSchema(
        workflow=WorkflowType.EMERGENCY_SUPPORT,
        optional_fields=("emergency_symptoms", "notes"),
    ),
}


def get_workflow_schema(workflow: WorkflowType) -> WorkflowSchema:
    return WORKFLOW_SCHEMAS[workflow]


def get_missing_fields(workflow: WorkflowType, visit_data: VisitData) -> list[str]:
    """Return unanswered required fields in their configured collection order."""

    schema = get_workflow_schema(workflow)
    missing_fields = [
        field_name
        for field_name in schema.required_fields
        if getattr(visit_data, field_name) is None
    ]
    missing_fields.extend(get_conditional_missing_fields(workflow, visit_data))
    return missing_fields


def get_conditional_missing_fields(
    workflow: WorkflowType,
    visit_data: VisitData,
) -> list[str]:
    """Return nested fields required only after the user starts an optional section."""

    missing_fields: list[str] = []

    if workflow is WorkflowType.APPOINTMENT_PREPARATION and visit_data.address is not None:
        for field_name in ("street", "city", "state", "postal_code"):
            if getattr(visit_data.address, field_name) is None:
                missing_fields.append(f"address.{field_name}")

    if workflow is WorkflowType.APPOINTMENT_PREPARATION and visit_data.insurance_info is not None:
        for field_name in ("provider_name", "policy_number"):
            if getattr(visit_data.insurance_info, field_name) is None:
                missing_fields.append(f"insurance_info.{field_name}")

    if workflow in {
        WorkflowType.APPOINTMENT_PREPARATION,
        WorkflowType.REPORT_ALLERGY,
    } and visit_data.allergies:
        for index, allergy in enumerate(visit_data.allergies):
            if allergy.reaction is None:
                missing_fields.append(f"allergies.{index}.reaction")

    return missing_fields


def is_workflow_complete(workflow: WorkflowType, visit_data: VisitData) -> bool:
    return not get_missing_fields(workflow, visit_data)


def get_next_missing_field(workflow: WorkflowType, visit_data: VisitData) -> str | None:
    missing_fields = get_missing_fields(workflow, visit_data)
    return missing_fields[0] if missing_fields else None


def get_question_for_field(workflow: WorkflowType, field_name: str) -> str | None:
    return get_workflow_schema(workflow).question_by_field.get(field_name)


def refresh_state_completeness(state: ConversationState) -> list[str]:
    """Recalculate and store missing fields for the state's selected workflow."""

    if state.workflow is None:
        state.missing_fields = []
    else:
        state.missing_fields = get_missing_fields(state.workflow, state.visit_data)
    return state.missing_fields
