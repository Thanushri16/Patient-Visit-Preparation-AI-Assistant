# Software Requirements Specification (SRS)

# Project: AI Patient Visit Preparation Assistant

> **Purpose**
>
> Build an AI-powered healthcare chatbot that helps patients prepare for a doctor's appointment by collecting relevant information, organizing it into a structured summary, and improving communication between patients and healthcare providers.
>
> **Note:** This application is intended for educational and appointment preparation purposes only. It is **not** designed to diagnose, treat, or replace medical professionals.

## Development Approach and Iterations

This project will be delivered in three development iterations:

1. **Iteration 1 — Simple Chatbot MVP**
   - Build a simple chatbot experience with a basic text input box or chat window for user interaction.
   - Support essential conversation flow for collecting appointment-preparation information.
   - Focus on core usability, basic safety messaging, and a simple visit summary output.

   **STAR Summary of Achievement**
   - **Situation:** The project required a lightweight MVP that could support appointment-preparation conversations without introducing unnecessary architectural complexity.
   - **Task:** Deliver an intuitive chatbot experience capable of collecting patient-relevant information, maintaining conversational context, and generating a structured visit summary.
   - **Action:** A FastAPI-based web interface was developed to provide a simple chat UI, while backend logic was implemented to manage conversational state through session-based memory. The solution incorporated technical concepts such as in-memory session persistence, expiration-based session lifecycle management, and structured data extraction for summary generation. The prompting strategy was designed to guide users step-by-step, reduce ambiguity, and preserve context across turns. From a UI/design perspective, this resulted in a minimal, low-friction interface that prioritized clarity, conversational flow, and readable interaction over complex form-based input.
   - **Result:** The MVP demonstrated a technically sound approach to stateful interaction by combining session handling, persistence of conversational context, and structured output generation. This improved usability for end users while establishing a scalable foundation for future enhancements such as persistent storage, authentication, and more advanced data management.

2. **Iteration 2 — AI/API Capability Expansion**
   - Build a more robust AI-driven conversation experience using the concepts introduced in the recent course.
   - Strengthen the chatbot’s ability to understand user intent, follow safety rules, coordinate multi-step workflows, and produce structured outputs.

   **Course-Based Feature Requirements**

   **1. System Prompt → Chat Endpoint**
   - The system shall expose a chat endpoint that accepts a user message and returns a conversational response aligned to the chatbot’s role and safety constraints.
   - The endpoint shall support context-aware interaction so the assistant can respond appropriately across multiple turns.
   - **Basic skills learning:** API route design, request/response handling, and prompt injection basics such as stronger system prompts and input/output block lists. *[Status: Implementation Complete]*
   - **Advanced skills learning:** Intent classification for detecting prompt-injection attempts. *[Status: Implementation Complete]*

   **2. Classification → Intent Classifier**
   - The system shall classify incoming user messages into relevant intent categories such as greeting, symptom reporting, medication inquiry, allergy reporting, emergency concern, or summary request.
   - The classifier shall support routing to the appropriate conversational flow based on the detected intent.
   - **Basic skills learning:** Text classification, label definition, simple intent mapping, and confidence scoring. *[Status: Implementation Complete]*
   - **Advanced skills learning:** Few-shot classification, dynamic routing logic, fallback handling for unknown intents, and classifier evaluation metrics. *[Status: Implementation Complete]*

   **3. Moderation → Safety Service**
   - The system shall detect unsafe, inappropriate, or emergency-related content and trigger appropriate safety responses.
   - The system shall ensure that medical advice remains educational and non-diagnostic, in line with the project safety requirements.
   - **Basic skills learning:** Guardrails, safety rules, emergency detection keywords, and response blocking or redirection. *[Status: Implementation Complete]*
   - **Advanced skills learning:** Moderation pipelines, policy-based checks, layered safety validation, and escalation logic for high-risk cases. *[Status: Implementation Complete]*

   **4. Prompt Chaining → AI Workflow**
   - The system shall support a multi-step AI workflow that guides the conversation from intake to clarification to summary generation.
   - The workflow shall ensure that follow-up questions are asked only when necessary and that previously collected information is reused appropriately.
   - The living implementation architecture and status diagram shall be maintained in [`prompt_chaining_architecture.md`](./prompt_chaining_architecture.md).
   - **Basic skills learning:** Prompt chaining, step-by-step workflow design, state tracking across turns, and role-based system prompts for different functionalities instead of relying on a single prompt for everything. *[Status: Implementation Complete]*
   - **Advanced skills learning:** Orchestration logic, conditional branching, workflow state machines, modular prompt design, reusable sub-prompts, and task decomposition. *[Status: Implementation Complete]*

   **Prompt-chaining architectures selected for implementation:**
   - **State Machine Architecture:** Controls explicit conversation phases such as menu selection, data collection, review, confirmation, completion, and emergency escalation.
   - **Router–Worker Architecture:** Classifies the user's intent and routes the request to the appropriate workflow, such as appointment preparation, symptom reporting, allergy reporting, medication questions, summary review, or emergency support.
   - **Schema-Driven Collection:** Defines the required and optional fields for each workflow and uses the schema as the source of truth for completeness.
   - **Extract–Validate–Respond Pattern:** Extracts structured updates from a user message, validates them, and then generates the next appropriate response.
   - **Looping or Iterative Collection Chain:** Repeats extraction, validation, and missing-field checks until the workflow has enough information to produce a summary.
   - **Conditional Branching Chains:** Adds or skips follow-up questions based on information already supplied by the user.
   - **Structured Output Chaining:** Requires prompt nodes to exchange typed, schema-validated data instead of ambiguous free-form text.
   - **Memory and State Separation:** Stores conversation history and application-owned workflow state outside the model so previously collected information can be reused reliably.
   - **Confirmation and Correction Chain:** Presents the collected information for confirmation and supports corrections before the visit summary is finalized.
   - **Guardrail Chain:** Applies input safety checks before workflow processing and output safety checks before returning a response.
   - **Confidence-Based Routing:** Continues automatically for sufficiently reliable classifications and asks for clarification or shows the menu when confidence is low.
   - **Fallback and Recovery Chain:** Uses bounded retries and safe fallback behavior for malformed model output, validation failures, and unsupported requests.

   **5. Structured Output → JSON Visit Summary**
   - The system shall generate a structured JSON summary of the conversation that includes key patient preparation details.
   - The output shall follow a consistent schema and be suitable for storage, review, or future export.
   - **Basic skills learning:** JSON schema design, structured prompting, and parsing responses into dictionaries or objects. *[Status: Implementation Complete]*
   - **Advanced skills learning:** Schema validation, typed models, robust parsing fallback, consistency checks, and error recovery. *[Status: Implementation Complete]*

   **6. Evaluation → Response Quality Checks**
   - The system shall include quality checks for response relevance, completeness, safety, and clarity.
   - The system shall flag responses that fail quality expectations so they can be reviewed or improved before being delivered to the user.
   - **Basic skills learning:** Response review criteria, rubric-based evaluation, and simple pass/fail checks. *[Status: Implementation Complete]*
   - **Advanced skills learning:** Automated evaluation pipelines, scoring heuristics, human-in-the-loop review, and regression testing for prompt quality. *[Status: In use / Partially complete]*

3. **Iteration 3 — Scaled Platform**
   - Expand the solution based on the architecture, security, privacy, scalability, and reliability requirements defined in this document.
   - Add the broader features highlighted in the functional and non-functional requirements, including session management, structured data handling, export capabilities, feedback, and administrative monitoring.
   - Scale the backend, deployment, and observability approach to meet the target usage and availability expectations.

   **Current Implementation Status**

   **Functional Requirements**

   | Requirement | Status | Notes |
   |---|---|---|
   | FR-1 User Authentication | Not complete | There is no login, account creation, or user-owned conversation model yet. |
   | FR-2 Session Management | Partially complete | The chatbot supports in-memory session creation, reuse, and expiration, but it does not yet support authenticated multi-user session management. |
   | FR-3 Patient Intake | Complete | The chatbot collects appointment-preparation, symptom, allergy, and medication information through structured workflows. |
   | FR-4 Conversational Follow-up | Complete | The chatbot asks conditional follow-up questions based on missing or incomplete fields. |
   | FR-5 Conversation Context | Complete | The chatbot keeps workflow state and message memory separate and reuses collected values across turns. |
   | FR-6 Emergency Symptom Detection | Complete | Input moderation detects emergency content and routes the conversation into emergency support. |
   | FR-7 Emergency Escalation | Complete | Emergency cases immediately exit normal intake and return the emergency response. |
   | FR-8 Safety Guardrails | Complete | The system blocks unsafe medical behavior and keeps responses educational and non-diagnostic. |
   | FR-9 Educational Assistance | Partially complete | The medication workflow provides general education and safety framing, but the broader educational assistant scope is not fully built out. |
   | FR-10 Visit Summary Generation | Complete | The system renders a faithful summary from structured visit data and asks for confirmation before completion. |
   | FR-11 Structured Output | Complete | The chatbot stores and validates typed visit data internally for downstream processing and persistence. |
   | FR-12 Conversation History | Complete | Message history is maintained per session in memory. |
   | FR-13 Export Visit Summary | Not complete | The system does not yet export summaries to PDF, text, or downloadable JSON. |
   | FR-14 User Feedback | Not complete | There is no feedback capture flow for user ratings or comments. |
   | FR-15 Administrative Monitoring | Not complete | There is no administrator dashboard or reporting surface. |
   | FR-16 Healthcare Menu Navigation | Complete | The application exposes a healthcare-focused menu for the implemented workflows. |
   | FR-17 Menu-Based Conversation Routing | Complete | Menu options and simple commands route users to the correct workflow. |
   | FR-18 Safety-First Emergency Handling | Complete | Emergency-related turns are prioritized and routed away from routine intake. |

   **Non-Functional Requirements**

   | Requirement | Status | Notes |
   |---|---|---|
   | NFR-1 Performance | Not complete | There is no measured response-time target enforcement yet. |
   | NFR-2 Availability | Not complete | The application is not deployed with an availability target such as 99.5%. |
   | NFR-3 Scalability | Not complete | The current runtime is single-process and in-memory rather than horizontally scalable. |
   | NFR-4 Reliability | Partially complete | The chatbot has bounded retries and safe fallbacks, but it does not yet provide durable recovery across process restarts. |
   | NFR-5 Security | Partially complete | The application has safety guardrails and session validation, but it does not yet include secure authentication, encryption, or full authorization controls. |
   | NFR-6 Privacy | Partially complete | The implementation avoids logging patient values in telemetry and keeps confirmed records local, but it does not yet provide production-grade encryption. |
   | NFR-7 Maintainability | Complete | The implementation is modular and split across routing, extraction, summary, persistence, moderation, and evaluation modules. |
   | NFR-8 Observability | Partially complete | The chatbot emits privacy-safe chain events and evaluation output, but it does not yet expose a full production metrics stack. |
   | NFR-9 Extensibility | Complete | The current prompt chain is modular and designed for future expansion. |
   | NFR-10 Testability | Complete | The repository includes unit tests and deterministic evaluation scripts. |
   | NFR-11 Cost Efficiency | Not complete | There is no cost monitoring or optimization layer yet. |
   | NFR-12 Accessibility | Complete | The UI is a simple browser-based chat interface that works across desktop and mobile viewports. |

# Appendix A — Solution Notes (Implementation Status and Remaining Design Work)

> **This appendix now records the implementation direction already established in the codebase.**
>
> The sections below summarize what has been implemented so far and what remains in the scaled-platform scope.

## System Architecture

Implemented direction:

- FastAPI serves the web UI and `/chat` API.
- `src/app.py` owns typed in-memory session management and session expiry.
- `src/chatbot.py` orchestrates moderation, routing, extraction, confirmation, and persistence.
- `src/routing.py` handles menu commands, workflow selection, emergency continuation, and completion handling.
- `src/extraction.py` performs structured field extraction, validation, and safe merging into visit state.
- `src/summary_workflow.py` renders the faithful summary and classifies confirmation or correction replies.
- `src/persistence.py` writes confirmed summaries locally as versioned JSON records.
- `src/observability.py` emits privacy-safe chain events without logging raw patient values.
- `src/evaluators/` contains deterministic and model-backed evaluation scripts for regression checks.

Remaining platform design work:

- Multi-user authentication and authorization boundaries
- Durable session storage and shared backend state
- Production deployment topology and scaling strategy

---

## Backend Services

Implemented direction:

- Chat service is already implemented in the FastAPI runtime.
- Conversation management is typed and session-scoped.
- Summary generation is implemented as a faithful render of `VisitData`.
- Local persistence is limited to confirmed summaries.
- Safety, moderation, and routing are split into dedicated modules.

Remaining platform design work:

- Authentication service
- Notification or messaging service
- Administrative service
- Export service

---

## Database Design

Implemented direction:

- In-memory session state stores active conversations during runtime.
- Confirmed summaries are persisted locally as JSON files with UUID-based filenames.
- Observability records store anonymized session references instead of raw session IDs.

Remaining platform design work:

- Persistent relational or document storage for users, sessions, visits, and exports
- Retention and archival rules
- Audit logging model
- Encryption at rest design

---

## API Design

Implemented direction:

- `GET /` serves the browser chat UI.
- `POST /chat` accepts a message and a session ID and returns one chatbot reply.
- Session IDs are validated server-side.

Remaining platform design work:

- Authentication APIs
- Session management APIs
- Export APIs
- Feedback APIs
- Administrative APIs

---

## AI Design

Implemented direction:

- Prompt chaining is split into routing, extraction, summary, confirmation, and guardrail stages.
- Conversation memory is separated from workflow state.
- Structured extraction uses typed models and schema validation.
- Confirmation is classified with a dedicated prompt after summary review.
- Input and output guardrails run before and after workflow processing.

Remaining platform design work:

- Higher-coverage evaluation of prompt quality and failure modes
- Model governance for larger-scale deployment
- Potential retrieval or knowledge integration for future features

---

## Security Design

Implemented direction:

- Safety guardrails block or escalate high-risk inputs.
- Session IDs are validated and not logged as raw identifiers in observability events.
- Confirmed visit records are saved locally with restricted file handling.

Remaining platform design work:

- Authentication strategy
- Authorization model
- Encryption at rest and in transit
- Role-based admin access
- Production session hardening

---

## Deployment

Implemented direction:

- The application can run locally with FastAPI and `uv`.
- Tests and evaluators run from the repository for regression checking.

Remaining platform design work:

- Containerization
- Orchestration
- CI/CD
- Environment-specific infrastructure

---

## Monitoring

Implemented direction:

- Privacy-safe chain events are emitted for key workflow stages.
- Evaluators generate reports for prompt-chain and safety regression checks.

Remaining platform design work:

- Centralized logging
- Metrics collection
- Alerting
- Distributed tracing
- Admin reporting dashboards

---

## Testing Strategy

Implemented direction:

- Unit tests cover routing, extraction, summary workflow, persistence, observability, session state, and models.
- Integration-style tests cover the chatbot flow and deterministic chain evaluation.
- Prompt-injection and intent-classifier evaluators support safety and routing regression checks.

Remaining platform design work:

- Performance testing at target scale
- End-to-end tests against a deployed environment
- Load and availability testing

---

## Future Enhancements

Examples include:

- Knowledge retrieval
- Clinical document ingestion
- Voice conversations
- Appointment scheduling
- Provider dashboard
- Multi-language support
- AI-powered symptom trend analysis
- Wearable device integrations
