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

---

# 1. Functional Requirements

## FR-1 User Authentication

The system shall allow users to securely create an account, log in, and access their own conversations.

---

## FR-2 Session Management

The system shall allow users to create, resume, and manage multiple chat sessions.

---

## FR-3 Patient Intake

The chatbot shall collect relevant patient information, including but not limited to:

- Chief complaint
- Symptom duration
- Symptom severity
- Existing medical conditions
- Current medications
- Allergies
- Lifestyle information (optional)

---

## FR-4 Conversational Follow-up

The chatbot shall dynamically ask follow-up questions based on previous user responses to gather sufficient context.

---

## FR-5 Conversation Context

The chatbot shall maintain context throughout the conversation and reference previously collected information when generating responses.

---

## FR-6 Emergency Symptom Detection

The chatbot shall detect potentially life-threatening symptoms during the conversation.

Examples include:

- Chest pain
- Difficulty breathing
- Stroke-like symptoms
- Severe allergic reactions
- Severe bleeding
- Loss of consciousness

---

## FR-7 Emergency Escalation

If emergency symptoms are detected, the chatbot shall immediately recommend seeking emergency medical attention and discontinue normal conversational flow.

---

## FR-8 Safety Guardrails

The chatbot shall not:

- Diagnose diseases
- Prescribe medication
- Recommend stopping prescribed medication
- Replace professional medical advice

---

## FR-9 Educational Assistance

The chatbot shall provide general health education and encourage users to discuss concerns with licensed healthcare professionals.

---

## FR-10 Visit Summary Generation

The chatbot shall generate a structured summary that can be shared with a healthcare provider before the appointment.

---

## FR-11 Structured Output

The chatbot shall internally organize collected information into structured data suitable for downstream processing.

---

## FR-12 Conversation History

The system shall maintain conversation history for each user session.

---

## FR-13 Export Visit Summary

The system shall allow users to export their visit summary in multiple formats.

Examples:

- PDF
- Plain text
- JSON

---

## FR-14 User Feedback

The system shall allow users to submit feedback regarding chatbot responses.

---

## FR-15 Administrative Monitoring

The system shall allow administrators to review anonymized platform metrics, chatbot usage statistics, and reported issues.

---

## FR-16 Healthcare Menu Navigation

The system shall provide a simple healthcare-focused menu of options for appointment preparation, symptom reporting, allergy reporting, medication questions, summary review, emergency support, and general health education.

---

## FR-17 Menu-Based Conversation Routing

The system shall allow users to select a menu option using simple text commands and route them to the appropriate healthcare conversation flow.

---

## FR-18 Safety-First Emergency Handling

If a menu option relates to urgent or emergency symptoms, the system shall prioritize safety guidance, recommend immediate emergency care, and redirect the conversation away from routine intake.

---

# 2. Non-Functional Requirements

## NFR-1 Performance

The chatbot should respond to normal user requests within **5 seconds** under expected load.

---

## NFR-2 Availability

The platform should target an availability of **99.5%**.

---

## NFR-3 Scalability

The platform shall support horizontal scaling of backend services.

---

## NFR-4 Reliability

The system shall gracefully recover from transient failures without data loss.

---

## NFR-5 Security

The platform shall:

- Secure user authentication
- Encrypt sensitive data
- Protect user sessions
- Prevent unauthorized access

---

## NFR-6 Privacy

The platform shall minimize collection of sensitive information and ensure user conversations remain private.

---

## NFR-7 Maintainability

The application shall be modular and support independent development of major components.

---

## NFR-8 Observability

The system shall capture logs, metrics, and traces necessary for monitoring platform health.

---

## NFR-9 Extensibility

The architecture shall allow future integration of additional AI capabilities without major redesign.

---

## NFR-10 Testability

The platform shall support:

- Unit testing
- Integration testing
- End-to-end testing

---

## NFR-11 Cost Efficiency

The system shall monitor AI usage costs and optimize resource utilization where appropriate.

---

## NFR-12 Accessibility

The chatbot interface should be usable across desktop and mobile devices.

---

# 3. Expected Scale

## MVP

- 100–500 registered users
- 1,000 chatbot conversations per day
- Average conversation length: 15–25 messages
- Single geographic deployment

---

## Target Scale

- 10,000 registered users
- 100,000 chatbot messages per day
- Thousands of concurrent conversations
- Horizontally scalable backend services
- Support for future multi-region deployment

---

# Appendix A — Solution Notes (For Future Design)

> **This appendix is intentionally non-prescriptive.**
>
> The following topics are placeholders for implementation decisions that will be designed later.

## System Architecture

_To be designed._

Possible considerations:

- High-level architecture
- Service boundaries
- Request flow
- Deployment topology

---

## Backend Services

_To be designed._

Possible considerations:

- Authentication
- Chat service
- Conversation management
- Summary generation
- Notification service

---

## Database Design

_To be designed._

Possible considerations:

- Entity relationships
- Table design
- Data retention
- Audit logging

---

## API Design

_To be designed._

Possible considerations:

- Authentication APIs
- Chat APIs
- Session APIs
- Export APIs
- Administrative APIs

---

## AI Design

_To be designed._

Possible considerations:

- Prompt design
- Conversation memory
- Safety validation
- Structured output generation
- Guardrail implementation

---

## Security Design

_To be designed._

Possible considerations:

- Authentication strategy
- Authorization model
- Encryption
- Session management

---

## Deployment

_To be designed._

Possible considerations:

- Containerization
- Orchestration
- CI/CD
- Infrastructure

---

## Monitoring

_To be designed._

Possible considerations:

- Logging
- Metrics
- Alerting
- Tracing

---

## Testing Strategy

_To be designed._

Possible considerations:

- Unit tests
- Integration tests
- API tests
- End-to-end tests
- Performance testing

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
