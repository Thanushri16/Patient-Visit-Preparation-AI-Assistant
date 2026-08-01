# Software Requirements Specification (SRS)

# Project: AI Patient Visit Preparation Assistant

> **Purpose**
>
> Build an AI-powered healthcare application that helps patients prepare for a doctor's appointment by collecting structured pre-visit information, answering appointment-preparation questions using Retrieval-Augmented Generation (RAG), and generating a provider-ready visit summary.
>
> **Note:** This application is intended for educational and appointment-preparation purposes only. It does not diagnose, prescribe treatment, or replace licensed healthcare professionals.

## Development Approach and Iterations

This project will be delivered in four development iterations:

1. **Iteration 1 — Simple Chatbot MVP**
   - Build a simple chatbot experience with a basic text input box or chat window for user interaction.
   - Support essential conversation flow for collecting appointment-preparation information.
   - Focus on core usability, basic safety messaging, and a simple visit summary output.

   **STAR Summary of Achievement**
   - **Situation:** The project required a lightweight MVP that could support appointment-preparation conversations without introducing unnecessary architectural complexity.
   - **Task:** Deliver an intuitive chatbot experience capable of collecting patient-relevant information, maintaining conversational context, and generating a structured visit summary.
   - **Action:** A FastAPI-based web interface was developed to provide a simple chat UI, while backend logic was implemented to manage conversational state through session-based memory. The solution incorporated technical concepts such as in-memory session persistence, expiration-based session lifecycle management, and structured data extraction for summary generation. The prompting strategy was designed to guide users step-by-step, reduce ambiguity, and preserve context across turns.
   - **Result:** The MVP demonstrated a technically sound approach to stateful interaction by combining session handling, persistence of conversational context, and structured output generation.

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
   - **Advanced skills learning:** Automated evaluation pipelines, scoring heuristics, human-in-the-loop review, and regression testing for prompt quality. *[Status: Implementation Complete]*

3. **Iteration 3 — Production RAG Platform**
   - Transform the chatbot into a production-ready pre-visit preparation assistant powered by Retrieval-Augmented Generation (RAG).
   - Add clinic knowledge ingestion, vector search, grounded answers with citations, and an admin portal for managing knowledge sources.
   - Introduce persistent sessions and editable visit summaries so users can resume conversations safely.

   **Major Features**

   - Patient experience: user authentication, persistent conversations, resume previous sessions, editable visit summaries, and export summaries in PDF, JSON, and text formats.
   - Retrieval-Augmented Generation: clinic knowledge base ingestion, document chunking, embedding generation, vector search, hybrid retrieval, metadata filtering, query rewriting, citation-backed answers, and an "I don't know" fallback when evidence is insufficient.
   - Knowledge sources: appointment preparation guides, clinic FAQs, medication preparation guidance, insurance instructions, telehealth instructions, required documents, and accessibility information.
   - Admin portal: upload knowledge documents, re-index the vector database, manage document versions, and administer the knowledge base.
   - Backend: PostgreSQL, pgvector, Redis, FastAPI, and background indexing workers.
   - Evaluation: retrieval recall@K, citation correctness, faithfulness, groundedness, hallucination detection, and prompt regression tests.

   **Current implementation status**

   **Functional requirements**

   | Requirement | Status | Notes |
   |---|---|---|
   | FR-1 Authentication | Not complete | There is no account creation, login, or authenticated user session model yet. |
   | FR-2 Session Management | Partially complete | The app supports typed in-memory sessions, but not persistent authenticated session resumption. |
   | FR-3 Patient Intake | Complete | Structured pre-visit information is already collected through the current intake workflows. |
   | FR-4 Guided Follow-up | Complete | The chatbot asks only the follow-up questions needed to fill missing information, ordered clinical detail first, with a vaguely answered field clarified ahead of an unaddressed one. |
   | FR-5 Conversation Context | Partially complete | Workflow state and message memory are preserved in memory, but not across durable resumed sessions. |
   | FR-6 Emergency Detection | Complete | Emergency symptoms are detected and routed away from normal intake, with escalation guidance specific to the emergency and past, already-treated events not escalated. |
   | FR-7 Safety | Complete | The chatbot already avoids diagnosis and prescription behavior. |
   | FR-8 Educational Assistance | Partially complete | The chatbot answers common preparation questions — documents to bring, fasting, pre-visit forms, telehealth, transportation, accessibility — from curated non-prescriptive content, and declines out-of-scope requests. These answers are not yet citation-backed clinic-document answers. |
   | FR-9 RAG | Not complete | Retrieval, vector search, citations, and grounded response generation are not implemented yet. |
   | FR-10 Knowledge Management | Not complete | There is no knowledge upload, versioning, or admin document portal yet. |
   | FR-11 Visit Summary | Partially complete | Structured summaries already exist, but editable summary workflows are not implemented. |
   | FR-12 Export | Not complete | PDF, JSON, and text export are not implemented yet. |
   | FR-13 Feedback | Not complete | User feedback collection is not implemented yet. |
   | FR-14 Administrative Monitoring | Not complete | Admin dashboards are not implemented yet. |
   | FR-15 Analytics | Not complete | Retrieval and conversation analytics are not implemented yet. |

   **Non-functional requirements**

   | Requirement | Status | Notes |
   |---|---|---|
   | NFR-1 Performance | Not complete | There is no measured P95 latency target or production load testing yet. |
   | NFR-2 Availability | Not complete | The system is not deployed to a 99.5% availability target. |
   | NFR-3 Scalability | Not complete | The current runtime is still single-process and in-memory. |
   | NFR-4 Reliability | Partially complete | The code has bounded retries and fallbacks, but no durable recovery layer. |
   | NFR-5 Security | Partially complete | Guardrails and input validation exist, but authentication, authorization, and encryption are not in place. |
   | NFR-6 Privacy | Partially complete | Telemetry avoids raw patient values, but production privacy controls are not complete. |
   | NFR-7 Maintainability | Complete | The current codebase is already modular and separated by concern. |
   | NFR-8 Observability | Partially complete | Privacy-safe events and evaluation outputs exist, but not a full production observability stack. |
   | NFR-9 Testability | Partially complete | The repo has an offline deterministic unit suite, deterministic evaluators, and a 210-scenario behavioural benchmark against the running API, but not a full CI-backed automated pipeline. |
   | NFR-10 Cost Efficiency | Not complete | AI token usage monitoring is not implemented yet. |
   | NFR-11 Accessibility | Complete | The UI is already responsive across desktop and mobile viewports. |
   | NFR-12 Extensibility | Complete | The architecture already supports future AI capability expansion. |

4. **Iteration 4 — Enterprise Production Platform**
   - Prepare the platform for production deployment, reliability, observability, scalability, and real-world validation.
   - Add the operational controls needed for secure, scalable, and maintainable deployment.

   **Major Features**

   - Security: RBAC, JWT authentication, encryption, secrets management, and rate limiting.
   - Reliability: retry policies, timeouts, circuit breakers, idempotency, and graceful degradation.
   - Observability: OpenTelemetry, distributed tracing, metrics, logging, dashboards, and alerts.
   - Deployment: Docker, CI/CD, Kubernetes or Azure Container Apps, and infrastructure as code.
   - Testing: unit tests, integration tests, end-to-end tests, load testing, security testing, and the RAG evaluation pipeline.
   - Validation: user studies, performance benchmarks, latency measurements, cost analysis, and retrieval quality reports.

   **Current implementation status**

   **Functional requirements**

   | Requirement | Status | Notes |
   |---|---|---|
   | FR-1 Authentication | Not complete | RBAC, JWT, and secure login are not implemented yet. |
   | FR-2 Session Management | Partially complete | Sessions exist in memory, but not as durable authenticated sessions. |
   | FR-10 Knowledge Management | Not complete | No admin portal exists for document upload, versioning, or re-indexing. |
   | FR-11 Visit Summary | Partially complete | Summaries are generated, but editable summary workflows are not present. |
   | FR-12 Export | Not complete | Export formats are not implemented yet. |
   | FR-13 Feedback | Not complete | Feedback capture is not implemented yet. |
   | FR-14 Administrative Monitoring | Not complete | Admin dashboards are not implemented yet. |
   | FR-15 Analytics | Not complete | Retrieval and operational analytics are not implemented yet. |

   **Non-functional requirements**

   | Requirement | Status | Notes |
   |---|---|---|
   | NFR-1 Performance | Not complete | No production performance target or load testing is in place. |
   | NFR-2 Availability | Not complete | No availability target deployment exists yet. |
   | NFR-3 Scalability | Not complete | No horizontally scalable backend deployment exists yet. |
   | NFR-4 Reliability | Partially complete | Local retries and fallbacks exist, but no durable production recovery. |
   | NFR-5 Security | Partially complete | Basic safety checks exist, but not enterprise security controls. |
   | NFR-6 Privacy | Partially complete | Logging is privacy-conscious, but production privacy hardening is not complete. |
   | NFR-7 Maintainability | Complete | The codebase is already modular enough to support future expansion. |
   | NFR-8 Observability | Partially complete | There are chain events and reports, but not full metrics/tracing/dashboards. |
   | NFR-9 Testability | Partially complete | Unit tests, evaluators, and a 210-scenario API benchmark exist, but not the full CI-backed suite described here. |
   | NFR-10 Cost Efficiency | Not complete | Cost monitoring and token-usage controls are not implemented yet. |
   | NFR-11 Accessibility | Complete | The interface is already usable on desktop and mobile. |
   | NFR-12 Extensibility | Complete | The architecture is intended to support future AI capabilities. |

# Expected Scale

## MVP

- 500 users
- 1,000 conversations per day

## Production Target

- 10,000+ users
- 100,000+ messages per day
- Thousands of concurrent sessions

# Appendix A — Solution Architecture

## Implementation Snapshot

| Iteration | Status | Notes |
|---|---|---|
| Iteration 1 | Complete | Conversation workflow, typed visit summary, in-memory session handling, and basic safety messaging are already implemented. |
| Iteration 2 | Complete | Intent routing, moderation, prompt chaining, structured extraction, summary review, confirmation, persistence, and evaluation are implemented. Behaviour is measured against a 210-scenario benchmark whose findings have been fed back into the conversation design. |
| Iteration 3 | Not complete | The codebase has foundations, but RAG ingestion, citations, persistent auth, exports, and admin knowledge management are still planned. |
| Iteration 4 | Not complete | Production deployment, security hardening, observability, autoscaling, and real-world validation are still planned. |

## Iteration 1

- Implemented: conversation workflow, typed visit summary, in-memory session handling, and basic safety messaging.

## Iteration 2

- Implemented: intent routing, moderation, prompt chaining, structured extraction, summary review, confirmation, persistence, and evaluation.
- Implemented alongside intake: educational answers to general preparation questions, acknowledgement of expressed worry, greetings and farewells, declining out-of-scope requests, and answering questions about what has already been recorded from state rather than from the model.
- Emergency handling returns guidance specific to the emergency — an EpiPen for anaphylaxis, Poison Control for an overdose, the 988 crisis line for self-harm, FAST for stroke — and treats a frightening symptom the patient describes as past and already treated as history rather than escalating it.
- Behaviour is measured end to end by a 210-scenario Excel-driven benchmark against the running API, described under Testing Strategy.

## Iteration 3

- Planned: authentication, PostgreSQL, pgvector, Redis, RAG ingestion pipeline, hybrid retrieval, citations, admin document portal, persistent sessions, exports, and retrieval evaluation.
- Partial foundation already present in the codebase: typed in-memory sessions, structured summaries, local persistence of confirmed summaries, modular routing, and privacy-safe observability.

## Iteration 4

- Planned: production deployment, CI/CD, Kubernetes, OpenTelemetry, monitoring, autoscaling, load testing, security hardening, disaster recovery, and real-user validation.
- Partial foundation already present in the codebase: modular services, unit tests, deterministic evaluation scripts, and simple local deployment support.

## Design Choices

### System Architecture

- Iteration 1 established the core conversation workflow and typed summary handling.
- Iteration 2 established prompt chaining, state-aware routing, moderation, extraction, confirmation, and local persistence.
- Iteration 3 will add RAG ingestion and retrieval paths without replacing the existing prompt-chain foundation.
- Iteration 4 will add production hardening, scale, and operational controls around the same modular application boundary.

### Backend Services

- The current backend is a FastAPI application with typed session handling and local conversation memory.
- RAG-era backend services are expected to add PostgreSQL, Redis, and background indexing workers.
- The later platform stage is expected to split operational concerns such as admin tools, exports, and analytics into dedicated services or modules.

### Database Design

- The current design uses in-memory session state and local JSON persistence for confirmed summaries.
- Iteration 3 is expected to introduce durable storage for users, sessions, knowledge documents, and vector embeddings.
- Iteration 4 will extend storage design for operational data, audits, and recovery workflows.

### API Design

- The current API surface is intentionally small: the browser UI and a single `/chat` endpoint.
- Iteration 3 is expected to add APIs for authentication, session resume, export, knowledge management, and administrative operations.
- Iteration 4 is expected to add operational and monitoring endpoints as needed for the production platform.

### AI Design

- The current AI design is a prompt chain with routing, extraction, summary review, confirmation, and guardrails.
- Model-backed nodes are deliberately narrow. The classifier chooses a workflow, the extractor pulls stated fields into a typed schema, the follow-up generator phrases one question, and the confirmation classifier reads a yes or a correction. Field ordering, validation, completeness, summary rendering, and every safety decision are deterministic application logic, so behaviour is reproducible and auditable rather than dependent on a model's judgement.
- A message that selects a workflow is also treated as content for that workflow, so information the patient volunteers in their opening sentence is recorded instead of being re-requested.
- Clinical detail is collected before administrative detail, and a field the patient answered vaguely is clarified before fields they have not addressed at all.
- Iteration 3 adds RAG: document ingestion, chunking, embeddings, vector retrieval, citation-backed answers, and grounded fallback behavior.
- Iteration 4 focuses on the reliability, evaluation, and governance controls for production AI behavior.

### Security Design

- The current codebase uses guardrails, validation, and privacy-safe observability.
- Iteration 3 adds authentication, authorization, and stronger session handling for persistent user state.
- Iteration 4 adds encryption, secrets management, RBAC, and rate limiting for the enterprise platform.

### Deployment

- The current application runs locally with FastAPI and `uv`.
- Iteration 3 is expected to introduce durable backend services and worker processes for indexing and retrieval.
- Iteration 4 is expected to introduce containerization, CI/CD, and production deployment infrastructure.

### Monitoring

- The current implementation emits privacy-safe chain events and evaluator output.
- Benchmark runs additionally report how hard the suite pushed against the model provider — attempts, rate-limit hits, retries, permanently failed cases, concurrency adjustments, and total time spent backing off — which is the operational signal for whether throughput is limited locally or upstream.
- Iteration 3 is expected to add retrieval metrics, citation quality checks, and knowledge-base reporting.
- Iteration 4 is expected to add dashboards, tracing, metrics, and alerting.

### Testing Strategy

- The current repository supports offline unit tests, deterministic evaluation scripts, and a 210-scenario behavioural benchmark driven from an Excel workbook.
- The unit suite fakes every model client, so it is deterministic and makes no paid API calls. It covers workflow schemas and completeness, routing, extraction and merging, question selection, summary and confirmation, safety guardrails, and the benchmark runner's own rate-limit and checkpoint behaviour.
- The benchmark scores each scenario in three layers: deterministic response-contract checks, concrete conversation-state checks, and an LLM-as-judge assessment of intent handling, behaviour, criteria, safety, and tone. A scenario passes only when every applicable layer passes.
- The benchmark is rate-limit aware by construction: parallelism is discovered at runtime by an AIMD governor rather than fixed, every call retries throttling with exponential backoff and full jitter, results are checkpointed per batch so an interrupted run resumes, and a case that exhausts its retry budget is recorded as an error and skipped rather than failing the run.
- Iteration 3 adds RAG-specific evaluation such as recall, groundedness, faithfulness, and citation checks.
- Iteration 4 adds load, security, and end-to-end validation against production-like infrastructure.

### Future Enhancements

- Knowledge retrieval
- Clinical document ingestion
- Voice conversations
- Appointment scheduling
- Provider dashboard
- Multi-language support
- AI-powered symptom trend analysis
- Wearable device integrations
