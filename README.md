# AI Patient Visit Preparation Assistant

An AI-powered healthcare chatbot that helps patients prepare for a clinical appointment. It collects relevant information through a guided conversation, validates and organizes the answers, asks only necessary follow-up questions, and produces a structured visit summary for patient review and confirmation.

The application includes intent-based routing, explicit workflow state, typed structured extraction, emergency escalation, medical-safety guardrails, correction handling, bounded recovery, privacy-safe telemetry, and confirmation-only local persistence.

This project is for educational and appointment-preparation purposes. It does not diagnose conditions, prescribe medication, recommend stopping prescribed medication, or replace care from a licensed medical professional.

## Current scope

The implemented prompt chain supports:

- Appointment preparation and patient intake
- Symptom, allergy, and medication-information workflows
- Healthcare menu and intent-based routing
- Required identity and contact details for appointment preparation
- Conditional questions for address, insurance, and allergy reactions
- Emergency symptom detection and escalation
- Structured summary review, correction, and confirmation
- Local storage of confirmed summaries under UUID visit IDs

See [the SRS](documentation/ai-chatbot-for-healthcare-srs.md) for full product requirements and future platform scope. See [the prompt-chaining architecture](documentation/prompt_chaining_architecture.md) for the runtime diagram and implementation tracker.

The current codebase covers the MVP and AI workflow expansion. The SRS now reserves Iteration 3 for the production RAG platform and Iteration 4 for the enterprise production platform.

## Project structure

```text
.
├── documentation/
│   ├── ai-chatbot-for-healthcare-srs.md       # Requirements and delivery iterations
│   ├── prompt_chaining_architecture.md        # Implemented architecture and status
│   └── prompt_chaining_architecture_concepts.md
├── src/
│   ├── app.py                                 # FastAPI application, web UI, and sessions
│   ├── chatbot.py                             # Prompt-chain orchestration
│   ├── chatbot_content.py                     # Menu and safety constants
│   ├── models.py                              # Typed visit, message, and state models
│   ├── routing.py                             # State-aware workflow routing
│   ├── workflow_catalog.py                    # Menu and intent metadata source of truth
│   ├── workflow_schemas.py                    # Required fields and completeness rules
│   ├── extraction.py                          # Structured extraction and validation
│   ├── questions.py                           # Deterministic next-question selection
│   ├── summary_workflow.py                    # Summary and confirmation workflow
│   ├── moderation.py                          # Input/output safety checks
│   ├── persistence.py                         # Confirmed-visit JSON persistence
│   ├── observability.py                       # Privacy-safe prompt-chain events
│   ├── prompts/                               # Model-backed extractor and confirmation prompts
│   └── evaluators/                            # Regression and Excel-driven benchmark evaluations
│       └── benchmarks/                        # Loader, API runner, 3-layer scorer, reports, and CLI
├── tests/                                     # Unit and workflow tests
├── reports/                                   # Generated evaluation reports
├── db/visits/                                 # Runtime-only confirmed visit records
├── .env.example                               # API-key template
├── pyproject.toml                             # Python project and dependencies
└── uv.lock                                    # Locked dependency versions
```

The `db/visits/` directory is created when a confirmed summary is saved and is ignored by Git.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for intent classification, structured extraction, and ambiguous confirmation classification

## Setup

From the repository root, install the locked dependencies:

```bash
uv sync
```

Create the local environment file:

```bash
cp .env.example .env
```

Then replace the placeholder in `.env`:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit `.env` or real API keys.

## Run the chatbot

Start the FastAPI development server from the repository root:

```bash
uv run uvicorn src.app:app --reload
```

Open <http://127.0.0.1:8000> in a browser. Stop the server with `Ctrl+C`.

Confirmed summaries are stored locally in `db/visits/`. They are not submitted to a provider or external healthcare system.

## Run tests

Run the complete test suite:

```bash
uv run python -m unittest discover -s tests -v
```

Run one test module, for example:

```bash
uv run python -m unittest tests.test_routing -v
```

Run the deterministic prompt-chain regression evaluation:

```bash
uv run python src/evaluators/prompt_chain_evaluator.py
```

Run the full iteration-2 evaluation pipeline:

```bash
uv run python src/evaluators/iteration_2_evaluation_pipeline.py
```

The prompt-injection and intent-classifier evaluations use the configured OpenAI API and may incur API usage:

```bash
uv run python src/evaluators/prompt_injection_evaluator.py
uv run python src/evaluators/intent_classifier_evaluator.py
```

Evaluation reports are written to `reports/` by default.

### Run the 210-scenario API benchmark

Start the chatbot API in one terminal, then run the benchmark in another:

```bash
uv run python -m src.evaluators.benchmarks.run_benchmarks \
  --file src/evaluators/healthcare_assistant_benchmark_210.xlsx
```

The benchmark runs deterministic contract and state checks plus an LLM-as-judge
evaluation. It writes full, summary, and failure-only JSON reports under
`reports/benchmarks/`. Useful options include:

```bash
# Run one category
uv run python -m src.evaluators.benchmarks.run_benchmarks \
  --category "Emergency detection and escalation"

# Run selected scenarios without paid judge calls
uv run python -m src.evaluators.benchmarks.run_benchmarks \
  --test-id TC-001 --test-id TC-012 --no-judge

# Target another deployment
uv run python -m src.evaluators.benchmarks.run_benchmarks \
  --base-url http://staging:8000 --output-dir reports/staging-benchmark
```

## Data and deployment notes

- Session state is held in memory and expires after 15 minutes.
- Only completed, explicitly confirmed summaries are persisted.
- Local JSON records use UUID filenames and restricted file permissions.
- Local records are not encrypted; authentication, production encryption, export formats, durable multi-user sessions, RAG knowledge management, and administrative monitoring belong to later iterations in the SRS.
