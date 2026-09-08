# AI Patient Visit Preparation Assistant

An AI-powered healthcare chatbot that helps patients prepare for a clinical appointment. It collects relevant information through a guided conversation, validates and organizes the answers, asks only necessary follow-up questions, and produces a structured visit summary for patient review and confirmation.

The application includes intent-based routing, explicit workflow state, typed structured extraction, emergency escalation, medical-safety guardrails, correction handling, bounded recovery, privacy-safe telemetry, and confirmation-only local persistence.

This project is for educational and appointment-preparation purposes. It does not diagnose conditions, prescribe medication, recommend stopping prescribed medication, or replace care from a licensed medical professional.

## Current scope

The implemented prompt chain supports:

- Appointment preparation and patient intake, including visit reason, provider,
  date and time, insurance, documents, accessibility needs, and pre-visit
  instructions
- Symptom, allergy, and medication-information workflows
- Healthcare menu and intent-based routing, where a message that selects a
  workflow is also extracted rather than discarded
- Clinical detail collected before administrative detail, so a reported symptom
  is followed up on before contact information is requested
- Conditional questions for medication doses, allergy reactions, insurance
  details, and address
- Emergency detection with guidance specific to the emergency, and past,
  resolved events treated as history rather than escalated
- Educational answers to general preparation questions, acknowledgement of
  expressed worry, and answers to questions about what has already been recorded
- Structured summary review, correction, and confirmation, in prose for reading
  and as a schema-complete JSON document when one is requested
- Citation-backed answers about diagnostic tests and procedures — what a test
  is, how to prepare, how it will feel, what the risks are — retrieved from a
  corpus of clinical documents, with an explicit "I don't have documentation on
  that" when the evidence is missing
- A LangGraph orchestrator serving every turn, proven to reproduce the previous
  hand-rolled chain exactly before it became the default; `ORCHESTRATOR=chain`
  switches back in one word
- Refusal to answer from documents where the answer is a clinical judgement:
  whether to stop a medication, what a symptom means, whether a reaction was an
  allergy. These never reach the retriever.
- Local storage of confirmed summaries under UUID visit IDs

See [the SRS](documentation/ai-chatbot-for-healthcare-srs.md) for full product requirements and future platform scope. See [the prompt-chaining architecture](documentation/prompt_chaining_architecture.md) for the runtime diagram and implementation tracker.

The current codebase covers the MVP, the AI workflow expansion, and Part A of
the Iteration 3 RAG platform. See [the RAG architecture](documentation/rag_architecture.md)
for the design, the measured results, and what is deliberately left undone.
Iteration 4 remains the enterprise production platform.

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
│   ├── extraction.py                          # Structured extraction, validation, and merging
│   ├── questions.py                           # Next-question selection and adaptive follow-ups
│   ├── guidance.py                            # Educational answers, empathy, and state recall
│   ├── summary_workflow.py                    # Summary and confirmation workflow
│   ├── moderation.py                          # Input/output safety checks
│   ├── persistence.py                         # Confirmed-visit JSON persistence
│   ├── observability.py                       # Privacy-safe prompt-chain events
│   ├── prompts/                               # Model-backed extractor, follow-up, and confirmation prompts
│   ├── rag/                                   # Iteration 3 knowledge branch
│   │   ├── config.py                          # Embedding profile, chunking, retrieval, guard settings
│   │   ├── documents.py                       # PDF extraction, cleaning, manifest-matched sectioning
│   │   ├── chunking.py                        # LlamaIndex TokenTextSplitter, 400/50
│   │   ├── sentence_window.py                 # Sentence parser and window builder (Part C)
│   │   ├── embeddings.py                      # OpenAI embeddings, with a query cache
│   │   ├── store.py                           # LlamaIndex PGVectorStore wrapper
│   │   ├── retrievers.py                      # Retriever protocol, BasicChunkRetriever, SentenceWindowRetriever
│   │   ├── query.py                           # Category inference, compound-question splitting
│   │   ├── policy.py                          # What may never be answered from documents
│   │   ├── evidence.py                        # Deterministic sufficiency check and near-miss guards
│   │   ├── generation.py                      # Grounded answer generation
│   │   ├── citations.py                       # Citation binding and validation
│   │   ├── pipeline.py                        # The branch, end to end
│   │   ├── integration.py                     # Seam into the chat flow; degrades to curated content
│   │   └── ingest.py                          # Corpus ingestion CLI
│   ├── graph/                                 # LangGraph orchestrator (Iteration 3, Part B)
│   │   ├── state.py                           # AssistantState
│   │   ├── nodes.py                           # Thin adapters over the existing modules
│   │   ├── build.py                           # Graph construction, conditional edges, checkpointer
│   │   └── runner.py                          # run_turn() — the /chat entry point
│   └── evaluators/
│       ├── regression_suite.py             # Prompt-chain, injection, and intent evaluations
│       ├── healthcare_assistant_benchmark.xlsx
│       ├── rag_benchmark_questions.xlsx     # 68 RAG questions and their expectations
│       ├── rag/
│       │   ├── dataset.py                   # Benchmark loader and the tune/holdout split
│       │   ├── deterministic.py             # Fact, citation and near-miss metrics
│       │   ├── shadow.py                    # Curated-vs-retrieved divergence classes
│       │   ├── judged.py                    # DeepEval metrics, batched
│       │   ├── similarity_profile.py        # Per-strategy similarity distributions
│       │   ├── compare_arms.py              # Part C arm comparison
│       │   └── run_benchmark.py             # RAG benchmark CLI
│       ├── equivalence/                     # Chain-vs-graph proof (Part B)
│       │   ├── harness.py                   # Turn-by-turn comparison, STRICT and STRUCTURAL
│       │   ├── caching_client.py            # Records model calls so replays are deterministic
│       │   ├── conversations.py             # The 20 compared conversations
│       │   ├── run_equivalence.py           # Equivalence CLI
│       │   └── rag_through_chat.py          # The 68 RAG questions through /chat, both orchestrators
│       └── benchmarks/
│           ├── test_loader.py              # Excel scenario loader
│           ├── preconditions.py            # Resolves a scenario's stated setup into turns
│           ├── test_runner.py              # Governed API execution
│           ├── evaluator.py                # Contract, state, and LLM-judge scoring
│           ├── conversation_loader.py      # Multi-turn flow loader
│           ├── conversation_runner.py      # Sequential session execution
│           ├── conversation_evaluator.py   # Session-integrity scoring
│           ├── rate_limiter.py             # Adaptive concurrency and jittered backoff
│           ├── checkpoint.py               # Batch checkpointing and resume
│           ├── report_generator.py         # Console and JSON reports
│           ├── run_benchmarks.py           # Scenario CLI
│           └── run_conversation_flows.py   # Conversation CLI
├── clinical_docs/                             # RAG corpus: MedlinePlus PDFs (gitignored) + manifest.yaml
├── docker-compose.yml                         # PostgreSQL + pgvector on port 5433
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
- An OpenAI API key for intent classification, structured extraction, ambiguous
  confirmation classification, embeddings and grounded answers
- Docker, for the knowledge store. **Optional**: without it the assistant runs
  exactly as before, answering preparation questions from curated content
  instead of from documents. Intake never depends on the database.

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

## Set up the knowledge store

Optional. Skip it and the assistant still runs; the knowledge branch stands down
and preparation questions are answered from curated content.

Start PostgreSQL with pgvector:

```bash
docker compose up -d
```

It listens on port 5433, deliberately, so it cannot be confused with a system
Postgres on 5432. `DATABASE_URL` in `.env.example` already matches.

Then ingest the corpus — 11 documents, 88 nodes, a few seconds and well under a
cent of embeddings:

```bash
uv run python -m src.rag.ingest plan     # what would change; no API calls
uv run python -m src.rag.ingest ingest   # extract, chunk, embed, store
uv run python -m src.rag.ingest status   # what is currently stored
```

There is no migration step: the table, its HNSW index and the `vector` extension
are created on first use, sized to the configured embedding model.

Ingestion is idempotent by content fingerprint — the PDF's hash plus a pipeline
version. Re-running it embeds nothing unchanged. Editing the cleaning or chunking
code means bumping `PIPELINE_VERSION` in `src/rag/config.py`, because a code
change alters the stored text while leaving the source file untouched.

The corpus itself is 12 MedlinePlus PDFs in `clinical_docs/`, described by
`clinical_docs/manifest.yaml`. The PDFs are gitignored; the manifest is tracked.

## Run the chatbot

Start the FastAPI development server from the repository root:

```bash
uv run uvicorn src.app:app --reload
```

Open <http://127.0.0.1:8000> in a browser. Stop the server with `Ctrl+C`.

Confirmed summaries are stored locally in `db/visits/`. They are not submitted to a provider or external healthcare system.

### Choosing the orchestrator

Two implementations serve a turn, and they are interchangeable. A LangGraph
`StateGraph` is the default; the original hand-rolled chain is one word away.

```bash
ORCHESTRATOR=chain uv run uvicorn src.app:app --reload   # roll back
```

It can also be chosen per request, which is how the equivalence harness runs
both against the same input:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "Do I need to fast before a blood test?",
       "session_id": "demo", "orchestrator": "chain"}'
```

The graph's nodes call the same functions the chain calls, so this is a choice
of control flow rather than of behaviour. That claim is checked rather than
asserted — see the equivalence harness below.

## Run tests

Run the complete test suite:

```bash
uv run python -m unittest discover -s tests -v
```

Run one test module, for example:

```bash
uv run python -m unittest tests.test_routing -v
```

The suite is offline and deterministic: model clients are faked, so no test
makes a paid API call. It covers the workflow schemas and completeness rules,
routing and global commands, extraction and merging, question selection, the
summary and confirmation workflow, the safety guardrails, and the benchmark
runner's rate-limit and checkpoint behaviour.

## Evaluations

Behaviour is checked at three levels, each answering a different question. They
are reported separately and never averaged, because a high score at one level
says nothing about the others.

| Level | Question it answers | Cost |
|---|---|---|
| Regression suite | Do the chain's nodes still behave? | Free for the deterministic half |
| RAG benchmark | Does the knowledge branch answer, refuse, and cite correctly? | One retrieval and one answer per question |
| Scenario benchmark | Was each individual reply right? | One API call per scenario, plus a judge |
| Conversation benchmark | Does a whole session hold together? | One session per flow, plus a judge |

### Regression suite

The prompt-chain, prompt-injection, and intent-classifier evaluations live in a
single suite, `src/evaluators/regression_suite.py`. Run all three:

```bash
uv run python -m src.evaluators.regression_suite
```

The prompt-chain half is deterministic and makes no API calls, so it runs
without a key and is the one to use in CI:

```bash
uv run python -m src.evaluators.regression_suite --only prompt-chain
```

The other two use the configured OpenAI API and may incur usage:

```bash
uv run python -m src.evaluators.regression_suite --only injection
uv run python -m src.evaluators.regression_suite --only intent
uv run python -m src.evaluators.regression_suite --only intent --dataset my_labels.json
```

Without an API key the suite still runs its deterministic half and reports the
model-backed evaluations as skipped, rather than failing outright.

Reports are written to `reports/` by default. That directory is generated output
and is ignored by Git.

### Scenario benchmark — 215 single-turn cases

Start the chatbot API in one terminal, then run the benchmark in another:

```bash
uv run python -m src.evaluators.benchmarks.run_benchmarks \
  --file src/evaluators/healthcare_assistant_benchmark.xlsx
```

Each scenario is scored in three layers — a response-contract check, a
conversation-state check, and an LLM-as-judge assessment — and passes only when
every applicable layer passes. Splitting them matters diagnostically: the layer
that failed says immediately whether the defect is in the data flow or the
wording. Full, summary, and failure-only JSON reports are written under
`reports/benchmarks/`.

#### Rate-limit handling

The suite drives an API that itself makes several model calls per turn, so the
practical ceiling on throughput is the provider's rate limit. Rather than
hard-coding a concurrency figure, the runner discovers a safe one:

- Scenarios run in small batches, starting at a deliberately low parallelism.
  An AIMD governor raises the limit by one after a sustained run of clean
  responses and halves it the moment a rate limit appears, so the suite settles
  at a level the provider tolerates.
- Every call — both benchmark turns and judge calls — retries throttling and
  transient server errors with exponential backoff and full jitter, honouring a
  server-supplied `Retry-After` when one is present.
- Each finished batch is appended to a checkpoint file. Re-running the same
  command resumes from it and evaluates only what is left; pass `--fresh` to
  start over.
- A case that exhausts its retry budget is recorded as `ERROR` and skipped. A
  rate limit never fails the run, and the full report is always produced.

The end of the run prints the counters that describe this — attempts, rate-limit
hits, retries, permanent failures, concurrency changes, and total time spent
backing off.

Useful options include:

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

# Stay deliberately gentle on a constrained API key
uv run python -m src.evaluators.benchmarks.run_benchmarks \
  --start-concurrency 1 --max-concurrency 2 --batch-size 5 --batch-pause 2
```

### Conversation benchmark — 34 multi-turn sessions

The `Conversation Flows` sheet holds whole sessions rather than isolated turns.
Each is replayed in order on one session ID, and scored on three things a
single-turn suite cannot see:

- **State persistence** — everything captured earlier is still present at the
  end, except where a later turn legitimately changed it.
- **Recovery correctness** — after a correction or restart the old value is gone
  *and* the new one is there. Half of that is not a recovery.
- **Tone and safety consistency** — an escalation or a refused injection holds
  for the rest of the session rather than lapsing on the next turn.

A conversation fails on exactly three things: a turn error, corrupted state, or
an emergency or injection turn that fails to override the normal flow.

Per-turn expectations and the session judge are reported as **graded
diagnostics**, not veto conditions — a turn-expectation percentage per
conversation and across the run. They measure per-turn correctness, which the
scenario suite already measures directly; letting one of them sink an otherwise
sound eight-turn session would make this metric a worse copy of that one instead
of measuring what only it can, which is whether a session holds together.

```bash
uv run python -m src.evaluators.benchmarks.run_conversation_flows

# One flow, or the first few
uv run python -m src.evaluators.benchmarks.run_conversation_flows --conv-id CONV-14
uv run python -m src.evaluators.benchmarks.run_conversation_flows --limit 10

# Print the single-turn score alongside it, for contrast
uv run python -m src.evaluators.benchmarks.run_conversation_flows \
  --scenario-summary reports/benchmarks/run15/benchmark_summary_*.json
```

Results land in `reports/conversations/`: `conversation_flow_report.json` at the
end, and `checkpoint.jsonl` written per conversation as the run proceeds, so a
long run can be inspected or resumed while it is still going.

Both benchmarks share the same rate-limit handling, checkpointing and resume
behaviour described above.

### Equivalence harness — chain against graph

The check that let the LangGraph orchestrator become the default. It runs both
implementations over the same 20 conversations and compares every reply and
every state field.

```bash
# Record the model calls once, then compare
uv run python -m src.evaluators.equivalence.run_equivalence --record

# Replay offline and free; fails if any call is not cached
uv run python -m src.evaluators.equivalence.run_equivalence

# The 68 RAG questions through /chat, both orchestrators
uv run python -m src.evaluators.equivalence.rag_through_chat
```

Model calls are recorded to `reports/partb/model_cache.json` and replayed, for
a reason worth knowing before reading the output: measured chain-against-chain,
**20% of conversations diverge from themselves**, because four model calls in
the turn can each flip. Without recording, a correct reimplementation would fail
a byte-identical criterion a fifth of the time and a real bug would be
indistinguishable from that jitter.

A run that reports "equivalent" while recording cache misses is treated as a
failure, not a pass. Several call sites catch their own model failures and
return a fallback, so a miss produces the *same* fallback on both sides — the
comparison would otherwise pass by agreeing about nothing.

Re-record after any prompt change; the cache is stale by design when prompts
move. Note that only chat calls are recorded, not embeddings, so retrieval can
occasionally reorder and cause a stray miss.

Reports are written to `reports/partb/`.

### RAG benchmark — 68 questions

Needs the knowledge store running and an API key.

```bash
uv run python -m src.evaluators.rag.run_benchmark --split holdout
uv run python -m src.evaluators.rag.run_benchmark --split tune
uv run python -m src.evaluators.rag.run_benchmark --group near_miss
```

The set is split in half by a hash of the question id. **Tune on `tune`, quote
`holdout`** — every threshold and prompt in the branch was tuned by looking at
results, and a number measured on the questions it was tuned against is
optimistic by an unknown amount.

Questions fall into six groups, and the ones that matter most are the negative
ones: `near_miss` questions have no answer in the corpus but sit close to one
that does, and `never_route` questions must be refused before retrieval runs at
all. A wrong answer to either is invisible to a faithfulness metric, because an
answer grounded in the wrong passage is still faithful to that passage.

Current results are measured on corpus version 2, the 11 documents that remain
after two licensed Medical Encyclopedia sources were excluded and replaced with
thinner NLM-authored Health Topic pages. On the held-out split: outcome accuracy
94.3%, and near-miss resistance, gap disclosure, citation validation and
never-route compliance all at 100% with zero wrong-document grounding. The
version 1 files are kept alongside for the comparison, and the licence
remediation is documented in
[the RAG architecture](documentation/rag_architecture.md#a13-licence-remediation-and-benchmark-impact).

The retrieval strategy is selectable, which is how Part C compared them:

```bash
# Sentence-window retrieval, window size 2, against the sentence table
uv run python -m src.evaluators.rag.run_benchmark --strategy sentence_window \
    --window 2 --top-k 6 --label sw_w2

# Lay the arm reports side by side
uv run python -m src.evaluators.rag.compare_arms --split holdout
```

Sentence windows are **not** the default. They win every retrieval metric and
lose faithfulness and fact coverage, which is the wrong trade here; the
reasoning and the numbers are in
[Part C of the RAG architecture](documentation/rag_architecture.md). The
sentence table is populated by `ingest --strategy sentence_window`.

Reports are written to `reports/rag/`.

#### Judged metrics (DeepEval)

Optional second reporting section, off by default:

```bash
uv run python -m src.evaluators.rag.run_benchmark --split holdout --judge
```

It adds faithfulness, answer relevancy, and contextual precision/recall/relevancy
— the things that need a model to read both the answer and the context, and
which the deterministic metrics cannot express. Faithfulness in particular is
what citation validation does *not* cover: that check confirms a marker
resolves, not that the passage it points at supports the claim.

Four things to keep in mind:

- **Judged scores are not reproducible** and are comparable only against the
  same judge, which is recorded in each report as `judge_model`.
- **Nothing is gated on them.** The promotion gates stay deterministic — a
  safety gate that moves with a judge's model version can loosen silently.
- **Only answered cases are judged**; refusals and fallbacks are scored in the
  deterministic section, where "did it refuse" is a fact rather than a judgement.
- **It is slow and metered** — five metrics, several calls each, per case. Pair
  it with `--split`, `--group` or `--limit`.

See [the RAG architecture](documentation/rag_architecture.md#a121-judged-metrics-deepeval)
for what stays deterministic and why.

## Data and deployment notes

- Session state is held in memory and expires after 15 minutes.
- Only completed, explicitly confirmed summaries are persisted.
- Local JSON records use UUID filenames and restricted file permissions.
- Local records are not encrypted; authentication, production encryption, export formats, durable multi-user sessions, RAG knowledge management, and administrative monitoring belong to later iterations in the SRS.
- The knowledge corpus is general patient education from MedlinePlus, not a
  particular clinic's own instructions, and answers say so. It never overrides
  what the ordering clinic told the patient.
- The knowledge store holds no patient data — only the public documents and
  their embeddings. Retrieval telemetry records node ids, scores and token
  counts, never the question text.
