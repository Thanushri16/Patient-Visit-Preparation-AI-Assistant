# Prompt-Chaining Architecture Concepts for Chatbots

Prompt chaining is the practice of splitting a complex chatbot workflow into multiple focused LLM calls, where each step performs one clear responsibility and passes structured output to the next step.

For a chatbot with a menu, data collection, validation, summarization, and confirmation phases, the best architecture usually combines several of the patterns below.

---

## 1. State Machine Architecture

A state machine models the chatbot as a set of explicit conversation phases.

Example states:

```text
MENU
DATA_COLLECTION
VALIDATION
SUMMARIZATION
CONFIRMATION
COMPLETED
ESCALATED
```

Each user message is processed according to the current state.

Example flow:

```text
MENU
  ↓
DATA_COLLECTION
  ↓
VALIDATION
  ↓
SUMMARIZATION
  ↓
CONFIRMATION
  ↓
COMPLETED
```

### Best for

- Form-like chatbots
- Customer support workflows
- Intake assistants
- Applications and claims
- Guided data collection
- Multi-step service requests

### Key design principle

The application should control state transitions. The model should not independently decide the entire workflow.

```python
if missing_fields:
    state = "DATA_COLLECTION"
elif validation_errors:
    state = "VALIDATION"
else:
    state = "SUMMARIZATION"
```

### Benefits

- Predictable behavior
- Easy debugging
- Clear phase ownership
- Better testing
- Safer workflow execution

---

## 2. Router–Worker Architecture

A router first identifies the user's intent and sends the request to a specialized worker.

```text
User Message
     ↓
   Router
     ↓
 ┌───────────────┬────────────────┬────────────────┐
 │ Report Issue  │ Request Service│ Check Status   │
 │ Worker        │ Worker         │ Worker         │
 └───────────────┴────────────────┴────────────────┘
```

Each worker has its own prompt, schema, and validation rules.

### Best for

- Menu-based chatbots
- Customer support systems
- Bots with multiple workflows
- Systems with highly different intents

### Router output example

```json
{
  "workflow": "report_issue",
  "confidence": 0.94
}
```

### Benefits

- Keeps prompts small
- Prevents instruction conflicts
- Improves specialization
- Makes evaluation easier
- Supports independent workflow updates

---

## 3. Extract–Validate–Respond Pattern

Every data-collection turn is split into three stages.

```text
User Message
    ↓
Extract Structured Data
    ↓
Validate Data
    ↓
Generate Next Response
```

### Example

The user says:

```text
It started yesterday and I am using a MacBook.
```

Extraction output:

```json
{
  "start_time": "yesterday",
  "device_type": "MacBook"
}
```

Validation checks whether values are valid and complete.

The response generator asks only for the next missing information.

### Best for

- Structured intake
- Support forms
- Registration flows
- Troubleshooting assistants
- Claims and applications

### Benefits

- Separates reasoning from wording
- Reduces hallucination
- Supports corrections
- Makes field-level evaluation possible

---

## 4. Schema-Driven Collection

Define the required and optional fields for each workflow before starting data collection.

```python
WORKFLOW_SCHEMAS = {
    "report_issue": {
        "required": [
            "issue_type",
            "issue_description",
            "start_time",
            "location"
        ],
        "optional": [
            "device_type",
            "error_message",
            "troubleshooting_attempted"
        ]
    }
}
```

The schema becomes the source of truth for completeness.

### Best for

- Structured workflows
- Forms
- Applications
- Service requests
- Compliance-heavy systems

### Benefits

- Deterministic completeness checks
- Easier versioning
- Easier validation
- Better analytics
- Lower prompt complexity

### Recommendation

Use code, not the LLM, to determine which required fields are still missing.

---

## 5. Conditional Branching Chains

The required workflow changes based on earlier answers.

```text
Was anyone injured?
     ↓
   Yes ─────────→ Collect injury information
     ↓
    No ─────────→ Skip injury section
```

### Example

```python
if issue_type == "payment":
    required_fields += ["transaction_id", "amount"]

if issue_type == "technical":
    required_fields += ["device_type", "error_message"]
```

### Best for

- Dynamic forms
- Claims processing
- Troubleshooting
- Medical or insurance intake
- Eligibility workflows

### Benefits

- Avoids irrelevant questions
- Shortens conversations
- Improves user experience
- Supports personalized flows

---

## 6. Plan-and-Execute Architecture

The system first creates a compact workflow plan and then executes it step by step.

Example plan:

```json
{
  "workflow": "insurance_claim",
  "steps": [
    "collect_incident_details",
    "collect_policy_details",
    "collect_damage_details",
    "review_summary",
    "submit_claim"
  ]
}
```

### Best for

- Long workflows
- Workflows with optional sections
- Dynamic task sequences
- Multi-step service operations
- Complex enterprise assistants

### Benefits

- Makes the process inspectable
- Supports dynamic plans
- Allows step retries
- Separates planning from execution

### Caution

Do not use plan-and-execute for simple flows that a state machine can handle more reliably.

---

## 7. Sequential Prompt Chain

Each step runs in a fixed order.

```text
Prompt 1 → Prompt 2 → Prompt 3 → Prompt 4
```

Example:

```text
Intent Detection
    ↓
Field Extraction
    ↓
Validation
    ↓
Summarization
```

### Best for

- Predictable pipelines
- Document processing
- Simple multi-stage classification
- Fixed chatbot flows

### Benefits

- Easy to understand
- Easy to implement
- Easy to test

### Limitation

It is less suitable when the workflow must branch or repeat steps.

---

## 8. Looping or Iterative Collection Chain

The chatbot repeatedly extracts data, checks completeness, and asks the next question.

```text
User Answer
    ↓
Extract Fields
    ↓
Check Missing Fields
    ↓
Ask Next Question
    ↓
Repeat
```

### Best for

- Multi-turn forms
- Guided interviews
- Support intake
- Onboarding
- Troubleshooting

### Exit condition

```python
if not missing_fields and not validation_errors:
    move_to_summary()
```

### Benefits

- Natural multi-turn interaction
- Supports partial answers
- Can capture multiple fields from one message
- Avoids asking already answered questions

---

## 9. Hierarchical Prompt Chaining

A high-level controller delegates work to lower-level specialized chains.

```text
Conversation Controller
      ↓
Workflow Controller
      ↓
Task-Specific Prompt Chain
```

Example:

```text
Main Router
 ├── Billing Workflow
 │    ├── Collect Transaction
 │    ├── Validate Amount
 │    └── Create Summary
 └── Technical Support Workflow
      ├── Collect Device
      ├── Diagnose Error
      └── Recommend Next Step
```

### Best for

- Large enterprise chatbots
- Many business domains
- Large prompt libraries
- Systems maintained by multiple teams

### Benefits

- Modular architecture
- Independent testing
- Better ownership boundaries
- Easier scaling

---

## 10. Parallel Prompt Chaining

Multiple prompts run at the same time and their outputs are combined.

```text
                 ┌→ Intent Classification
User Message ────┼→ Safety Check
                 ├→ Sentiment Analysis
                 └→ Entity Extraction
                         ↓
                    Merge Results
```

### Best for

- Independent analyses
- Latency-sensitive workflows
- Multi-signal decision systems
- Input preprocessing

### Example parallel tasks

- Intent detection
- Prompt injection detection
- Sentiment analysis
- Entity extraction
- Language detection

### Benefits

- Lower overall latency
- Independent components
- Better modularity

### Caution

Only parallelize tasks that do not depend on each other's output.

---

## 11. Fan-Out and Fan-In Architecture

One input is sent to multiple specialized prompts, then an aggregator combines the results.

```text
                 ┌→ Specialist A
Input ───────────┼→ Specialist B
                 └→ Specialist C
                         ↓
                     Aggregator
```

### Best for

- Complex analysis
- Multi-perspective review
- Policy checking
- Domain-specific validation
- High-value decision support

### Example

A support request could be reviewed by:

- A policy checker
- A completeness checker
- A risk classifier

The aggregator chooses the final action.

### Benefits

- Diverse analysis
- Better fault tolerance
- Stronger decision quality

### Caution

This pattern increases cost and latency.

---

## 12. Generator–Critic Architecture

One prompt generates an answer, and another prompt reviews it.

```text
Generator
    ↓
Critic
    ↓
Revision or Approval
```

### Critic checks

- Missing information
- Unsupported claims
- Policy violations
- Contradictions
- Incorrect formatting

### Best for

- High-quality summaries
- Customer-facing responses
- Important reports
- Compliance workflows
- Complex explanations

### Benefits

- Improves output quality
- Catches omissions
- Reduces unsupported content

### Caution

The critic should have clear evaluation criteria, not vague instructions such as “review this.”

---

## 13. Generate–Evaluate–Revise Loop

The model produces an output, an evaluator scores it, and the model revises it if needed.

```text
Generate
   ↓
Evaluate
   ↓
Pass? ── Yes → Return
   ↓ No
Revise
   ↓
Evaluate Again
```

### Best for

- Complex summaries
- Important final responses
- Strict formatting
- High-quality writing
- Policy-sensitive outputs

### Example evaluator output

```json
{
  "passed": false,
  "issues": [
    "The summary omitted the user's preferred date.",
    "The response added an unsupported assumption."
  ]
}
```

### Recommendation

Set a maximum revision count to prevent infinite loops.

---

## 14. Guardrail Chain

A guardrail stage checks input or output before the workflow proceeds.

```text
User Message
    ↓
Input Guardrail
    ↓
Workflow Chain
    ↓
Output Guardrail
    ↓
User
```

### Input guardrail checks

- Prompt injection
- Unsupported requests
- Sensitive data
- Abusive content
- Attempts to reveal system instructions
- Attempts to bypass required steps

### Output guardrail checks

- Personally identifiable information
- Policy violations
- Hallucinated claims
- Unsafe instructions
- Invalid formatting

### Best for

- Public chatbots
- Enterprise assistants
- Regulated systems
- Systems with tool access
- Systems handling personal data

---

## 15. Human-in-the-Loop Chain

The system escalates to a human when certain conditions are met.

```text
Automated Workflow
       ↓
Escalation Condition
       ↓
Human Agent
```

### Escalation triggers

```python
if user_requests_human:
    escalate()

if confidence < threshold:
    escalate()

if validation_failures >= 3:
    escalate()

if workflow_not_supported:
    escalate()
```

### Handoff payload

```json
{
  "reason_for_handoff": "Unable to validate account information",
  "selected_workflow": "report_issue",
  "collected_data": {},
  "missing_fields": [],
  "conversation_summary": ""
}
```

### Best for

- Customer support
- High-risk decisions
- Complex exceptions
- Low-confidence classification
- Sensitive issues

### Benefit

The user should not have to repeat previously collected information.

---

## 16. Confirmation and Correction Chain

The chatbot summarizes collected information and asks the user to confirm it before performing an action.

```text
Collected Data
    ↓
Summary
    ↓
User Confirmation
    ├── Confirm → Complete
    └── Correct → Update → Revalidate → Resummarize
```

### Best for

- Submissions
- Orders
- Claims
- Appointments
- Payments
- Account updates

### Confirmation classifier output

```json
{
  "action": "confirm",
  "corrections": {}
}
```

or:

```json
{
  "action": "correct",
  "corrections": {
    "preferred_date": "Monday"
  }
}
```

### Benefits

- Prevents accidental submissions
- Supports natural corrections
- Improves trust
- Creates a clear audit point

---

## 17. Memory and State Separation

Conversation state should be stored outside the model.

```python
conversation_state = {
    "phase": "data_collection",
    "selected_workflow": "report_issue",
    "collected_data": {},
    "missing_fields": [],
    "validation_errors": [],
    "confirmed": False
}
```

### Best practice

The model interprets language. The application owns:

- Current phase
- Collected values
- Required fields
- Workflow rules
- Retry count
- Submission status
- Tool execution state

### Benefits

- Prevents memory loss
- Supports retries
- Enables analytics
- Reduces prompt size
- Makes workflows auditable

---

## 18. Structured Output Chaining

Each prompt returns JSON or another strict schema rather than free-form text.

Example:

```json
{
  "updates": {
    "device_type": "MacBook"
  },
  "corrections": {},
  "uncertain_fields": [],
  "next_action": "ask_question"
}
```

### Best for

- Tool calling
- Workflow control
- Field extraction
- Validation
- Routing
- Confirmation

### Benefits

- Easier parsing
- Easier testing
- Less ambiguity
- Safer handoffs between prompts

### Recommendation

Validate model output with a typed schema such as Pydantic, Zod, or JSON Schema.

---

## 19. Retrieval-Augmented Prompt Chain

The chatbot retrieves relevant policies, documentation, or knowledge before generating a response.

```text
User Question
    ↓
Query Rewriting
    ↓
Retrieval
    ↓
Context Filtering
    ↓
Answer Generation
```

### Best for

- Knowledge-base chatbots
- Policy assistants
- Documentation support
- Product support
- Enterprise Q&A

### Important sub-chains

- Query classification
- Query rewriting
- Document retrieval
- Relevance filtering
- Citation generation
- Answer validation

### Caution

Do not use retrieval for data the user has already provided in the conversation state.

---

## 20. Tool-Orchestrated Prompt Chain

The model determines when a tool is needed, while the application validates and executes the tool call.

```text
User Request
    ↓
Tool Decision
    ↓
Argument Extraction
    ↓
Validation
    ↓
Tool Execution
    ↓
Result Interpretation
    ↓
User Response
```

### Best for

- Scheduling
- Database lookup
- Ticket creation
- Order status
- Email operations
- CRM actions

### Safety rule

Require confirmation before irreversible or sensitive tool actions.

---

## 21. Event-Driven Workflow Architecture

Each event triggers a specific node in the chain.

Example events:

```text
USER_MESSAGE_RECEIVED
WORKFLOW_SELECTED
FIELD_UPDATED
VALIDATION_FAILED
DATA_COMPLETE
SUMMARY_CONFIRMED
REQUEST_SUBMITTED
```

### Best for

- Distributed systems
- Long-running workflows
- Multi-channel chatbots
- Systems with queues
- Enterprise orchestration

### Benefits

- Clear audit trail
- Resilient retries
- Easier integration
- Supports asynchronous systems

---

## 22. Graph-Based Prompt Chaining

The workflow is represented as a graph rather than a simple sequence.

```text
Menu
  ↓
Collect Data
  ├── Missing Data → Ask Question → Collect Data
  ├── Invalid Data → Clarify → Collect Data
  └── Complete → Summarize → Confirm
                         ├── Correct → Collect Data
                         └── Confirm → Complete
```

### Best for

- Branching workflows
- Cycles and retries
- Complex state transitions
- Multi-agent or multi-tool systems

### Benefits

- Models real conversation flows
- Supports loops
- Handles exceptions
- Easier to visualize

---

## 23. Supervisor–Agent Architecture

A supervisor model coordinates specialized agents.

```text
Supervisor
 ├── Menu Agent
 ├── Data Collection Agent
 ├── Validation Agent
 ├── Summary Agent
 └── Escalation Agent
```

### Best for

- Large systems
- Multiple specialist capabilities
- Complex domain reasoning
- Multi-agent experimentation

### Caution

For a predictable menu and intake chatbot, a deterministic state machine is usually simpler, cheaper, and safer than a fully autonomous multi-agent system.

---

## 24. Confidence-Based Routing

The chain changes behavior based on model confidence.

```python
if confidence >= 0.85:
    continue_workflow()
elif confidence >= 0.60:
    ask_clarifying_question()
else:
    show_menu_or_escalate()
```

### Best for

- Intent classification
- Entity extraction
- Workflow selection
- Risk detection

### Caution

Model-reported confidence is not automatically calibrated. Evaluate and calibrate thresholds using a labeled dataset.

---

## 25. Fallback and Recovery Chain

The chatbot has explicit recovery behavior when a step fails.

```text
Primary Prompt
    ↓
Parsing Failed?
    ├── No → Continue
    └── Yes → Retry Prompt
                  ↓
             Still Failed?
                  ├── No → Continue
                  └── Yes → Safe Fallback
```

### Recovery strategies

- Retry with stricter formatting
- Use a smaller fallback schema
- Ask the user a direct question
- Route to a human
- Log the failed output

### Best for

- Production chatbots
- Structured output parsing
- Tool calls
- External API workflows

---

# Recommended Architecture for a Menu + Data Collection + Summary Chatbot

A strong production architecture is:

```text
1. Input Guardrail
2. Menu Intent Router
3. Workflow Schema Loader
4. Field Extractor
5. Deterministic Validator
6. Semantic Validator
7. Missing-Field Checker
8. Next-Question Generator
9. Summary Generator
10. Confirmation and Correction Classifier
11. Final Action or Submission
12. Output Guardrail
13. Human Escalation Fallback
```

Flow:

```text
User Message
    ↓
Input Guardrail
    ↓
State-Based Router
    ↓
Menu Selection
    ↓
Data Extraction Loop
    ↓
Validation
    ↓
Completeness Check
    ↓
Summary
    ↓
Confirmation
    ├── Correction → Revalidate → Resummarize
    ├── Confirm → Submit
    └── Unclear → Ask Direct Confirmation
```

---

# Practical Prompt-Chain Components

A useful prompt library could contain:

```text
P1: Menu Intent Classifier
P2: Field Extractor
P3: Semantic Validator
P4: Next-Question Generator
P5: Summary Generator
P6: Confirmation and Correction Classifier
P7: Input Guardrail
P8: Output Guardrail
P9: Escalation Summary Generator
P10: Final Response Generator
```

For a minimum viable implementation:

```text
P1: Router
P2: Extractor
P3: Question Generator
P4: Summarizer
P5: Confirmation Classifier
```

Use normal application code for:

- Required-field checks
- Allowed values
- Date parsing
- Numeric validation
- State transitions
- Retry limits
- Tool execution
- Submission status

---

# Prompt-Chaining Design Tips

## Keep one responsibility per prompt

Good:

```text
Extract structured values from the latest user message.
```

Bad:

```text
Understand the request, route it, collect data, validate it, summarize it,
decide whether to escalate, and respond naturally.
```

---

## Keep workflow control in code

Use the LLM for:

- Language interpretation
- Entity extraction
- Classification
- Semantic validation
- Natural-language generation

Use code for:

- State transitions
- Required fields
- Schema validation
- Retry counts
- Tool permissions
- Final execution decisions

---

## Pass only relevant context

For routing, pass:

```text
Menu options
Latest user message
```

For extraction, pass:

```text
Workflow schema
Current structured state
Latest user message
```

For summarization, pass:

```text
Validated structured data
```

Avoid passing the full conversation to every node.

---

## Support corrections at every phase

Users may say:

```text
Actually, change the date to Monday.
```

The extractor should distinguish:

```json
{
  "updates": {},
  "corrections": {
    "preferred_date": "Monday"
  }
}
```

Corrections should override earlier values and trigger revalidation.

---

## Prevent silent assumptions

Include instructions such as:

```text
Do not infer or invent missing values.
Use null when the user has not explicitly provided the information.
```

---

## Use typed output validation

Validate each prompt output with:

- Pydantic
- Zod
- JSON Schema
- Dataclasses
- Typed dictionaries

Reject or retry malformed output.

---

## Add observability

Log each chain node's:

- Prompt version
- Input
- Output
- State before and after
- Latency
- Token usage
- Validation result
- Retry count
- Error category

Do not log sensitive user data unless necessary and permitted.

---

## Version prompts independently

Use prompt identifiers such as:

```text
menu_router_v3
field_extractor_v5
summary_generator_v2
```

This allows safe testing and rollback.

---

## Evaluate each node separately

Do not evaluate only the final chatbot response.

Evaluate:

- Router accuracy
- Field extraction precision and recall
- Correction detection
- Completeness detection
- Validation quality
- Question relevance
- Summary faithfulness
- Confirmation classification
- Escalation correctness

---

## Limit retries and loops

Every chain loop should have an exit rule.

```python
MAX_EXTRACTION_RETRIES = 2
MAX_VALIDATION_ATTEMPTS = 3
MAX_REVISION_ROUNDS = 2
```

After the limit, use a safe fallback or escalate.

---

## Require confirmation before irreversible actions

Examples:

- Submit a claim
- Place an order
- Send an email
- Update an account
- Schedule an appointment
- Process a payment

The confirmation should clearly show the data or action being approved.

---

## Design for interruption

The user may change topics or request a human at any point.

Support global intents such as:

```text
CANCEL
RESTART
GO_BACK
CHANGE_ANSWER
SHOW_MENU
HUMAN_SUPPORT
HELP
```

These should be handled before normal phase logic.

---

# Architecture Selection Guide

| Scenario | Recommended Pattern |
|---|---|
| Fixed menu and form flow | State Machine |
| Many menu workflows | Router–Worker |
| Structured data collection | Extract–Validate–Respond |
| Dynamic required fields | Conditional Branching |
| Long dynamic workflows | Plan-and-Execute |
| Repeated question loop | Iterative Collection |
| Independent analyses | Parallel Chain |
| Multiple specialist reviews | Fan-Out and Fan-In |
| High-quality final output | Generator–Critic |
| Strict output requirements | Generate–Evaluate–Revise |
| Knowledge-base answers | Retrieval-Augmented Chain |
| External operations | Tool-Orchestrated Chain |
| Complex loops and branches | Graph-Based Chain |
| Public or regulated chatbot | Guardrail Chain |
| Low-confidence or exception cases | Human-in-the-Loop |
| Production resilience | Fallback and Recovery |

---

# Recommended Default

For most menu-based chatbots, start with:

```text
State Machine
+ Router–Worker
+ Schema-Driven Collection
+ Extract–Validate–Respond
+ Iterative Collection Loop
+ Confirmation and Correction
+ Guardrails
+ Human Fallback
```

This combination is usually more reliable than a fully autonomous agent architecture.

The core principle is:

> Let the LLM interpret language, but let your application control the workflow.
