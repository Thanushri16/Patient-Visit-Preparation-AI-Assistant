# Reports

Generated evidence. Nothing here is hand-written, and nothing here is an input
to the application — these are the artefacts the claims in
`documentation/rag_architecture.md` are checked against.

## Layout

One directory per producer. Within a directory, filenames are **descriptive and
stable**: a re-run overwrites its file rather than adding a timestamped sibling.
The generation time lives *inside* each file, in `generated_at`. Timestamped
filenames were used at first and produced piles of near-identical reports that
had to be pruned by hand, with no way to tell which one a document was quoting.

| Directory | Produced by | Contents |
|---|---|---|
| `benchmarks/` | `evaluators.benchmarks.run_benchmarks` | 215-case scenario benchmark, chain baseline |
| `conversations/` | `evaluators.benchmarks.run_conversation_flows` | 34-session conversation benchmark, chain baseline |
| `rag/` | `evaluators.rag.run_benchmark` | Part A: RAG quality, per corpus version and split |
| `partb/` | `evaluators.equivalence.*`, plus benchmark re-runs | Part B: chain-vs-graph equivalence |

`benchmarks/` and `conversations/` keep the tool's own timestamped naming
because those paths are the CLI defaults and predate this convention. They hold
the chain baselines the Part B comparison is measured against.

## Part A — `rag/`

`rag_partA_<kind>_corpus_v<N>_<split>.json`

- `kind` — `baseline` (deterministic metrics) or `deepeval` (judged metrics)
- `corpus_v1` was the corpus before the licence remediation; `corpus_v2` is the
  current 11-document, 81-node corpus. Both are kept because §A.13 compares them
- `split` — `tune`, `holdout`, or `all`. The holdout file is the one to quote:
  thresholds were tuned on `tune`, so `all` includes questions the settings saw

## Part B — `partb/`

| File | What it evidences |
|---|---|
| `partB_equivalence.json` | 20 conversations, chain vs graph, cached replay. The headline: zero divergence, zero misdeclared, 100% cache hit rate |
| `partB_rag_through_chat.json` | All 68 RAG questions routed through `/chat` on both orchestrators. Part A's own benchmark cannot do this — it calls the pipeline directly and never reaches the orchestrator |
| `partB_scenario_chain.json` | 215-case scenario benchmark, chain, on Part B code |
| `partB_scenario_graph.json` | The same, graph. Read alongside the chain file: the difference between them is within the benchmark's own run-to-run noise, which §B.5 quantifies |
| `partB_conversation_graph.json` | 34-session conversation benchmark, graph. Compare with `conversations/` |
| `model_cache.json` | **Not a report.** The recorded model responses that make equivalence replays deterministic and free. Regenerate with `--record` after any prompt change; it is stale by design when prompts move |

The two scenario files are summaries only. The per-case transcript dumps were
dropped: every figure cited in §B.5 is in the summaries, and the dumps ran to
roughly 900 KB of duplicated conversation text.

## Regenerating

```bash
# Part A
uv run python -m src.evaluators.rag.run_benchmark --split holdout [--judge]

# Part B equivalence (needs --record once after any prompt change)
uv run python -m src.evaluators.equivalence.run_equivalence --record
uv run python -m src.evaluators.equivalence.rag_through_chat --record

# Benchmarks: need a running server, and the orchestrator is chosen by env var
ORCHESTRATOR=graph uv run uvicorn app:app --app-dir src --port 8000
uv run python -m src.evaluators.benchmarks.run_benchmarks --fresh --output-dir reports/partb
```
