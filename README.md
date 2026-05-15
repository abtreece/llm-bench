# llm-bench

Local LLM coding benchmark for Ollama models on javelin.

A self-contained benchmark: a small purpose-built test application (`money`
domain), a corpus of bug-fix cases against it, and a harness that runs each
model against each case and grades deterministically via `pytest`.

The inner benchmarking loop talks only to local Ollama at
`http://localhost:11434`. No cloud LLM is in the per-case path.

See `BENCH_PLAN.md` for the full execution plan. A richer README lands in
Phase 6.
