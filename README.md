# llm-bench

Local LLM coding benchmark for Ollama models on javelin.

A self-contained benchmark: two small purpose-built target modules
(`app/money.py` currency arithmetic, `app/analysis.py` transaction
analytics over CSV), and a corpus of cases against them spanning two
categories — `coding` and `data-analysis` — with bug-fix, greenfield, and
adversarial variants. The harness runs each model against each case and
grades deterministically: `pytest` pass/regression for most cases, refusal
grading for adversarial ones.

The inner benchmarking loop talks only to local Ollama at
`http://localhost:11434`. No cloud LLM is in the per-case path.

See `BENCH_PLAN.md` for the full execution plan. A richer README lands in
Phase 6.
