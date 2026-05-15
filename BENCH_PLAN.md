# Local LLM Coding Benchmark — Execution Plan

> This is a handoff document for Claude Code (running on javelin) to follow.
> The plan is **phased with mandatory checkpoints**. Do not chain phases.

---

## Context

The user (abtreece) wants to benchmark the local Ollama models installed on
javelin (Dell R430, Ubuntu 24.04, Ollama 0.23.4) on real coding tasks. The
benchmark must be self-contained: a small purpose-built test application,
a corpus of bug-fix cases against it, and a harness that runs each model
against each case and grades deterministically via the test suite.

Claude Code is the **orchestrator and builder only**. Once the harness is
built, the inner benchmarking loop talks to local Ollama. No Claude Code or
Anthropic API call goes in the per-case hot loop.

---

## Goals

- A reproducible benchmark living in a fresh local git repo on javelin
- Deterministic grading: pass/fail comes from `pytest`, not a judge
- Per-model output: pass rate, regression rate, latency, tokens/sec
- Captures variance: each (model, case) pair runs N=3 times at temperature 0
- Runs entirely offline against `http://localhost:11434`

## Non-Goals

- Not benchmarking Claude Code itself
- Not pointing at any Spreedly or production code
- Not using any cloud LLM in the per-case loop
- No Slicer / microVM sandboxing in v1 — git worktrees are sufficient
- No multi-turn agentic loops in v1 — single-shot prompts only

---

## Architecture

```
test app (canonical "good" state, pytest passes clean)
   │
   └── corpus/*.yaml  ── each case = { breaking_patch, test_patch, prompt, reference_patch }
                          │
                          ▼
                    harness/run.py
                          │
              ┌───────────┼───────────┐
              ▼                       ▼
   git worktree per case      Ollama HTTP @ localhost:11434
              │                       │
              ▼                       ▼
        apply patches            model output (whole-file blocks)
              │                       │
              └──────────┬────────────┘
                         ▼
                   pytest target → pass/fail + regression count
                         │
                         ▼
                   results/<run_id>.csv → REPORT.md
```

**Edit format** (instructed to the model under test): emit one or more
fenced code blocks, each prefixed with a path comment:

````
```python
# path: app/money.py
<complete file contents>
```
````

Whole-file replacement is chosen over unified diff because small local
models reliably fail at producing applyable diffs. Whole-file blocks are
trivial to parse and apply. Trade-off: cases must target small files
(<300 lines). Design the test app accordingly.

---

## Phased Execution

### Phase 0 — Setup and choices

**Ask the user** for these decisions before doing anything else:

1. **Test app domain.** Pick one:
   - `money` — currency parse/format, rounding modes, minor↔major unit
     conversion, multi-currency arithmetic. Payments-domain adjacent.
   - `cidr` — subnet math, CIDR overlap/containment, host enumeration,
     IPv4/IPv6 parsing. Infra-adjacent.
   - `webhook` — HMAC signature verify, timestamp tolerance, payload
     canonicalization. Security-adjacent.
2. **Repo path.** Suggest `~/llm-bench`. Confirm or override.
3. **Python version.** Suggest the system's default Python 3 (verify
   `python3 --version`). Confirm.

Do **not** assume defaults. Wait for explicit answers.

---

### Phase 1 — Repo scaffolding

- `git init` at the chosen path
- Project layout:
  ```
  llm-bench/
  ├── README.md
  ├── BENCH_PLAN.md         (copy this file in)
  ├── pyproject.toml
  ├── models.yaml           (list of Ollama tags to benchmark)
  ├── app/                  (the test application — empty for now)
  ├── tests/                (the test suite — empty for now)
  ├── cases/                (corpus YAMLs — empty for now)
  ├── harness/
  │   ├── __init__.py
  │   ├── run.py
  │   ├── ollama_client.py
  │   ├── scorer.py
  │   └── report.py
  └── results/              (gitignored)
  ```
- `pyproject.toml` deps: `pyyaml`, `requests`, `pytest`
- `models.yaml` seeded with javelin's installed tags (see Reference below)
- `.gitignore` for `__pycache__`, `results/`, `.venv/`, worktree scratch dirs
- README is a short stub — fuller docs come in Phase 6
- Commit: `chore: scaffold repo`

**🛑 CHECKPOINT 1.** Stop. Show the user the tree and the seeded
`models.yaml`. Wait for approval.

---

### Phase 2 — Build the test application

- Implement the chosen domain as `app/<domain>.py`, single file, target
  150–250 lines
- 10–20 public functions covering meaningful operations in the domain
- Comprehensive `tests/` covering happy paths, edge cases (negatives,
  zeros, unicode, boundary values, locale variations as applicable), and
  error conditions
- `pytest` on the clean state must produce **zero failures**
- Commit: `feat: implement <domain> test application`

**🛑 CHECKPOINT 2.** Stop. Show the user the function list and pytest
output. Wait for approval. The user will likely run `pytest` themselves
to confirm.

---

### Phase 3 — Generate the corpus

Create 10–15 cases. For each case:

1. Pick a target function in the test app
2. Author a realistic bug — off-by-one, swapped operands, wrong default,
   missing edge case, incorrect type coercion, etc.
3. Write a focused test that catches the bug
4. Capture both as unified-diff patches relative to the clean state
5. Write a prompt that describes the problem **without giving away the
   bug**. The prompt should reference the failing test by path.
6. Save the reference fix (the inverse of the breaking patch) for diff-size
   comparison only — **never shown to models**

**Vary difficulty.** Include:
- 3–4 "obvious" bugs (wrong constant, simple typo)
- 5–7 "moderate" bugs (boundary conditions, edge cases)
- 2–3 "subtle" bugs (require reasoning across functions, or about real
  domain semantics — e.g. banker's rounding vs. half-up)
- 1 adversarial case: an impossible / ill-specified task where the
  correct answer is to push back rather than produce code. Graded by
  presence of a refusal phrase plus absence of file blocks.

Case file schema (one YAML per case):

```yaml
id: "001"
title: "round_to_cents fails on negative amounts"
difficulty: moderate           # obvious | moderate | subtle | adversarial
target_file: app/money.py
test_command: "pytest tests/test_round_negative.py -x"
prompt: |
  The test in tests/test_round_negative.py is failing. Read the failing
  test and the target file (app/money.py) and produce a corrected
  version of app/money.py. Respond with a single fenced code block
  using the required path-comment format. Do not modify the test.
breaking_patch: |
  <unified diff applied to clean state>
test_patch: |
  <unified diff that adds the failing test>
reference_patch: |
  <unified diff representing the canonical fix — NEVER shown to models>
```

Commit: `feat: add corpus of N cases`

**🛑 CHECKPOINT 3.** Stop. Show the user the case index (id, title,
difficulty). The user will spot-check 2–3 cases. Wait for approval.

---

### Phase 4 — Build the harness

Four modules:

**`harness/ollama_client.py`** — thin HTTP wrapper around
`POST /api/chat`. Single-message, system + user. Returns
`(content, prompt_eval_count, eval_count, total_duration_ns)`.
`temperature=0`, `num_ctx=16384`. Per-request timeout: 300s.

**`harness/scorer.py`** —
1. `extract_blocks(text) -> dict[path, content]` parses fenced blocks with
   the required `# path:` header. Tolerate the comment in `//`, `#`, or
   `--` form. Reject blocks without a path header.
2. `apply_blocks(worktree, blocks)` writes them to the worktree.
3. `run_tests(worktree, test_command) -> TestResult` runs pytest and
   captures pass/fail of the target test plus the full pytest exit code
   for regression detection (a clean pass needs both: target passes AND
   no other test newly fails relative to the broken+test-patch baseline).

**`harness/run.py`** — for each (model, case, attempt) where
`attempt ∈ {1..N}` (default N=3):
1. `git worktree add /tmp/llm-bench/<run_id>/<case_id>-<model>-<n>
    <base_sha>`
2. Apply `breaking_patch`, then `test_patch`. Sanity-check: target test
   must now fail. If it doesn't, abort the case with a clear error.
3. Build prompt: system prompt explaining the edit format + the case's
   `prompt`. Read the target file's current content and the failing
   test's content into the user message.
4. Call Ollama. Capture content + token counts + wall time.
5. `extract_blocks` → `apply_blocks` → `run_tests`.
6. Record row to `results/<run_id>.csv`:
   `run_id, model, case_id, attempt, schema_ok, target_passed,
    regressions, latency_ms, prompt_tokens, completion_tokens,
    reference_diff_lines, model_diff_lines, error`
7. `git worktree remove --force` the scratch dir.

Stream a one-line status per case to stdout so a long run is observable.
Write the CSV incrementally so a crash doesn't lose progress.

**`harness/report.py`** — load a results CSV, emit `REPORT.md` with:
- Per-model summary table: cases attempted, pass rate, mean
  regressions, median latency, mean tokens/sec
- Per-case difficulty breakdown: which models passed which case
- Pareto table: pass rate × speed
- Top 5 "interesting" rows (e.g. fastest model that still passed all
  obvious cases; biggest gap between difficulty tiers)

Commit: `feat: build harness`

**🛑 CHECKPOINT 4.** Stop. Show the user the harness module layout and
key function signatures. Wait for approval.

---

### Phase 5 — Smoke test

Run **one** model against **one** case end-to-end:

```bash
python -m harness.run --models qwen3:4b --cases 001 --attempts 1
```

The expected outcome is that the harness completes a clean loop — not
necessarily that the model passes the case. Print the captured model
output verbatim so the user can verify the edit format is being honored.
If `qwen3:4b` can't produce parseable blocks, repeat with `qwen2.5:14b`.

**🛑 CHECKPOINT 5.** Stop. Show the user the captured output, the parse
result, the test result, and the CSV row. Wait for approval before
running the full matrix.

---

### Phase 6 — Full run

```bash
python -m harness.run --all --attempts 3
```

Expected workload: 6 models × ~12 cases × 3 attempts ≈ 216 invocations.
On javelin's CPU-only inference, the 30B and 23GB models will dominate
wall time. Likely overnight. Stream progress; the user can tail the CSV.

Write a richer README in this phase: what the benchmark is, how to add
cases, how to add a model to `models.yaml`, how to interpret the report.

Commit: `docs: full README` and `chore: results from initial run`
(results CSV stays gitignored; a sanitized summary can be committed).

---

### Phase 7 — Analysis

Run `python -m harness.report results/<run_id>.csv > REPORT.md`. Read it.
Surface anything surprising to the user — unexpectedly strong showings,
models that failed the schema gate entirely (likely the 4B), tiers that
cluster together.

If a case looks broken (every model passes or every model fails for a
weird reason), flag it for revision rather than treating the data as
gospel.

---

## Conventions for Claude Code

- **Pause at every 🛑 checkpoint.** Do not chain phases.
- Before each phase, **summarize what you're about to do** in 3–5 lines
  and wait for "go".
- **Commit between phases**, with descriptive messages.
- **Surface decisions** you'd otherwise make silently (dependency choice,
  naming convention, error-handling philosophy). The user prefers to
  ratify these.
- **Verify each phase** before declaring it done — pytest passes,
  smoke run completes, etc.
- **No external services.** Everything offline on javelin.
- **No CI, no Dockerfile, no fancy tooling.** Plain Python, plain git.
- **Don't refactor the test application** mid-corpus. Once Phase 2 is
  approved, `app/` is frozen as the canonical clean state; bugs are
  introduced only via case patches.

---

## Reference: models on javelin

```
laguna-xs.2:latest    23 GB
qwen3-coder:30b       18 GB
qwen2.5:14b            9.0 GB
qwen3.6:latest        23 GB
gemma4:latest          9.6 GB
qwen3:4b               2.5 GB
```

Ollama endpoint: `http://localhost:11434`. Verify with
`curl http://localhost:11434/api/version` before Phase 5.

Default `models.yaml` order: smallest first (qwen3:4b) to catch
harness bugs quickly. The 4B is expected to fail most/all cases and
serves as the floor sanity check.

---

## Cost guardrail (re-stated)

Claude Code is the orchestrator. The inner benchmarking loop calls
**local Ollama only**. The full Phase 6 run should consume near-zero
Claude tokens — at most one tool call to kick off `run.py` and another
to read the CSV. If you find yourself routing the per-case prompt
through Claude Code, **stop and re-read this section**.
