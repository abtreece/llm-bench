"""Per-(model, case, attempt) benchmark driver.

CLI:
    python -m harness.run [--models MODEL ...] [--cases CASE_ID ...]
                          [--attempts N] [--run-id ID] [--keep-worktrees]
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import difflib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from harness import ollama_client, scorer


REPO = Path(__file__).resolve().parents[1]
CASES_DIR = REPO / "cases"
MODELS_YAML = REPO / "models.yaml"
RESULTS_DIR = REPO / "results"
WORKTREE_ROOT = Path("/tmp/llm-bench")
PYTEST_BIN = REPO / ".venv" / "bin" / "pytest"

SYSTEM_PROMPT = """\
You are a coding assistant fixing a Python source file.

Output format (MANDATORY):
- Reply with one or more fenced code blocks, no prose between or after.
- Each block MUST start with a path comment on the line immediately after the
  opening fence, using one of these forms:
      # path: <relative/path>
      // path: <relative/path>
      -- path: <relative/path>
- Each block MUST contain the COMPLETE replacement contents of that file.
  Do not emit diffs or partial snippets.
- Do not modify any test file.

Example:
```python
# path: app/example.py
<complete file contents here>
```
"""

USER_PROMPT_TEMPLATE = """\
{case_prompt}

The failing test is at {test_path}:

```python
{test_source}
```

The current (broken) contents of {target_path}:

```python
{target_source}
```

Produce the corrected {target_path} using the required edit format. Do not modify {test_path}.
"""


CSV_FIELDS = [
    "run_id",
    "model",
    "case_id",
    "attempt",
    "schema_ok",
    "target_passed",
    "regressions",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "reference_diff_lines",
    "model_diff_lines",
    "error",
]


@dataclass
class Case:
    id: str
    title: str
    difficulty: str
    target_file: str
    test_command: str
    prompt: str
    breaking_patch: str
    test_patch: str
    reference_patch: str | None


def load_models(only: list[str] | None) -> list[str]:
    data = yaml.safe_load(MODELS_YAML.read_text())
    names = [m["name"] for m in data["models"]]
    if only:
        wanted = set(only)
        missing = wanted - set(names)
        if missing:
            raise SystemExit(f"unknown models: {sorted(missing)}")
        return [n for n in names if n in wanted]
    return names


def load_cases(only: list[str] | None) -> list[Case]:
    cases: list[Case] = []
    for p in sorted(CASES_DIR.glob("*.yaml")):
        d = yaml.safe_load(p.read_text())
        cases.append(
            Case(
                id=d["id"],
                title=d["title"],
                difficulty=d["difficulty"],
                target_file=d["target_file"],
                test_command=d["test_command"],
                prompt=d["prompt"],
                breaking_patch=d["breaking_patch"],
                test_patch=d["test_patch"],
                reference_patch=d.get("reference_patch"),
            )
        )
    if only:
        wanted = set(only)
        cases = [c for c in cases if c.id in wanted]
        missing = wanted - {c.id for c in cases}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")
    return cases


def git_apply(patch: str, cwd: Path) -> None:
    r = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input=patch,
        text=True,
        cwd=cwd,
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git apply failed: {r.stderr}")


def slugify(s: str) -> str:
    return s.replace(":", "-").replace("/", "_")


def parse_test_path_from_patch(test_patch: str) -> str:
    """Pull the 'b/<path>' from a unified diff like '+++ b/tests/foo.py'."""
    for line in test_patch.splitlines():
        if line.startswith("+++ b/"):
            return line[len("+++ b/"):].strip()
    raise ValueError("could not find +++ b/<path> in test_patch")


def diff_line_count(a: str, b: str) -> int:
    """Count of changed lines (added+removed) in a unified diff a→b."""
    diff = difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="")
    n = 0
    for line in diff:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            n += 1
    return n


def setup_worktree(dest: Path) -> None:
    """git worktree add at dest pointing at HEAD."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["git", "worktree", "add", "--detach", str(dest), "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"worktree add failed: {r.stderr}")


def remove_worktree(dest: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(dest)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def status_line(model: str, case_id: str, attempt: int, status: str, ms: int) -> str:
    return f"[{model:>22}] case {case_id} attempt {attempt} → {status} ({ms} ms)"


def run_one(
    run_id: str,
    model: str,
    case: Case,
    attempt: int,
    csv_writer: csv.DictWriter,
    csv_file,
    artifacts_dir: Path,
    keep_worktrees: bool,
) -> dict:
    work = WORKTREE_ROOT / run_id / f"{case.id}-{slugify(model)}-{attempt}"
    row = {f: "" for f in CSV_FIELDS}
    row.update(
        run_id=run_id,
        model=model,
        case_id=case.id,
        attempt=attempt,
        schema_ok=False,
        target_passed=False,
        regressions=0,
        latency_ms=0,
        prompt_tokens=0,
        completion_tokens=0,
        reference_diff_lines=0,
        model_diff_lines=0,
        error="",
    )
    try:
        setup_worktree(work)
        git_apply(case.breaking_patch, work)
        git_apply(case.test_patch, work)

        baseline_failures = scorer.collect_baseline_failures(work, PYTEST_BIN)
        test_path = parse_test_path_from_patch(case.test_patch)
        if not baseline_failures:
            row["error"] = "baseline_clean"
            csv_writer.writerow(row); csv_file.flush()
            return row

        target_path = work / case.target_file
        target_source = target_path.read_text()
        test_source = (work / test_path).read_text()

        if case.reference_patch:
            clean_target = target_source
            git_apply(case.reference_patch, work)
            ref_source = target_path.read_text()
            git_apply(_reverse_patch(case.reference_patch), work)
            row["reference_diff_lines"] = diff_line_count(clean_target, ref_source)

        user_msg = USER_PROMPT_TEMPLATE.format(
            case_prompt=case.prompt,
            test_path=test_path,
            test_source=test_source,
            target_path=case.target_file,
            target_source=target_source,
        )

        t0 = time.monotonic()
        try:
            result = ollama_client.chat(model, SYSTEM_PROMPT, user_msg)
        except Exception as e:
            row["error"] = f"infra_error:{type(e).__name__}:{e}"
            row["latency_ms"] = int((time.monotonic() - t0) * 1000)
            csv_writer.writerow(row); csv_file.flush()
            return row
        latency_ms = int((time.monotonic() - t0) * 1000)
        row["latency_ms"] = latency_ms
        row["prompt_tokens"] = result.prompt_eval_count
        row["completion_tokens"] = result.eval_count

        attempt_artifacts = artifacts_dir / model.replace(":", "-").replace("/", "_") / case.id / str(attempt)
        attempt_artifacts.mkdir(parents=True, exist_ok=True)
        (attempt_artifacts / "response.txt").write_text(result.content)

        try:
            blocks = scorer.extract_blocks(result.content)
        except ValueError as e:
            row["error"] = f"parse_error:{e}"
            csv_writer.writerow(row); csv_file.flush()
            return row

        row["schema_ok"] = True
        scorer.apply_blocks(work, blocks)

        new_target = (work / case.target_file).read_text() if (work / case.target_file).exists() else ""
        row["model_diff_lines"] = diff_line_count(target_source, new_target)

        for relpath in blocks:
            (attempt_artifacts / f"file_{relpath.replace('/', '_')}").write_text(blocks[relpath])

        tr = scorer.run_tests(work, test_path, baseline_failures, pytest_bin=PYTEST_BIN)
        (attempt_artifacts / "pytest.txt").write_text(tr.stdout)
        row["target_passed"] = tr.target_passed
        row["regressions"] = tr.regressions

    except Exception as e:
        row["error"] = f"harness_error:{type(e).__name__}:{e}"
    finally:
        if not keep_worktrees:
            remove_worktree(work)

    csv_writer.writerow(row); csv_file.flush()
    return row


def _reverse_patch(patch: str) -> str:
    """Flip a unified diff so applying it reverts the original. Naive but
    sufficient for the patches we author (no rename/binary)."""
    out = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("--- "):
            out.append("+++ " + line[4:])
        elif line.startswith("+++ "):
            out.append("--- " + line[4:])
        elif line.startswith("+") and not line.startswith("+++"):
            out.append("-" + line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            out.append("+" + line[1:])
        else:
            out.append(line)
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run llm-bench against local Ollama.")
    p.add_argument("--models", nargs="*", help="restrict to these model tags")
    p.add_argument("--cases", nargs="*", help="restrict to these case ids")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--run-id", default=None)
    p.add_argument("--keep-worktrees", action="store_true")
    args = p.parse_args(argv)

    if not PYTEST_BIN.exists():
        raise SystemExit(f"pytest not found at {PYTEST_BIN} — set up .venv first")

    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    models = load_models(args.models)
    cases = load_cases(args.cases)

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"{run_id}.csv"
    artifacts_dir = RESULTS_DIR / run_id
    artifacts_dir.mkdir(exist_ok=True)

    print(f"# run_id={run_id}")
    print(f"# models={len(models)} cases={len(cases)} attempts={args.attempts}")
    print(f"# csv={csv_path}")
    print(f"# artifacts={artifacts_dir}")

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for model in models:
            for case in cases:
                for attempt in range(1, args.attempts + 1):
                    row = run_one(
                        run_id, model, case, attempt, w, fh,
                        artifacts_dir, args.keep_worktrees,
                    )
                    status = (
                        "PASS" if row["target_passed"] and row["regressions"] == 0
                        else ("REGRESS" if row["target_passed"] else
                              ("PARSE_ERR" if not row["schema_ok"] and row["error"].startswith("parse_error")
                               else ("INFRA" if row["error"].startswith("infra_error") else "FAIL")))
                    )
                    print(status_line(model, case.id, attempt, status, row["latency_ms"]))
    print(f"done. csv -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
