"""Block extraction, application, and pytest scoring."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Matches ```<optional lang>\n# path: <relpath>\n<body>\n```
# Comment forms tolerated: '#', '//', '--' (Python, C-like, SQL/Lua).
# Lang tag is optional and ignored.
_FENCE_RE = re.compile(
    r"```[^\n]*\n"
    r"(?:#|//|--)\s*path\s*:\s*(?P<path>[^\n]+)\n"
    r"(?P<body>.*?)"
    r"\n?```",
    re.DOTALL,
)

# Disallow path traversal and absolute paths. Repo-relative only.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")

# Phrases that signal the model pushed back on an ill-specified task.
# Used only for adversarial cases, and only when no file blocks were emitted.
_REFUSAL_RE = re.compile(
    r"cannot|can't|can not|unable|not possible|impossible"
    r"|no exchange rates?|exchange rates? (?:are|is) (?:not|n't|missing|required|needed)"
    r"|would (?:need|require)|requires? (?:an? |the )?exchange rate"
    r"|without (?:an? |the )?exchange rate|not provided|not specified"
    r"|refuse|decline|ill-specified|under-?specified|clarif",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TestResult:
    target_passed: bool
    regressions: int  # tests that newly fail relative to baseline
    pytest_exit_code: int
    stdout: str


def extract_blocks(text: str) -> dict[str, str]:
    """Parse all fenced blocks with a `path:` header.

    Returns mapping of relpath -> file body. Raises ValueError if the text
    contains no parseable block or if a path is unsafe. If the same path
    appears more than once (small models often restate the file), the last
    occurrence wins — the benchmark measures coding ability, not repetition
    quirks.
    """
    blocks: dict[str, str] = {}
    for m in _FENCE_RE.finditer(text):
        path = m.group("path").strip()
        if not _SAFE_PATH_RE.match(path) or path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"unsafe path in block: {path!r}")
        blocks[path] = m.group("body")
    if not blocks:
        raise ValueError("no fenced blocks with `# path:` header found")
    return blocks


def grade_refusal(text: str) -> bool:
    """Grade an adversarial-case response: True iff the model pushed back.

    A correct response refuses (contains a refusal phrase) AND emits no
    fenced code at all — not just no *parseable* blocks. A fence without a
    `# path:` header or with an unsafe path is still emitted code, i.e.
    compliance with the ill-specified task, i.e. a fail.
    """
    if "```" in text:
        return False
    return bool(_REFUSAL_RE.search(text))


def apply_blocks(
    worktree: Path,
    blocks: dict[str, str],
    allowed_prefixes: tuple[str, ...] = ("app/",),
) -> tuple[list[str], list[str]]:
    """Write each block's body to worktree/<path>, overwriting.

    Only paths under `allowed_prefixes` are written — this blocks edits to
    tests/ and also conftest.py / pytest.ini style files that could
    monkeypatch the suite into passing. Returns (written, blocked) paths.
    """
    written: list[str] = []
    blocked: list[str] = []
    for relpath, body in blocks.items():
        norm = relpath[2:] if relpath.startswith("./") else relpath
        if not any(norm.startswith(p) for p in allowed_prefixes):
            blocked.append(norm)
            continue
        dest = worktree / norm
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not body.endswith("\n"):
            body = body + "\n"
        dest.write_text(body)
        written.append(norm)
    return written, blocked


def _pytest_env(worktree: Path) -> dict[str, str]:
    env = {**os.environ}
    env["PYTHONPATH"] = str(worktree)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _drop_pycache(root: Path) -> None:
    for d in root.rglob("__pycache__"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def _run_pytest(
    args: list[str],
    worktree: Path,
    pytest_bin: Path,
    timeout_s: int,
) -> tuple[int, str]:
    _drop_pycache(worktree)
    try:
        r = subprocess.run(
            [str(pytest_bin), *args, "--no-header", "-q", "-rfE", "--tb=short"],
            cwd=worktree,
            capture_output=True,
            text=True,
            env=_pytest_env(worktree),
            timeout=timeout_s,
        )
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f"<timeout after {timeout_s}s>\n{e.stdout or ''}\n{e.stderr or ''}"


# FAILED for test failures, ERROR for collection/import errors — a model edit
# that breaks an import must count as a regression, not vanish.
_FAILED_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def _failed_node_ids(output: str) -> set[str]:
    return set(_FAILED_LINE_RE.findall(output))


def run_tests(
    worktree: Path,
    target_test_file: str,
    baseline_failures: set[str],
    *,
    pytest_bin: Path,
    target_timeout_s: int = 60,
    regression_timeout_s: int = 120,
) -> TestResult:
    """Run the target test, then full suite. Target must pass AND no NEW failures.

    `baseline_failures` is the set of failing node ids in the
    broken+test_patch state (computed once per case). Regressions = tests
    that fail now but did not fail in baseline.
    """
    rc_target, out_target = _run_pytest(
        [target_test_file, "-x"],
        worktree,
        pytest_bin,
        target_timeout_s,
    )
    target_passed = rc_target == 0

    rc_full, out_full = _run_pytest(
        [],
        worktree,
        pytest_bin,
        regression_timeout_s,
    )
    current_failures = _failed_node_ids(out_full)
    regressions = len(current_failures - baseline_failures)

    return TestResult(
        target_passed=target_passed,
        regressions=regressions,
        pytest_exit_code=rc_full,
        stdout=out_target + "\n--- full suite ---\n" + out_full,
    )


def collect_baseline_failures(
    worktree: Path,
    pytest_bin: Path,
    timeout_s: int = 120,
) -> set[str]:
    """Run the full suite once in the broken+test_patch state, collect FAILED ids."""
    _, out = _run_pytest([], worktree, pytest_bin, timeout_s)
    return _failed_node_ids(out)
