"""CSV → REPORT.md aggregator.

CLI:
    python -m harness.report <results.csv> [--out REPORT.md]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _b(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _i(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


def render(rows: list[dict]) -> str:
    if not rows:
        return "# llm-bench report\n\n(no rows)\n"

    by_model: dict[str, list[dict]] = defaultdict(list)
    by_model_case: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
        by_case[r["case_id"]].append(r)
        by_model_case[(r["model"], r["case_id"])].append(r)

    def clean_pass(r: dict) -> bool:
        return _b(r["target_passed"]) and _i(r["regressions"]) == 0

    out: list[str] = []
    out.append("# llm-bench report")
    out.append("")
    out.append(f"- rows: {len(rows)}")
    out.append(f"- models: {len(by_model)}")
    out.append(f"- cases: {len(by_case)}")
    out.append("")

    out.append("## Per-model summary")
    out.append("")
    out.append("| model | attempts | pass-rate | mean regressions | median latency (ms) | mean tok/s |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for model, rs in sorted(by_model.items()):
        passed = sum(1 for r in rs if clean_pass(r))
        regs = [_i(r["regressions"]) for r in rs]
        lats = [_i(r["latency_ms"]) for r in rs if _i(r["latency_ms"]) > 0]
        tok_per_s = [
            _i(r["completion_tokens"]) / (_i(r["latency_ms"]) / 1000.0)
            for r in rs
            if _i(r["latency_ms"]) > 0 and _i(r["completion_tokens"]) > 0
        ]
        med_lat = int(statistics.median(lats)) if lats else 0
        tok_str = f"{statistics.mean(tok_per_s):.1f}" if tok_per_s else "-"
        out.append(
            f"| `{model}` | {len(rs)} | "
            f"{passed}/{len(rs)} ({100*passed/len(rs):.0f}%) | "
            f"{statistics.mean(regs):.2f} | {med_lat} | {tok_str} |"
        )
    out.append("")

    out.append("## Per-case results (best of N attempts)")
    out.append("")
    models = sorted(by_model.keys())
    out.append("| case | difficulty | " + " | ".join(f"`{m}`" for m in models) + " |")
    out.append("|---|---|" + "|".join("---" for _ in models) + "|")
    # difficulty isn't in csv; pull from anywhere we can find it (later: load cases)
    for case_id in sorted(by_case.keys()):
        cells = []
        for m in models:
            rs = by_model_case.get((m, case_id), [])
            if not rs:
                cells.append("·")
                continue
            best = any(clean_pass(r) for r in rs)
            any_pass = any(_b(r["target_passed"]) for r in rs)
            if best:
                cells.append("✅")
            elif any_pass:
                cells.append("⚠")  # passed target but caused regressions
            else:
                cells.append("❌")
        out.append(f"| {case_id} | - | " + " | ".join(cells) + " |")
    out.append("")

    out.append("## Pareto: pass-rate × median latency")
    out.append("")
    out.append("| model | pass-rate | median latency (ms) |")
    out.append("|---|---:|---:|")
    rows_pareto = []
    for model, rs in by_model.items():
        passed = sum(1 for r in rs if clean_pass(r))
        rate = passed / len(rs) if rs else 0
        lats = [_i(r["latency_ms"]) for r in rs if _i(r["latency_ms"]) > 0]
        med = int(statistics.median(lats)) if lats else 0
        rows_pareto.append((model, rate, med))
    for model, rate, med in sorted(rows_pareto, key=lambda x: (-x[1], x[2])):
        out.append(f"| `{model}` | {rate*100:.0f}% | {med} |")
    out.append("")

    out.append("## Errors")
    out.append("")
    err_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("error"):
            kind = r["error"].split(":", 1)[0]
            err_counts[kind] += 1
    if err_counts:
        for kind, n in sorted(err_counts.items(), key=lambda x: -x[1]):
            out.append(f"- `{kind}`: {n}")
    else:
        out.append("- (none)")
    out.append("")

    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    rows = load_rows(args.csv)
    text = render(rows)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
