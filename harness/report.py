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

import yaml

REPO = Path(__file__).resolve().parents[1]
CASES_DIR = REPO / "cases"
MODELS_YAML = REPO / "models.yaml"


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


def load_case_meta() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for p in sorted(CASES_DIR.glob("*.yaml")):
        d = yaml.safe_load(p.read_text())
        out[str(d["id"])] = {
            "difficulty": d.get("difficulty", "-"),
            "category": d.get("category", "-"),
        }
    return out


def load_model_sizes() -> dict[str, float]:
    # Missing file is fine (sizes render as "-"); a malformed one should
    # surface loudly rather than silently dropping the column.
    try:
        data = yaml.safe_load(MODELS_YAML.read_text())
    except FileNotFoundError:
        return {}
    return {m["name"]: float(m.get("size_gb", 0)) for m in data["models"]}


def _is_harness_row(r: dict) -> bool:
    """Rows that say nothing about the model: harness bugs and infra failures."""
    err = r.get("error", "") or ""
    return err.startswith("harness_error") or err.startswith("infra_error") or err == "baseline_clean"


def clean_pass(r: dict) -> bool:
    return _b(r["target_passed"]) and _i(r["regressions"]) == 0


def render(rows: list[dict]) -> str:
    if not rows:
        return "# llm-bench report\n\n(no rows)\n"

    excluded = [r for r in rows if _is_harness_row(r)]
    scored = [r for r in rows if not _is_harness_row(r)]

    by_model: dict[str, list[dict]] = defaultdict(list)
    by_model_case: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_model[r["model"]].append(r)
        by_case[r["case_id"]].append(r)
        by_model_case[(r["model"], r["case_id"])].append(r)

    case_meta = load_case_meta()
    sizes = load_model_sizes()

    out: list[str] = []
    out.append("# llm-bench report")
    out.append("")
    out.append(f"- rows: {len(rows)} ({len(scored)} scored, {len(excluded)} excluded as harness/infra)")
    out.append(f"- models: {len(by_model)}")
    out.append(f"- cases: {len(by_case)}")
    skipped_models = sorted({r["model"] for r in excluded} - set(by_model))
    if skipped_models:
        out.append(
            "- ⚠ models with no scored rows (all attempts excluded): "
            + ", ".join(f"`{m}`" for m in skipped_models)
        )
    out.append("")

    # Per-model stats, reused by summary + recommendation.
    stats: dict[str, dict] = {}
    for model, rs in by_model.items():
        passed = sum(1 for r in rs if clean_pass(r))
        regs = [_i(r["regressions"]) for r in rs]
        lats = [_i(r["latency_ms"]) for r in rs if _i(r["latency_ms"]) > 0]
        # tok/s from Ollama's eval_duration (pure generation), falling back to
        # wall latency for older CSVs without the eval_ms column.
        tok_per_s = []
        for r in rs:
            toks = _i(r["completion_tokens"])
            dur_ms = _i(r.get("eval_ms", "")) or _i(r["latency_ms"])
            if toks > 0 and dur_ms > 0:
                tok_per_s.append(toks / (dur_ms / 1000.0))
        parse_errs = sum(1 for r in rs if (r.get("error") or "").startswith("parse_error"))
        timeouts = sum(1 for r in rs if (r.get("error") or "").startswith("timeout"))
        # Attempt 1 is the greedy (temperature 0) sample; later attempts are
        # temperature 0.4, so pass@1 is reported separately from the pooled rate.
        first = [r for r in rs if str(r.get("attempt", "")).strip() == "1"]
        stats[model] = {
            "attempts": len(rs),
            "passed": passed,
            "rate": passed / len(rs) if rs else 0.0,
            "pass1": sum(1 for r in first if clean_pass(r)),
            "n1": len(first),
            "mean_regs": statistics.mean(regs) if regs else 0.0,
            "med_lat": int(statistics.median(lats)) if lats else 0,
            "tok_s": statistics.mean(tok_per_s) if tok_per_s else 0.0,
            "parse_errs": parse_errs,
            "timeouts": timeouts,
            "size_gb": sizes.get(model, 0.0),
        }

    out.append("## Per-model summary")
    out.append("")
    out.append("| model | size (GB) | attempts | pass@1 | pass-rate (all) | parse errs | timeouts | mean regressions | median latency | mean tok/s |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in sorted(stats):
        s = stats[model]
        lat = f"{s['med_lat']/1000:.0f}s" if s["med_lat"] else "-"
        size = f"{s['size_gb']:.1f}" if s["size_gb"] else "-"
        tok = f"{s['tok_s']:.1f}" if s["tok_s"] else "-"
        p1 = (
            f"{s['pass1']}/{s['n1']} ({100*s['pass1']/s['n1']:.0f}%)"
            if s["n1"] else "-"
        )
        out.append(
            f"| `{model}` | {size} | {s['attempts']} | {p1} | "
            f"{s['passed']}/{s['attempts']} ({100*s['rate']:.0f}%) | "
            f"{s['parse_errs']} | {s['timeouts']} | "
            f"{s['mean_regs']:.2f} | {lat} | {tok} |"
        )
    out.append("")

    out.append("## Per-case results (best of N attempts)")
    out.append("")
    models = sorted(by_model.keys())
    out.append("| case | category | difficulty | " + " | ".join(f"`{m}`" for m in models) + " |")
    out.append("|---|---|---|" + "|".join("---" for _ in models) + "|")
    for case_id in sorted(by_case.keys()):
        cells = []
        for m in models:
            rs = by_model_case.get((m, case_id), [])
            if not rs:
                cells.append("·")
                continue
            if any(clean_pass(r) for r in rs):
                cells.append("✅")
            elif any(_b(r["target_passed"]) for r in rs):
                cells.append("⚠")  # passed target but caused regressions
            else:
                cells.append("❌")
        meta = case_meta.get(case_id, {})
        out.append(
            f"| {case_id} | {meta.get('category', '-')} | {meta.get('difficulty', '-')} | "
            + " | ".join(cells) + " |"
        )
    out.append("")

    # Per-category pass-rates, so capability areas (coding vs data-analysis
    # vs ...) can be compared instead of one pooled number.
    categories = sorted({
        case_meta.get(cid, {}).get("category", "-") for cid in by_case
    })
    if len(categories) > 1:
        out.append("## Per-category pass-rate (all attempts)")
        out.append("")
        out.append("| model | " + " | ".join(categories) + " |")
        out.append("|---|" + "|".join("---:" for _ in categories) + "|")
        for m in models:
            cells = []
            for cat in categories:
                rs = [
                    r for r in by_model[m]
                    if case_meta.get(r["case_id"], {}).get("category", "-") == cat
                ]
                if not rs:
                    cells.append("-")
                    continue
                passed = sum(1 for r in rs if clean_pass(r))
                cells.append(f"{passed}/{len(rs)} ({100*passed/len(rs):.0f}%)")
            out.append(f"| `{m}` | " + " | ".join(cells) + " |")
        out.append("")

    out.append("## Pareto: pass-rate × median latency")
    out.append("")
    out.append("Models not listed are dominated (another model is both more accurate and faster).")
    out.append("")
    out.append("| model | pass-rate | median latency | on frontier |")
    out.append("|---|---:|---:|---|")
    ranked = sorted(stats.items(), key=lambda kv: (-kv[1]["rate"], kv[1]["med_lat"]))
    frontier: list[str] = []
    best_lat = None
    for model, s in ranked:
        # A model with no recorded latency can't claim a frontier spot —
        # there is nothing to trade off against.
        on = s["med_lat"] > 0 and (best_lat is None or s["med_lat"] < best_lat)
        if on:
            frontier.append(model)
            best_lat = s["med_lat"]
        lat = f"{s['med_lat']/1000:.0f}s" if s["med_lat"] else "-"
        out.append(
            f"| `{model}` | {100*s['rate']:.0f}% | {lat} | {'✅' if on else ''} |"
        )
    out.append("")

    out.append("## Recommendation")
    out.append("")
    if frontier:
        max_attempts = max(s["attempts"] for s in stats.values())
        best = frontier[0]
        caveat = (
            f" ⚠ only {stats[best]['attempts']}/{max_attempts} attempts completed — rerun before trusting this."
            if stats[best]["attempts"] < max_attempts else ""
        )
        out.append(
            f"- **Best accuracy:** `{best}` — "
            f"{100*stats[best]['rate']:.0f}% clean-pass at ~{stats[best]['med_lat']/1000:.0f}s/case.{caveat}"
        )
        # Sweet spot: the highest-rate frontier model after the best one
        # (strictly faster by construction) that stays within 15 points of
        # the best pass-rate.
        faster = [m for m in frontier[1:] if stats[m]["rate"] >= stats[best]["rate"] - 0.15]
        if faster:
            sweet = faster[0]
            out.append(
                f"- **Sweet spot:** `{sweet}` — within 15 points of the best "
                f"({100*stats[sweet]['rate']:.0f}%) at {stats[sweet]['med_lat']/1000:.0f}s/case "
                f"vs {stats[best]['med_lat']/1000:.0f}s."
            )
        unusable = [m for m, s in stats.items() if s["parse_errs"] >= s["attempts"] * 0.5]
        if unusable:
            out.append(
                "- **Not viable on this task:** "
                + ", ".join(f"`{m}`" for m in sorted(unusable))
                + " — ≥50% of attempts failed the output format."
            )
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
