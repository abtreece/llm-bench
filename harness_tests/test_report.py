"""Tests for harness/report.py — row classification and rendering."""
from harness import report


def make_row(model, case_id="001", attempt=1, passed=False, regs=0, lat=10000,
             err="", toks=100, eval_ms=5000):
    return {
        "run_id": "t",
        "model": model,
        "case_id": case_id,
        "attempt": str(attempt),
        "schema_ok": "True",
        "target_passed": str(passed),
        "regressions": str(regs),
        "latency_ms": str(lat),
        "eval_ms": str(eval_ms),
        "load_ms": "0",
        "prompt_tokens": "500",
        "completion_tokens": str(toks),
        "reference_diff_lines": "2",
        "model_diff_lines": "4",
        "blocked_paths": "",
        "done_reason": "stop",
        "error": err,
    }


class TestRowClassification:
    def test_clean_pass_requires_no_regressions(self):
        assert report.clean_pass(make_row("m", passed=True)) is True
        assert report.clean_pass(make_row("m", passed=True, regs=1)) is False
        assert report.clean_pass(make_row("m", passed=False)) is False

    def test_harness_rows_excluded(self):
        assert report._is_harness_row(make_row("m", err="infra_error:x")) is True
        assert report._is_harness_row(make_row("m", err="harness_error:x")) is True
        assert report._is_harness_row(make_row("m", err="baseline_clean")) is True

    def test_timeout_and_truncated_rows_are_scored(self):
        assert report._is_harness_row(make_row("m", err="timeout:300s")) is False
        assert report._is_harness_row(make_row("m", err="truncated:x")) is False
        assert report._is_harness_row(make_row("m", err="parse_error:x")) is False


class TestRender:
    def test_empty_rows(self):
        assert "(no rows)" in report.render([])

    def test_pass_at_1_reported_separately_from_pooled_rate(self):
        rows = [
            make_row("model-a", attempt=1, passed=True),
            make_row("model-a", attempt=2, passed=False),
            make_row("model-a", attempt=3, passed=False),
        ]
        text = report.render(rows)
        assert "1/1 (100%)" in text  # pass@1: greedy attempt passed
        assert "1/3 (33%)" in text   # pooled rate across temperatures

    def test_timeout_stays_in_denominator(self):
        rows = [
            make_row("model-b", attempt=1, err="timeout:300s", lat=300000),
            make_row("model-b", attempt=2, passed=True, lat=5000),
        ]
        text = report.render(rows)
        assert "1/2 (50%)" in text

    def test_skipped_model_surfaces_in_header(self):
        rows = [
            make_row("model-a", passed=True),
            make_row("model-c", case_id="*", attempt=0, lat=0,
                     err="infra_error:warmup_failed:ConnectionError:x"),
        ]
        text = report.render(rows)
        assert "no scored rows" in text
        assert "`model-c`" in text

    def test_zero_latency_model_not_on_frontier(self):
        rows = [
            make_row("model-d", passed=True, lat=0, eval_ms=0),
            make_row("model-a", passed=True, lat=10000),
        ]
        text = report.render(rows)
        frontier_section = text.split("## Pareto")[1].split("##")[0]
        for line in frontier_section.splitlines():
            if "`model-d`" in line:
                assert "✅" not in line
            if "`model-a`" in line:
                assert "✅" in line

    def test_per_category_breakdown_when_multiple_categories(self, monkeypatch):
        monkeypatch.setattr(report, "load_case_meta", lambda: {
            "001": {"difficulty": "obvious", "category": "coding"},
            "013": {"difficulty": "moderate", "category": "data-analysis"},
        })
        rows = [
            make_row("model-a", case_id="001", passed=True),
            make_row("model-a", case_id="013", passed=False),
        ]
        text = report.render(rows)
        assert "## Per-category pass-rate" in text
        assert "data-analysis" in text

    def test_no_category_breakdown_for_single_category(self, monkeypatch):
        monkeypatch.setattr(report, "load_case_meta", lambda: {
            "001": {"difficulty": "obvious", "category": "coding"},
        })
        text = report.render([make_row("model-a", case_id="001", passed=True)])
        assert "## Per-category pass-rate" not in text

    def test_blocked_write_attempts_counted_per_model(self):
        rows = [
            make_row("model-a", attempt=1, passed=True) | {"blocked_paths": "tests/test_x.py"},
            make_row("model-a", attempt=2, passed=True),
        ]
        text = report.render(rows)
        assert "blocked writes" in text
        summary_line = next(
            line for line in text.splitlines()
            if line.startswith("| `model-a`") and "GB" not in line
        )
        # ... | parse errs | timeouts | blocked writes | ...
        assert "| 0 | 0 | 1 |" in summary_line

    def test_bench_version_shown_and_mixed_versions_flagged(self):
        v2 = make_row("model-a", passed=True) | {"bench_version": "2"}
        v3 = make_row("model-a", attempt=2, passed=True) | {"bench_version": "3"}
        assert "bench_version: 2\n" in report.render([v2, v2])
        assert "not comparable" not in report.render([v2, v2])
        assert "bench_version: 2, 3" in report.render([v2, v3])
        assert "not comparable" in report.render([v2, v3])

    def test_rows_without_bench_version_render_quietly(self):
        text = report.render([make_row("model-a", passed=True)])
        assert "bench_version" not in text

    def test_error_kinds_counted(self):
        rows = [
            make_row("m", attempt=1, err="timeout:300s"),
            make_row("m", attempt=2, err="parse_error:no blocks"),
        ]
        text = report.render(rows)
        assert "`timeout`: 1" in text
        assert "`parse_error`: 1" in text
