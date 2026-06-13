"""Tests for harness/run.py helpers (no Ollama required)."""
import pytest

from harness import run


class TestParseTestPath:
    def test_extracts_new_file_path(self):
        patch = "--- /dev/null\n+++ b/tests/test_001_x.py\n@@ -0,0 +1 @@\n+pass\n"
        assert run.parse_test_path_from_patch(patch) == "tests/test_001_x.py"

    def test_missing_header_raises(self):
        with pytest.raises(ValueError):
            run.parse_test_path_from_patch("not a diff")


class TestDiffCounters:
    def test_diff_line_count_counts_both_sides(self):
        assert run.diff_line_count("one\ntwo\nthree", "one\nTWO\nthree") == 2

    def test_diff_line_count_identical(self):
        assert run.diff_line_count("same", "same") == 0

    def test_patch_diff_line_count_excludes_headers(self):
        patch = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y\n"
        assert run.patch_diff_line_count(patch) == 2


class TestSlugify:
    def test_replaces_tag_separators(self):
        assert run.slugify("qwen2.5:7b") == "qwen2.5-7b"
        assert run.slugify("library/model:tag") == "library_model-tag"


class TestClassifyStatus:
    @pytest.mark.parametrize(
        "row,expected",
        [
            ({"error": "", "target_passed": True, "regressions": 0}, "PASS"),
            ({"error": "", "target_passed": True, "regressions": 2}, "REGRESS"),
            ({"error": "", "target_passed": False, "regressions": 0}, "FAIL"),
            ({"error": "timeout:300s", "target_passed": False, "regressions": 0}, "TIMEOUT"),
            ({"error": "truncated:x", "target_passed": False, "regressions": 0}, "TRUNC"),
            ({"error": "refused:prose reply", "target_passed": False, "regressions": 0}, "REFUSED"),
            ({"error": "parse_error:x", "target_passed": False, "regressions": 0}, "PARSE_ERR"),
            ({"error": "infra_error:x", "target_passed": False, "regressions": 0}, "INFRA"),
        ],
    )
    def test_status_mapping(self, row, expected):
        assert run.classify_status(row) == expected


class TestLoadCases:
    def test_loads_full_corpus(self):
        cases = run.load_cases(None)
        assert len(cases) == 15
        assert [c.id for c in cases] == [f"{i:03d}" for i in range(1, 16)]

    def test_data_analysis_cases_target_analysis_module(self):
        for case in run.load_cases(["013", "014", "015"]):
            assert case.category == "data-analysis"
            assert case.target_file == "app/analysis.py"

    def test_adversarial_case_has_no_patches(self):
        (case,) = run.load_cases(["012"])
        assert case.difficulty == "adversarial"
        assert case.grading == "refusal"
        assert case.breaking_patch == ""
        assert not case.reference_patch

    def test_every_case_declares_category_and_grading(self):
        for case in run.load_cases(None):
            assert case.category in run.CATEGORIES, case.id
            assert case.grading in run.GRADINGS, case.id

    def test_builder_vocab_matches_harness(self):
        # cases/_build.py keeps its own copies so it stays standalone;
        # this pin is what "keep in sync" leans on.
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "case_builder", run.REPO / "cases" / "_build.py"
        )
        builder = importlib.util.module_from_spec(spec)
        # dataclass creation inside the module needs it registered
        sys.modules["case_builder"] = builder
        try:
            spec.loader.exec_module(builder)
            assert builder.CATEGORIES == run.CATEGORIES
            assert builder.GRADINGS == run.GRADINGS
        finally:
            del sys.modules["case_builder"]

    def test_pytest_graded_cases_have_reference_patches(self):
        for case in run.load_cases(None):
            if case.grading == "pytest":
                assert case.reference_patch, case.id

    def test_unknown_case_id_exits(self):
        with pytest.raises(SystemExit):
            run.load_cases(["999"])

    def test_invalid_grading_value_exits(self, tmp_path, monkeypatch):
        (tmp_path / "099.yaml").write_text(
            "id: '099'\ntitle: t\ndifficulty: obvious\ncategory: coding\n"
            "grading: pytests\ntarget_file: app/money.py\nprompt: p\n"
            "breaking_patch: ''\ntest_patch: ''\nreference_patch: ''\n"
        )
        monkeypatch.setattr(run, "CASES_DIR", tmp_path)
        with pytest.raises(SystemExit, match="invalid category/grading"):
            run.load_cases(None)

    def test_stale_yaml_without_new_fields_exits(self, tmp_path, monkeypatch):
        (tmp_path / "099.yaml").write_text(
            "id: '099'\ntitle: t\ndifficulty: obvious\n"
            "target_file: app/money.py\nprompt: p\n"
            "breaking_patch: ''\ntest_patch: ''\nreference_patch: ''\n"
        )
        monkeypatch.setattr(run, "CASES_DIR", tmp_path)
        with pytest.raises(SystemExit, match="regenerate"):
            run.load_cases(None)

    def test_every_test_patch_yields_a_test_path(self):
        for case in run.load_cases(None):
            path = run.parse_test_path_from_patch(case.test_patch)
            assert path.startswith("tests/")


class TestLoadModels:
    def test_loads_models_yaml(self):
        models = run.load_models(None)
        assert len(models) > 0

    def test_unknown_model_exits(self):
        with pytest.raises(SystemExit):
            run.load_models(["no-such-model:1b"])
