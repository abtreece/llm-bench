"""Tests for harness/recommend.py — pure-function core, no subprocess/network."""
import pytest
import yaml

from harness import ollama_client, recommend


class TestModelsFromTags:
    def test_converts_bytes_to_decimal_gb(self):
        data = {"models": [{"name": "qwen2.5:14b", "size": 9_000_000_000}]}
        assert ollama_client.models_from_tags(data) == [
            {"name": "qwen2.5:14b", "size_gb": 9.0, "digest": "", "families": []}
        ]

    def test_empty_and_missing_models_key(self):
        assert ollama_client.models_from_tags({}) == []
        assert ollama_client.models_from_tags({"models": []}) == []

    def test_null_models_key(self):
        # Go marshals a nil slice as null, so {"models": null} is realistic.
        assert ollama_client.models_from_tags({"models": None}) == []

    def test_extracts_digest_and_families(self):
        data = {"models": [{
            "name": "nomic-embed-text:latest",
            "size": 274_000_000,
            "digest": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
            "details": {"family": "nomic-bert", "families": ["nomic-bert"]},
        }]}
        [m] = ollama_client.models_from_tags(data)
        assert m["digest"].startswith("0a109f422b47")
        assert m["families"] == ["nomic-bert"]

    def test_families_falls_back_to_family_singular(self):
        # Older payloads carry "family" with "families": null.
        data = {"models": [{
            "name": "llama3:8b", "size": 1,
            "details": {"family": "llama", "families": None},
        }]}
        assert ollama_client.models_from_tags(data)[0]["families"] == ["llama"]

    def test_missing_details_degrades_to_empty_families(self):
        data = {"models": [{"name": "x:latest", "size": 1}]}
        assert ollama_client.models_from_tags(data)[0]["families"] == []


class TestCatalog:
    def test_repo_catalog_loads_and_is_well_formed(self):
        entries = recommend.load_catalog()
        assert len(entries) >= 8
        for e in entries:
            assert isinstance(e["name"], str) and ":" in e["name"]
            assert e["size_gb"] > 0

    def test_missing_catalog_degrades_to_empty(self, tmp_path):
        assert recommend.load_catalog(tmp_path / "nope.yaml") == []


NVIDIA_ONE = "Tesla T4, 15360\n"
NVIDIA_TWO = "Tesla T4, 15360\nTesla T4, 15360\n"
ROCM_JSON = '{"card0": {"VRAM Total Memory (B)": "17163091968", "VRAM Total Used Memory (B)": "0"}}'
MEMINFO = "MemTotal:       65856380 kB\nMemFree:        1234 kB\n"
LSPCI_NVIDIA = "01:00.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)\n"
LSPCI_AMD = "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 21\n"
LSPCI_NONE = "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 630\n"


class TestParsers:
    def test_nvidia_smi_single_gpu(self):
        gpus = recommend.parse_nvidia_smi(NVIDIA_ONE)
        assert len(gpus) == 1
        assert gpus[0].name == "Tesla T4"
        assert abs(gpus[0].vram_gb - 16.1) < 0.1  # 15360 MiB

    def test_nvidia_smi_multi_gpu(self):
        gpus = recommend.parse_nvidia_smi(NVIDIA_TWO)
        assert len(gpus) == 2
        assert abs(sum(g.vram_gb for g in gpus) - 32.2) < 0.2

    def test_nvidia_smi_garbage(self):
        assert recommend.parse_nvidia_smi("NVIDIA-SMI has failed\n") == []
        assert recommend.parse_nvidia_smi("") == []

    def test_rocm_smi_happy_path(self):
        gpus = recommend.parse_rocm_smi(ROCM_JSON)
        assert len(gpus) == 1
        assert abs(gpus[0].vram_gb - 17.2) < 0.1

    def test_rocm_smi_missing_key(self):
        assert recommend.parse_rocm_smi('{"card0": {"other": "1"}}') == []
        assert recommend.parse_rocm_smi("not json") == []

    def test_meminfo(self):
        assert abs(recommend.parse_meminfo(MEMINFO) - 67.4) < 0.1

    def test_meminfo_garbage(self):
        assert recommend.parse_meminfo("nope") is None

    def test_sysctl_memsize(self):
        assert recommend.parse_sysctl_memsize("68719476736\n") == 68.719476736

    def test_sysctl_garbage(self):
        assert recommend.parse_sysctl_memsize("zzz") is None

    class TestLspciScan:
        def test_nvidia_hit(self):
            assert recommend.scan_lspci(LSPCI_NVIDIA) == "nvidia"

        def test_amd_hit(self):
            assert recommend.scan_lspci(LSPCI_AMD) == "amd"

        def test_no_discrete_gpu(self):
            assert recommend.scan_lspci(LSPCI_NONE) is None

        def test_empty(self):
            assert recommend.scan_lspci("") is None


class TestBuildHardware:
    def test_darwin_arm64_is_metal_unified_fraction(self):
        hw = recommend.build_hardware(
            "Darwin", "arm64", nvidia_out=None, rocm_out=None,
            lspci_out=None, ram_gb=64.0)
        assert hw.backend == "metal"
        assert hw.gpus == []
        assert abs(hw.budget_gb - 0.67 * 64.0) < 0.01

    def test_darwin_x86_is_cpu(self):
        hw = recommend.build_hardware(
            "Darwin", "x86_64", nvidia_out=None, rocm_out=None,
            lspci_out=None, ram_gb=32.0)
        assert hw.backend == "cpu"
        assert hw.budget_gb == 32.0 - 4.0

    def test_linux_nvidia(self):
        hw = recommend.build_hardware(
            "Linux", "x86_64", nvidia_out=NVIDIA_ONE, rocm_out=None,
            lspci_out=None, ram_gb=64.0)
        assert hw.backend == "cuda"
        assert abs(hw.budget_gb - 16.1) < 0.1

    def test_linux_nvidia_multi_gpu_sums(self):
        hw = recommend.build_hardware(
            "Linux", "x86_64", nvidia_out=NVIDIA_TWO, rocm_out=None,
            lspci_out=None, ram_gb=64.0)
        assert abs(hw.budget_gb - 32.2) < 0.2

    def test_linux_rocm(self):
        hw = recommend.build_hardware(
            "Linux", "x86_64", nvidia_out=None, rocm_out=ROCM_JSON,
            lspci_out=None, ram_gb=64.0)
        assert hw.backend == "rocm"
        assert abs(hw.budget_gb - 17.2) < 0.1

    def test_linux_nvidia_smi_garbage_falls_through(self):
        # nvidia-smi present but emitting garbage = driver gap, not cuda.
        hw = recommend.build_hardware(
            "Linux", "x86_64", nvidia_out="NVIDIA-SMI has failed\n",
            rocm_out=None, lspci_out=LSPCI_NVIDIA, ram_gb=64.0)
        assert hw.backend == "cpu"
        assert any("NVIDIA" in w for w in hw.warnings)

    def test_linux_driver_gap_warns_and_uses_cpu(self):
        hw = recommend.build_hardware(
            "Linux", "x86_64", nvidia_out=None, rocm_out=None,
            lspci_out=LSPCI_NVIDIA, ram_gb=64.0)
        assert hw.backend == "cpu"
        assert hw.budget_gb == 60.0
        assert any("NVIDIA" in w for w in hw.warnings)

    def test_linux_amd_driver_gap_warns(self):
        hw = recommend.build_hardware(
            "Linux", "x86_64", nvidia_out=None, rocm_out=None,
            lspci_out=LSPCI_AMD, ram_gb=64.0)
        assert hw.backend == "cpu"
        assert any("ROCm" in w for w in hw.warnings)

    def test_linux_no_lspci_notes_incomplete_detection(self):
        hw = recommend.build_hardware(
            "Linux", "x86_64", nvidia_out=None, rocm_out=None,
            lspci_out=None, ram_gb=64.0)
        assert hw.backend == "cpu"
        assert any("lspci" in w for w in hw.warnings)

    def test_low_ram_cpu_budget_clamps_to_zero(self):
        # 2 GB host: reserve exceeds RAM; budget must not go negative.
        hw = recommend.build_hardware(
            "Linux", "x86_64", nvidia_out=None, rocm_out=None,
            lspci_out=LSPCI_NONE, ram_gb=2.0)
        assert hw.budget_gb == 0.0

    def test_low_ram_darwin_x86_budget_clamps_to_zero(self):
        hw = recommend.build_hardware(
            "Darwin", "x86_64", nvidia_out=None, rocm_out=None,
            lspci_out=None, ram_gb=2.0)
        assert hw.budget_gb == 0.0

    def test_unsupported_os_raises(self):
        with pytest.raises(recommend.RecommendError):
            recommend.build_hardware("Windows", "AMD64", nvidia_out=None,
                                     rocm_out=None, lspci_out=None, ram_gb=8.0)

    def test_no_gpu_and_no_ram_raises(self):
        with pytest.raises(recommend.RecommendError):
            recommend.build_hardware("Linux", "x86_64", nvidia_out=None,
                                     rocm_out=None, lspci_out="", ram_gb=None)


T4_BUDGET = 16.1  # 15360 MiB


class TestFitRule:
    def test_t4_anchor_30b_excluded(self):
        assert recommend.fits(19.0, T4_BUDGET, 2.0) is False

    def test_t4_anchor_14b_admitted(self):
        assert recommend.fits(9.0, T4_BUDGET, 2.0) is True

    def test_boundary_is_inclusive(self):
        assert recommend.fits(14.0, 16.0, 2.0) is True
        assert recommend.fits(14.1, 16.0, 2.0) is False

    def test_headroom_override(self):
        assert recommend.fits(15.5, T4_BUDGET, 0.5) is True
        assert recommend.fits(15.5, T4_BUDGET, 2.0) is False


class TestTierSelection:
    INSTALLED = [
        {"name": "qwen2.5:14b", "size_gb": 9.0},
        {"name": "qwen3-coder:30b", "size_gb": 19.0},
    ]
    CATALOG = [
        {"name": "qwen2.5-coder:14b", "size_gb": 9.0},
        {"name": "qwen2.5-coder:32b", "size_gb": 20.0},
        {"name": "qwen2.5:14b", "size_gb": 9.0},  # also installed
    ]

    def test_tiers_on_t4(self):
        tiers = recommend.select_tiers(
            self.INSTALLED, self.CATALOG, budget_gb=T4_BUDGET, headroom_gb=2.0)
        assert [m.name for m in tiers.selected] == ["qwen2.5:14b"]
        assert [m.name for m in tiers.excluded] == ["qwen3-coder:30b"]
        # 32b doesn't fit, qwen2.5:14b already installed
        assert [m.name for m in tiers.worth_pulling] == ["qwen2.5-coder:14b"]

    def test_margin_is_budget_minus_size_minus_headroom(self):
        tiers = recommend.select_tiers(
            self.INSTALLED, self.CATALOG, budget_gb=T4_BUDGET, headroom_gb=2.0)
        assert abs(tiers.selected[0].margin_gb - (T4_BUDGET - 9.0 - 2.0)) < 0.01

    def test_exact_tag_match_only(self):
        installed = [{"name": "qwen2.5-coder:14b-instruct-q8_0", "size_gb": 15.0}]
        catalog = [{"name": "qwen2.5-coder:14b", "size_gb": 9.0}]
        tiers = recommend.select_tiers(installed, catalog,
                                       budget_gb=20.0, headroom_gb=2.0)
        assert [m.name for m in tiers.worth_pulling] == ["qwen2.5-coder:14b"]

    def test_empty_installed_still_suggests_catalog(self):
        tiers = recommend.select_tiers([], self.CATALOG,
                                       budget_gb=T4_BUDGET, headroom_gb=2.0)
        assert tiers.selected == [] and tiers.excluded == []
        assert len(tiers.worth_pulling) == 2

    def test_tiers_sorted_smallest_first(self):
        installed = [
            {"name": "big", "size_gb": 9.0},
            {"name": "small", "size_gb": 1.0},
        ]
        tiers = recommend.select_tiers(installed, [],
                                       budget_gb=16.0, headroom_gb=2.0)
        assert [m.name for m in tiers.selected] == ["small", "big"]

    def test_non_chat_models_dropped_from_all_tiers(self):
        installed = [
            {"name": "nomic-embed-text:latest", "size_gb": 0.27,
             "families": ["nomic-bert"]},
            {"name": "huge-embedder:latest", "size_gb": 99.0,
             "families": ["bert"]},
            {"name": "qwen2.5:14b", "size_gb": 9.0, "families": ["qwen2"]},
        ]
        tiers = recommend.select_tiers(installed, [],
                                       budget_gb=T4_BUDGET, headroom_gb=2.0)
        assert [m.name for m in tiers.selected] == ["qwen2.5:14b"]
        assert tiers.excluded == []

    def test_non_chat_filter_falls_back_to_name(self):
        # No families data (older Ollama): the name heuristic must catch it.
        installed = [{"name": "mxbai-embed-large:latest", "size_gb": 0.67}]
        tiers = recommend.select_tiers(installed, [],
                                       budget_gb=T4_BUDGET, headroom_gb=2.0)
        assert tiers.selected == []


class TestIsChatModel:
    def test_bert_families_are_not_chat(self):
        assert recommend.is_chat_model(
            {"name": "nomic-embed-text:latest", "families": ["nomic-bert"]}) is False
        assert recommend.is_chat_model(
            {"name": "mxbai-embed-large:latest", "families": ["bert"]}) is False

    def test_chat_family_is_chat(self):
        assert recommend.is_chat_model(
            {"name": "qwen2.5:14b", "families": ["qwen2"]}) is True

    def test_multimodal_families_are_chat(self):
        assert recommend.is_chat_model(
            {"name": "llava:7b", "families": ["llama", "clip"]}) is True

    def test_name_fallback_without_families(self):
        assert recommend.is_chat_model(
            {"name": "snowflake-arctic-embed:latest"}) is False
        assert recommend.is_chat_model({"name": "qwen2.5:14b"}) is True

    def test_embed_name_overrides_chat_family(self):
        # qwen3-embedding reports its base chat family but cannot chat.
        assert recommend.is_chat_model(
            {"name": "qwen3-embedding:0.6b", "families": ["qwen3"]}) is False


DEEPSEEK_DIGEST = "63fb193b3a9b4322a18e8c6b250ca2e70a5ff531e962dbf95ba089b2566f2fa5"


class TestDropInstalledDigests:
    WORTH = [
        recommend.ModelFit("deepseek-coder-v2:16b", 8.9, 5.2),
        recommend.ModelFit("qwen2.5-coder:14b", 9.0, 5.1),
    ]

    def test_drops_entry_resolving_to_installed_digest(self):
        tiers = recommend.drop_installed_digests(
            recommend.Tiers([], [], list(self.WORTH)),
            {"deepseek-coder-v2:16b": DEEPSEEK_DIGEST},
            {DEEPSEEK_DIGEST})
        assert [m.name for m in tiers.worth_pulling] == ["qwen2.5-coder:14b"]

    def test_unresolved_entries_fail_open(self):
        tiers = recommend.drop_installed_digests(
            recommend.Tiers([], [], list(self.WORTH)),
            {"deepseek-coder-v2:16b": None},
            {DEEPSEEK_DIGEST})
        assert len(tiers.worth_pulling) == 2

    def test_non_matching_digest_kept(self):
        tiers = recommend.drop_installed_digests(
            recommend.Tiers([], [], list(self.WORTH)),
            {"deepseek-coder-v2:16b": "f00d"},
            {DEEPSEEK_DIGEST})
        assert len(tiers.worth_pulling) == 2

    def test_other_tiers_untouched(self):
        selected = [recommend.ModelFit("qwen2.5:14b", 9.0, 5.1)]
        excluded = [recommend.ModelFit("qwen3-coder:30b", 19.0, -4.9)]
        tiers = recommend.drop_installed_digests(
            recommend.Tiers(selected, excluded, list(self.WORTH)),
            {"deepseek-coder-v2:16b": DEEPSEEK_DIGEST},
            {DEEPSEEK_DIGEST})
        assert tiers.selected == selected and tiers.excluded == excluded


class FakeHeadResp:
    def __init__(self, status_code=200, headers=None):
        self.status_code = status_code
        self.ok = status_code < 400
        # Real requests headers are case-insensitive; mirror that.
        self.headers = recommend.requests.structures.CaseInsensitiveDict(headers or {})


class TestRegistryDigest:
    def test_resolves_via_ollama_content_digest_header(self, monkeypatch):
        seen = {}
        def fake_head(url, **kw):
            seen["url"] = url
            return FakeHeadResp(headers={"Ollama-Content-Digest": DEEPSEEK_DIGEST})
        monkeypatch.setattr(recommend.requests, "head", fake_head)
        assert recommend.registry_digest("deepseek-coder-v2:16b") == DEEPSEEK_DIGEST
        assert seen["url"] == ("https://registry.ollama.ai/v2/library/"
                               "deepseek-coder-v2/manifests/16b")

    def test_strips_sha256_prefix_from_docker_header(self, monkeypatch):
        monkeypatch.setattr(
            recommend.requests, "head",
            lambda url, **kw: FakeHeadResp(
                headers={"Docker-Content-Digest": "sha256:abc123"}))
        assert recommend.registry_digest("qwen3:8b") == "abc123"

    def test_unreachable_registry_returns_none(self, monkeypatch):
        def boom(url, **kw):
            raise recommend.requests.ConnectionError("down")
        monkeypatch.setattr(recommend.requests, "head", boom)
        assert recommend.registry_digest("qwen3:8b") is None

    def test_http_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(recommend.requests, "head",
                            lambda url, **kw: FakeHeadResp(status_code=404))
        assert recommend.registry_digest("no-such-model:1b") is None

    def test_missing_digest_header_returns_none(self, monkeypatch):
        monkeypatch.setattr(recommend.requests, "head",
                            lambda url, **kw: FakeHeadResp())
        assert recommend.registry_digest("qwen3:8b") is None


def make_hw(**kw):
    base = dict(backend="cuda",
                gpus=[recommend.Gpu("Tesla T4", 16.1)],
                ram_gb=67.4, budget_gb=16.1, warnings=[])
    base.update(kw)
    return recommend.Hardware(**base)


def make_tiers(selected=(), excluded=(), worth=()):
    return recommend.Tiers(list(selected), list(excluded), list(worth))


class TestRender:
    def test_report_sections_and_numbers(self):
        tiers = make_tiers(
            selected=[recommend.ModelFit("qwen2.5:14b", 9.0, 5.1)],
            excluded=[recommend.ModelFit("qwen3-coder:30b", 19.0, -4.9)],
            worth=[recommend.ModelFit("qwen2.5-coder:14b", 9.0, 5.1)],
        )
        text = recommend.render_report(make_hw(), tiers, headroom_gb=2.0)
        assert "backend: cuda" in text
        assert "Tesla T4 (16.1 GB VRAM)" in text
        assert "budget: 16.1 GB (headroom 2.0 GB)" in text
        assert "margin 5.1 GB" in text
        assert "needs 21.0 GB > 16.1 GB budget" in text
        assert "ollama pull qwen2.5-coder:14b" in text

    def test_warnings_render(self):
        text = recommend.render_report(
            make_hw(warnings=["driver gap"]), make_tiers(), headroom_gb=2.0)
        assert "warning: driver gap" in text

    def test_ollama_down_note_replaces_installed_tiers(self):
        text = recommend.render_report(
            make_hw(), make_tiers(), headroom_gb=2.0,
            ollama_note="Ollama not reachable at http://localhost:11434")
        assert "Ollama not reachable" in text
        assert "# selected" not in text and "# excluded" not in text

    def test_empty_tiers_render_none(self):
        text = recommend.render_report(make_hw(), make_tiers(), headroom_gb=2.0)
        assert text.count("(none)") >= 3

    def test_snippet_smallest_first_and_yaml_parseable(self):
        tiers = make_tiers(
            selected=[recommend.ModelFit("qwen2.5:14b", 9.0, 5.1)],
            worth=[recommend.ModelFit("qwen2.5-coder:1.5b", 1.0, 13.1)],
        )
        snippet = recommend.render_snippet(tiers)
        data = yaml.safe_load(snippet)
        assert data == {"models": [
            {"name": "qwen2.5-coder:1.5b", "size_gb": 1.0},
            {"name": "qwen2.5:14b", "size_gb": 9.0},
        ]}

    def test_empty_snippet_is_omitted_from_report(self):
        text = recommend.render_report(make_hw(), make_tiers(), headroom_gb=2.0)
        assert "models.yaml snippet" not in text


class TestMain:
    def test_unsupported_os_exits_nonzero(self, monkeypatch, capsys):
        monkeypatch.setattr(recommend.platform, "system", lambda: "OpenVMS")
        rc = recommend.main([])
        assert rc != 0
        assert "unsupported OS" in capsys.readouterr().err

    def test_negative_headroom_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc:
            recommend.main(["--headroom-gb", "-1"])
        assert exc.value.code != 0
        assert "must be >= 0" in capsys.readouterr().err

    def test_ollama_down_still_reports(self, monkeypatch, capsys):
        monkeypatch.setattr(
            recommend, "probe_hardware",
            lambda: recommend.Hardware("cuda", [recommend.Gpu("T4", 16.1)],
                                       64.0, 16.1, []))
        def boom(base_url=None):
            raise recommend.requests.ConnectionError("down")
        monkeypatch.setattr(recommend.ollama_client, "list_local_models", boom)
        rc = recommend.main([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Ollama not reachable" in out
        assert "worth pulling" in out

    def test_digest_alias_and_embedding_model_filtered(self, monkeypatch, capsys):
        # The 2026-06-12 T4 run: deepseek-coder-v2:latest is the 16b default
        # tag (same digest), and nomic-embed-text cannot chat. Neither the
        # 16b catalog entry nor the embedding model may reach the output.
        monkeypatch.setattr(
            recommend, "probe_hardware",
            lambda: recommend.Hardware("cuda", [recommend.Gpu("Tesla T4", 16.1)],
                                       64.0, 16.1, []))
        installed = [
            {"name": "deepseek-coder-v2:latest", "size_gb": 8.9,
             "digest": DEEPSEEK_DIGEST, "families": ["deepseek2"]},
            {"name": "nomic-embed-text:latest", "size_gb": 0.27,
             "digest": "aaaa", "families": ["nomic-bert"]},
        ]
        monkeypatch.setattr(recommend.ollama_client, "list_local_models",
                            lambda base_url=None: installed)
        monkeypatch.setattr(
            recommend, "registry_digest",
            lambda name: DEEPSEEK_DIGEST if name == "deepseek-coder-v2:16b" else None)
        rc = recommend.main([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "deepseek-coder-v2:latest" in out
        assert "deepseek-coder-v2:16b" not in out
        assert "nomic-embed-text" not in out

    def test_no_registry_calls_without_installed_digests(self, monkeypatch, capsys):
        # Nothing installed -> nothing to dedupe -> stay off the network.
        monkeypatch.setattr(
            recommend, "probe_hardware",
            lambda: recommend.Hardware("cuda", [recommend.Gpu("Tesla T4", 16.1)],
                                       64.0, 16.1, []))
        monkeypatch.setattr(recommend.ollama_client, "list_local_models",
                            lambda base_url=None: [])
        def fail(name):
            raise AssertionError("registry_digest must not be called")
        monkeypatch.setattr(recommend, "registry_digest", fail)
        rc = recommend.main([])
        assert rc == 0
        assert "worth pulling" in capsys.readouterr().out

    def test_malformed_tags_json_degrades_like_ollama_down(self, monkeypatch, capsys):
        # requests>=2.27 raises requests.exceptions.JSONDecodeError (a
        # RequestException subclass) from Response.json(); pin that a
        # garbage /api/tags body degrades to the note instead of crashing.
        monkeypatch.setattr(
            recommend, "probe_hardware",
            lambda: recommend.Hardware("cuda", [recommend.Gpu("T4", 16.1)],
                                       64.0, 16.1, []))
        def garbage(base_url=None):
            raise recommend.requests.exceptions.JSONDecodeError("bad", "<html>", 0)
        monkeypatch.setattr(recommend.ollama_client, "list_local_models", garbage)
        rc = recommend.main([])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Ollama not reachable" in out
