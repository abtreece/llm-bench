"""Tests for harness/recommend.py — pure-function core, no subprocess/network."""
import pytest

from harness import ollama_client, recommend


class TestModelsFromTags:
    def test_converts_bytes_to_decimal_gb(self):
        data = {"models": [{"name": "qwen2.5:14b", "size": 9_000_000_000}]}
        assert ollama_client.models_from_tags(data) == [
            {"name": "qwen2.5:14b", "size_gb": 9.0}
        ]

    def test_empty_and_missing_models_key(self):
        assert ollama_client.models_from_tags({}) == []
        assert ollama_client.models_from_tags({"models": []}) == []


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
