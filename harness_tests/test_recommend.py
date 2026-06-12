"""Tests for harness/recommend.py — pure-function core, no subprocess/network."""
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
