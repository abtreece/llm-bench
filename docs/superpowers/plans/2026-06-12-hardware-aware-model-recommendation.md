# Hardware-Aware Model Recommendation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An advisory `python -m harness.recommend` command that probes the host's accelerator/RAM budget and reports which Ollama models fit, which installed ones don't, and which catalog models are worth pulling — plus a paste-ready `models.yaml` snippet.

**Architecture:** A new `harness/recommend.py` module with a pure-function core (parsers for `nvidia-smi`/`rocm-smi`/`/proc/meminfo`/`sysctl`/`lspci`, a backend-dispatch function, fit rule, tier selection, renderers) and a thin subprocess/CLI shell. A curated `catalog.yaml` at repo root supplies "worth pulling" candidates. `harness/ollama_client.list_local_models` is repurposed to return names + sizes from `/api/tags`.

**Tech Stack:** Python 3.12, stdlib (`dataclasses`, `subprocess`, `platform`, `argparse`, `json`, `re`), PyYAML, requests, pytest. Tests live in `harness_tests/` (outside the frozen `testpaths`), run explicitly.

**Spec:** `docs/superpowers/specs/2026-06-12-hardware-aware-model-recommendation-design.md` (in git: `git show ce9b7fe:docs/superpowers/specs/2026-06-12-hardware-aware-model-recommendation-design.md`)

**Registry-verified sizes (2026-06-12, ollama.com):** qwen2.5-coder:1.5b = 986MB ≈ 1.0 GB, :7b = 4.7, :14b = 9.0, :32b = 20.0, qwen3:8b = 5.2, qwen3:14b = 9.3, deepseek-coder-v2:16b = 8.9, qwen3-coder:30b = **19.0** (spec's 18.0 was a placeholder; corrected).

---

### Task 0: Workspace setup

**Files:** none (environment only)

- [x] **Step 1: Create venv and install deps** (worktree has no `.venv`)

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

- [x] **Step 2: Verify existing harness tests pass**

Run: `.venv/bin/pytest harness_tests/ -q`
Expected: all pass.

---

### Task 1: `ollama_client.list_local_models` returns names + sizes

**Files:**
- Modify: `harness/ollama_client.py:95-98`
- Test: `harness_tests/test_recommend.py` (new file)

`/api/tags` returns `{"models": [{"name": "qwen2.5:14b", "size": 8988124416, ...}]}` — `size` is bytes. Convert GB = bytes / 1e9 (matches the registry's decimal-GB convention and `models.yaml`). A pure `models_from_tags(data)` carries the logic so tests need no HTTP.

- [x] **Step 1: Write failing tests**

Create `harness_tests/test_recommend.py`:

```python
"""Tests for harness/recommend.py — pure-function core, no subprocess/network."""
from harness import ollama_client


class TestModelsFromTags:
    def test_converts_bytes_to_decimal_gb(self):
        data = {"models": [{"name": "qwen2.5:14b", "size": 9_000_000_000}]}
        assert ollama_client.models_from_tags(data) == [
            {"name": "qwen2.5:14b", "size_gb": 9.0}
        ]

    def test_empty_and_missing_models_key(self):
        assert ollama_client.models_from_tags({}) == []
        assert ollama_client.models_from_tags({"models": []}) == []
```

- [x] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest harness_tests/test_recommend.py -q`
Expected: FAIL — `AttributeError: ... no attribute 'models_from_tags'`

- [x] **Step 3: Implement**

In `harness/ollama_client.py`, replace `list_local_models` (the `Iterable` import becomes unused — remove it from the `typing` import line):

```python
def models_from_tags(data: dict) -> list[dict]:
    """Pure extraction from an /api/tags payload: [{"name", "size_gb"}].

    Ollama reports size in bytes; GB = bytes / 1e9 to match the registry's
    decimal-GB convention used in models.yaml and catalog.yaml.
    """
    return [
        {"name": m["name"], "size_gb": int(m.get("size", 0)) / 1e9}
        for m in data.get("models", [])
    ]


def list_local_models(base_url: str = DEFAULT_BASE_URL) -> list[dict]:
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    r.raise_for_status()
    return models_from_tags(r.json())
```

- [x] **Step 4: Run tests** — `.venv/bin/pytest harness_tests/test_recommend.py -q` → PASS; full `.venv/bin/pytest harness_tests/ -q` still green.

- [x] **Step 5: Commit** — `git add harness/ollama_client.py harness_tests/test_recommend.py && git commit -m "feat: return name and size_gb from list_local_models"`

---

### Task 2: `catalog.yaml` + loader

**Files:**
- Create: `catalog.yaml` (repo root)
- Create: `harness/recommend.py`
- Test: `harness_tests/test_recommend.py`

- [x] **Step 1: Write failing tests** (append to `harness_tests/test_recommend.py`)

```python
from harness import recommend


class TestCatalog:
    def test_repo_catalog_loads_and_is_well_formed(self):
        entries = recommend.load_catalog()
        assert len(entries) >= 8
        for e in entries:
            assert isinstance(e["name"], str) and ":" in e["name"]
            assert e["size_gb"] > 0

    def test_missing_catalog_degrades_to_empty(self, tmp_path):
        assert recommend.load_catalog(tmp_path / "nope.yaml") == []
```

- [x] **Step 2: Run to verify failure** — `ModuleNotFoundError`/`ImportError` on `recommend`.

- [x] **Step 3: Create `catalog.yaml`** (registry-verified sizes, smallest first):

```yaml
# Curated coder-capable models for harness.recommend suggestions.
# size_gb = on-disk size of the default (q4) tag, per the Ollama registry
# (verified 2026-06-12). Spanning 1.5B-32B so every hardware tier gets
# at least one suggestion.
catalog:
  - name: qwen2.5-coder:1.5b
    size_gb: 1.0
    notes: floor sanity check
  - name: qwen2.5-coder:7b
    size_gb: 4.7
  - name: deepseek-coder-v2:16b
    size_gb: 8.9
  - name: qwen2.5-coder:14b
    size_gb: 9.0
  - name: qwen3:8b
    size_gb: 5.2
  - name: qwen3:14b
    size_gb: 9.3
  - name: qwen3-coder:30b
    size_gb: 19.0
  - name: qwen2.5-coder:32b
    size_gb: 20.0
```

(Ordering nicety only; selection sorts explicitly.)

- [x] **Step 4: Create `harness/recommend.py`** with module header + loader:

```python
"""Hardware-aware model recommendation (advisory only).

Probes the host's accelerator/RAM budget, checks installed Ollama models
and the curated catalog.yaml against it, and prints a report plus a
paste-ready models.yaml snippet. Never writes any file; models.yaml
remains the single source of truth for harness.run.

CLI:
    python -m harness.recommend [--headroom-gb 2.0]
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CATALOG_YAML = REPO / "catalog.yaml"


def load_catalog(path: Path = CATALOG_YAML) -> list[dict]:
    # Missing file is fine (no suggestions); a malformed one should surface
    # loudly rather than silently dropping the tier (same philosophy as
    # report.py's models.yaml handling).
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        return []
    return [
        {"name": str(e["name"]), "size_gb": float(e["size_gb"])}
        for e in data["catalog"]
    ]
```

- [x] **Step 5: Run tests** → PASS. **Commit:** `git add catalog.yaml harness/recommend.py harness_tests/test_recommend.py && git commit -m "feat: add curated model catalog and loader"`

---

### Task 3: Probe-output parsers (pure functions)

**Files:**
- Modify: `harness/recommend.py`
- Test: `harness_tests/test_recommend.py`

Unit normalization: nvidia-smi `--nounits` memory.total is **MiB** → GB = MiB·1024²/1e9. `/proc/meminfo` MemTotal is **KiB** → GB = KiB·1024/1e9. `sysctl hw.memsize` is **bytes** → /1e9. rocm-smi JSON `"VRAM Total Memory (B)"` is **bytes** → /1e9.

- [x] **Step 1: Write failing tests**

```python
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
```

- [x] **Step 2: Run to verify failure.**

- [x] **Step 3: Implement** (add to `harness/recommend.py`; `import json`, `import re`, `from dataclasses import dataclass, field`):

```python
@dataclass(frozen=True)
class Gpu:
    name: str
    vram_gb: float


def parse_nvidia_smi(text: str) -> list[Gpu]:
    """Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`.

    memory.total is MiB with --nounits.
    """
    gpus: list[Gpu] = []
    for line in text.splitlines():
        name, sep, mib = line.rpartition(",")
        if not sep:
            continue
        try:
            vram_gb = float(mib.strip()) * 1024**2 / 1e9
        except ValueError:
            continue
        gpus.append(Gpu(name=name.strip(), vram_gb=vram_gb))
    return gpus


def parse_rocm_smi(text: str) -> list[Gpu]:
    """Parse `rocm-smi --showmeminfo vram --json` (best-effort: pinned to the
    documented {"cardN": {"VRAM Total Memory (B)": "<bytes>"}} shape; not
    verified on real ROCm hardware)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    gpus: list[Gpu] = []
    for card, info in sorted(data.items()):
        if not isinstance(info, dict):
            continue
        raw = info.get("VRAM Total Memory (B)")
        if raw is None:
            continue
        try:
            gpus.append(Gpu(name=card, vram_gb=float(raw) / 1e9))
        except ValueError:
            continue
    return gpus


def parse_meminfo(text: str) -> float | None:
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
    if not m:
        return None
    return int(m.group(1)) * 1024 / 1e9  # meminfo kB is KiB


def parse_sysctl_memsize(text: str) -> float | None:
    try:
        return int(text.strip()) / 1e9
    except ValueError:
        return None


def scan_lspci(text: str) -> str | None:
    """Spot a discrete NVIDIA/AMD VGA/3D controller a vendor tool missed."""
    for line in text.splitlines():
        if "VGA compatible controller" not in line and "3D controller" not in line:
            continue
        if "NVIDIA" in line:
            return "nvidia"
        if "AMD" in line or "ATI" in line:
            return "amd"
    return None
```

- [x] **Step 4: Run tests** → PASS. **Commit:** `git commit -am "feat: add hardware probe-output parsers"`

---

### Task 4: Backend dispatch + budget (pure `build_hardware`)

**Files:**
- Modify: `harness/recommend.py`
- Test: `harness_tests/test_recommend.py`

`build_hardware` takes raw probe results (`str | None` per tool; `None` = tool missing/failed) and makes every decision; the subprocess shell comes in Task 7. Budget rule: cuda/rocm = Σ VRAM; metal = 0.67·RAM; cpu = RAM − 4.0.

- [x] **Step 1: Write failing tests**

```python
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
        import pytest
        with pytest.raises(recommend.RecommendError):
            recommend.build_hardware("Windows", "AMD64", nvidia_out=None,
                                     rocm_out=None, lspci_out=None, ram_gb=8.0)

    def test_no_gpu_and_no_ram_raises(self):
        import pytest
        with pytest.raises(recommend.RecommendError):
            recommend.build_hardware("Linux", "x86_64", nvidia_out=None,
                                     rocm_out=None, lspci_out="", ram_gb=None)
```

- [x] **Step 2: Run to verify failure.**

- [x] **Step 3: Implement**

```python
# Metal's default wired-memory limit gives the GPU roughly two-thirds of
# physical RAM on Apple Silicon; --headroom-gb is the escape hatch.
APPLE_UNIFIED_FRACTION = 0.67
# RAM kept back for the OS and Ollama runtime on CPU-only hosts.
CPU_OS_RESERVE_GB = 4.0


class RecommendError(Exception):
    """Nothing to fit against (unsupported OS, no GPU and no readable RAM)."""


@dataclass(frozen=True)
class Hardware:
    backend: str            # "cuda" | "rocm" | "metal" | "cpu"
    gpus: list[Gpu]
    ram_gb: float | None
    budget_gb: float
    warnings: list[str] = field(default_factory=list)


def build_hardware(
    system: str,
    machine: str,
    *,
    nvidia_out: str | None,
    rocm_out: str | None,
    lspci_out: str | None,
    ram_gb: float | None,
) -> Hardware:
    """Decide backend and memory budget from raw probe output.

    Pure: probe_hardware() gathers the raw text, this makes every decision.
    """
    warnings: list[str] = []

    if system == "Darwin":
        if ram_gb is None:
            raise RecommendError("could not read RAM size via sysctl")
        if machine == "arm64":
            return Hardware("metal", [], ram_gb,
                            APPLE_UNIFIED_FRACTION * ram_gb, warnings)
        warnings.append(
            "Intel Mac: Ollama does not accelerate Intel GPUs; using CPU budget.")
        return Hardware("cpu", [], ram_gb, ram_gb - CPU_OS_RESERVE_GB, warnings)

    if system != "Linux":
        raise RecommendError(f"unsupported OS: {system}")

    gpus = parse_nvidia_smi(nvidia_out) if nvidia_out is not None else []
    if gpus:
        return Hardware("cuda", gpus, ram_gb,
                        sum(g.vram_gb for g in gpus), warnings)

    gpus = parse_rocm_smi(rocm_out) if rocm_out is not None else []
    if gpus:
        return Hardware("rocm", gpus, ram_gb,
                        sum(g.vram_gb for g in gpus), warnings)

    if lspci_out is None:
        warnings.append(
            "GPU detection incomplete: lspci not available (pciutils); "
            "assuming CPU-only.")
    else:
        vendor = scan_lspci(lspci_out)
        if vendor == "nvidia":
            warnings.append(
                "lspci shows an NVIDIA GPU but nvidia-smi is missing or "
                "unusable — install the NVIDIA driver to use it. "
                "Falling back to CPU budget.")
        elif vendor == "amd":
            warnings.append(
                "lspci shows an AMD GPU but rocm-smi is missing or "
                "unusable — install ROCm to use it. "
                "Falling back to CPU budget.")

    if ram_gb is None:
        raise RecommendError("no usable GPU and could not read RAM size")
    return Hardware("cpu", [], ram_gb, ram_gb - CPU_OS_RESERVE_GB, warnings)
```

- [x] **Step 4: Run tests** → PASS. **Commit:** `git commit -am "feat: add backend dispatch and memory budget"`

---

### Task 5: Fit rule + tier selection

**Files:**
- Modify: `harness/recommend.py`
- Test: `harness_tests/test_recommend.py`

`fits(model) := size_gb + headroom_gb <= budget_gb`. Tiers: selected (installed ∧ fits), excluded (installed ∧ ¬fits), worth pulling (catalog ∧ fits ∧ ¬installed, exact tag match).

- [x] **Step 1: Write failing tests**

```python
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
```

- [x] **Step 2: Run to verify failure.**

- [x] **Step 3: Implement**

```python
DEFAULT_HEADROOM_GB = 2.0  # KV cache at num_ctx=16384 plus runtime overhead


def fits(size_gb: float, budget_gb: float, headroom_gb: float) -> bool:
    return size_gb + headroom_gb <= budget_gb


@dataclass(frozen=True)
class ModelFit:
    name: str
    size_gb: float
    margin_gb: float  # budget - size - headroom; negative when excluded


@dataclass(frozen=True)
class Tiers:
    selected: list[ModelFit]       # installed, fits
    excluded: list[ModelFit]       # installed, does not fit
    worth_pulling: list[ModelFit]  # catalog, fits, not installed


def select_tiers(
    installed: list[dict],
    catalog: list[dict],
    *,
    budget_gb: float,
    headroom_gb: float,
) -> Tiers:
    def fit(m: dict) -> ModelFit:
        return ModelFit(m["name"], m["size_gb"],
                        budget_gb - m["size_gb"] - headroom_gb)

    by_size = lambda f: f.size_gb
    installed_names = {m["name"] for m in installed}
    selected = sorted(
        (fit(m) for m in installed if fits(m["size_gb"], budget_gb, headroom_gb)),
        key=by_size)
    excluded = sorted(
        (fit(m) for m in installed if not fits(m["size_gb"], budget_gb, headroom_gb)),
        key=by_size)
    worth_pulling = sorted(
        (fit(m) for m in catalog
         if m["name"] not in installed_names
         and fits(m["size_gb"], budget_gb, headroom_gb)),
        key=by_size)
    return Tiers(selected, excluded, worth_pulling)
```

- [x] **Step 4: Run tests** → PASS. **Commit:** `git commit -am "feat: add fit rule and selection tiers"`

---

### Task 6: Report + snippet rendering

**Files:**
- Modify: `harness/recommend.py`
- Test: `harness_tests/test_recommend.py`

- [x] **Step 1: Write failing tests**

```python
import yaml as yaml_mod


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
        data = yaml_mod.safe_load(snippet)
        assert data == {"models": [
            {"name": "qwen2.5-coder:1.5b", "size_gb": 1.0},
            {"name": "qwen2.5:14b", "size_gb": 9.0},
        ]}

    def test_empty_snippet_is_omitted_from_report(self):
        text = recommend.render_report(make_hw(), make_tiers(), headroom_gb=2.0)
        assert "models.yaml snippet" not in text
```

- [x] **Step 2: Run to verify failure.**

- [x] **Step 3: Implement**

```python
def render_snippet(tiers: Tiers) -> str:
    """models.yaml-shaped YAML of selected + worth-pulling, smallest first."""
    models = sorted(tiers.selected + tiers.worth_pulling,
                    key=lambda f: f.size_gb)
    lines = ["models:"]
    for f in models:
        lines.append(f"  - name: {f.name}")
        lines.append(f"    size_gb: {f.size_gb:.1f}")
    return "\n".join(lines) + "\n"


def render_report(
    hw: Hardware,
    tiers: Tiers,
    *,
    headroom_gb: float,
    ollama_note: str | None = None,
) -> str:
    out: list[str] = []
    out.append("# hardware")
    out.append(f"backend: {hw.backend}")
    for g in hw.gpus:
        out.append(f"gpu: {g.name} ({g.vram_gb:.1f} GB VRAM)")
    if hw.ram_gb is not None:
        out.append(f"ram: {hw.ram_gb:.1f} GB")
    out.append(f"budget: {hw.budget_gb:.1f} GB (headroom {headroom_gb:.1f} GB)")
    for w in hw.warnings:
        out.append(f"warning: {w}")
    out.append("")

    if ollama_note:
        out.append(f"note: {ollama_note}")
        out.append("")
    else:
        out.append("# selected (installed, fits)")
        for f in tiers.selected:
            out.append(f"{f.name:<24} {f.size_gb:>5.1f} GB   margin {f.margin_gb:.1f} GB")
        if not tiers.selected:
            out.append("(none)")
        out.append("")

        out.append("# excluded (installed, does not fit)")
        for f in tiers.excluded:
            out.append(
                f"{f.name:<24} {f.size_gb:>5.1f} GB   needs "
                f"{f.size_gb + headroom_gb:.1f} GB > {hw.budget_gb:.1f} GB budget")
        if not tiers.excluded:
            out.append("(none)")
        out.append("")

    out.append("# worth pulling (catalog, fits, not installed)")
    for f in tiers.worth_pulling:
        out.append(f"{f.name:<24} {f.size_gb:>5.1f} GB   ollama pull {f.name}")
    if not tiers.worth_pulling:
        out.append("(none)")
    out.append("")

    if tiers.selected or tiers.worth_pulling:
        out.append("# models.yaml snippet")
        out.append(render_snippet(tiers))
    return "\n".join(out)
```

- [x] **Step 4: Run tests** → PASS. **Commit:** `git commit -am "feat: render recommendation report and models.yaml snippet"`

---

### Task 7: Subprocess shell + CLI `main`

**Files:**
- Modify: `harness/recommend.py`
- Test: `harness_tests/test_recommend.py` (main wiring only; no subprocess in tests)

- [x] **Step 1: Write failing test** (exit codes via monkeypatched probe — still no subprocess):

```python
class TestMain:
    def test_unsupported_os_exits_nonzero(self, monkeypatch, capsys):
        monkeypatch.setattr(recommend.platform, "system", lambda: "OpenVMS")
        rc = recommend.main([])
        assert rc != 0
        assert "unsupported OS" in capsys.readouterr().err

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
```

- [x] **Step 2: Run to verify failure.**

- [x] **Step 3: Implement** (add `import argparse`, `import platform`, `import subprocess`, `import sys`, `import requests`, `from harness import ollama_client`):

```python
def _run(cmd: list[str]) -> str | None:
    """Thin shell: command stdout on success, None if missing/broken."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def probe_hardware() -> Hardware:
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        ram_gb = parse_sysctl_memsize(out) if out is not None else None
        return build_hardware(system, machine, nvidia_out=None,
                              rocm_out=None, lspci_out=None, ram_gb=ram_gb)

    if system == "Linux":
        try:
            ram_gb = parse_meminfo(Path("/proc/meminfo").read_text())
        except OSError:
            ram_gb = None
        nvidia_out = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                           "--format=csv,noheader,nounits"])
        # Only shell out to the next tool when the previous one yielded nothing.
        rocm_out = None
        lspci_out = None
        if not (nvidia_out and parse_nvidia_smi(nvidia_out)):
            rocm_out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
            if not (rocm_out and parse_rocm_smi(rocm_out)):
                lspci_out = _run(["lspci"])
        return build_hardware(system, machine, nvidia_out=nvidia_out,
                              rocm_out=rocm_out, lspci_out=lspci_out,
                              ram_gb=ram_gb)

    return build_hardware(system, machine, nvidia_out=None, rocm_out=None,
                          lspci_out=None, ram_gb=None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Recommend Ollama models for this host (advisory; "
                    "never writes models.yaml).")
    p.add_argument("--headroom-gb", type=float, default=DEFAULT_HEADROOM_GB,
                   help="memory kept free for KV cache and runtime overhead "
                        f"(default {DEFAULT_HEADROOM_GB})")
    args = p.parse_args(argv)

    try:
        hw = probe_hardware()
    except RecommendError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    ollama_note = None
    installed: list[dict] = []
    try:
        installed = ollama_client.list_local_models()
    except requests.RequestException:
        ollama_note = (
            f"Ollama not reachable at {ollama_client.DEFAULT_BASE_URL} — "
            "installed-model tiers skipped.")

    tiers = select_tiers(installed, load_catalog(),
                         budget_gb=hw.budget_gb, headroom_gb=args.headroom_gb)
    sys.stdout.write(render_report(hw, tiers, headroom_gb=args.headroom_gb,
                                   ollama_note=ollama_note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `test_unsupported_os_exits_nonzero` monkeypatches `platform.system`, so `probe_hardware` must reach `build_hardware` for the non-Darwin/non-Linux case (it does — last return) and `main` converts the `RecommendError` to exit 1.

- [x] **Step 4: Run tests** → PASS.

- [x] **Step 5: Smoke-run the CLI on this host (Darwin/arm64 → metal path)**

Run: `.venv/bin/python -m harness.recommend`
Expected: hardware section with `backend: metal`, a budget ≈ 0.67·RAM, an "Ollama not reachable" note (or installed tiers if a local daemon is up), and catalog suggestions. Exit 0.

- [x] **Step 6: Commit** — `git commit -am "feat: add harness.recommend CLI with hardware probe"`

---

### Task 8: Full verification

- [x] **Step 1: Full test suite** — `.venv/bin/pytest harness_tests/ tests/ -q` → all green.
- [x] **Step 2: Re-read spec, confirm every requirement maps to shipped code** (probe table, fit rule + constants, tiers, output sketch, error handling rows, test list).
- [x] **Step 3: Commit plan doc** — `git add docs/superpowers/plans/ && git commit -m "docs: implementation plan for hardware-aware model recommendation"`

---

## Self-review notes

- **Spec coverage:** probe table → Tasks 3/4/7; fit rule + constants → Tasks 4/5; tiers → Task 5; output sketch + snippet → Task 6; catalog → Task 2; ollama_client change → Task 1; all four error-handling rows → Tasks 4 (RecommendError) and 7 (main); every test bullet in the spec has a named test class.
- **Spec deviation (deliberate):** `qwen3-coder:30b` size corrected 18.0 → 19.0 per registry verification; the T4 sanity anchor still holds (19+2=21 > 16.1).
- **Open item resolved:** catalog sizes verified against ollama.com on 2026-06-12. rocm-smi parser marked best-effort in its docstring, pinned to documented output shape, per spec.
