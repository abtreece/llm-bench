"""Hardware-aware model recommendation (advisory only).

Probes the host's accelerator/RAM budget, checks installed Ollama models
and the curated catalog.yaml against it, and prints a report plus a
paste-ready models.yaml snippet. Never writes any file; models.yaml
remains the single source of truth for harness.run.

CLI:
    python -m harness.recommend [--headroom-gb 2.0]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
    """Parse `rocm-smi --showmeminfo vram --json`.

    Best-effort: pinned to the documented
    {"cardN": {"VRAM Total Memory (B)": "<bytes>"}} shape; not verified on
    real ROCm hardware.
    """
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
