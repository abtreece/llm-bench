"""Hardware-aware model recommendation (advisory only).

Probes the host's accelerator/RAM budget, checks installed Ollama models
and the curated catalog.yaml against it, and prints a report plus a
paste-ready models.yaml snippet. Non-chat models (embeddings) are skipped,
and worth-pulling suggestions already installed under an aliased tag are
deduped via registry manifest digests. Never writes any file; models.yaml
remains the single source of truth for harness.run.

CLI:
    python -m harness.recommend [--headroom-gb 2.0]
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

from harness import ollama_client

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
        return Hardware("cpu", [], ram_gb,
                        max(0.0, ram_gb - CPU_OS_RESERVE_GB), warnings)

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
    return Hardware("cpu", [], ram_gb,
                    max(0.0, ram_gb - CPU_OS_RESERVE_GB), warnings)


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


# Embedding models ship bert-family encoders per /api/tags details.families.
EMBEDDING_FAMILIES = {"bert", "nomic-bert"}


def is_chat_model(model: dict) -> bool:
    """Whether an installed model can serve /api/chat (benchmark warm-up).

    families is the /api/tags signal; the name check also catches embedding
    models that report their base chat family (e.g. qwen3-embedding -> qwen3)
    and is the only signal when families is absent.
    """
    families = model.get("families") or []
    if families and all(f in EMBEDDING_FAMILIES for f in families):
        return False
    return "embed" not in model["name"].lower()


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

    def by_size(f: ModelFit) -> float:
        return f.size_gb

    # Pasting a non-chat model (embeddings) into models.yaml would crash
    # harness.run at warm-up, so they never enter any tier.
    installed = [m for m in installed if is_chat_model(m)]
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


def drop_installed_digests(
    tiers: Tiers,
    catalog_digests: dict[str, str | None],
    installed_digests: set[str],
) -> Tiers:
    """Drop worth-pulling entries whose registry digest is already installed.

    Catches tag aliases (deepseek-coder-v2:latest IS the 16b default tag)
    that name comparison misses. Unresolved entries (None digest) fail open:
    the worst case is suggesting a redundant pull, while failing closed
    would hide a real suggestion.
    """
    worth = [f for f in tiers.worth_pulling
             if catalog_digests.get(f.name) not in installed_digests]
    return Tiers(tiers.selected, tiers.excluded, worth)


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


REGISTRY_BASE_URL = "https://registry.ollama.ai"


def registry_digest(name: str, *, base_url: str = REGISTRY_BASE_URL) -> str | None:
    """Manifest digest for a registry tag; None when offline or unknown.

    The registry answers manifest HEADs with an Ollama-Content-Digest header
    (bare-hex sha256 of the manifest, the same value /api/tags reports for
    installed models); Docker-Content-Digest is the spec-standard fallback.
    """
    model, _, tag = name.partition(":")
    repo = model if "/" in model else f"library/{model}"
    try:
        r = requests.head(
            f"{base_url}/v2/{repo}/manifests/{tag or 'latest'}",
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
            timeout=5,
        )
    except requests.RequestException:
        return None
    if not r.ok:
        return None
    digest = (r.headers.get("ollama-content-digest")
              or r.headers.get("docker-content-digest", ""))
    return digest.removeprefix("sha256:") or None


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


def _nonnegative_float(s: str) -> float:
    v = float(s)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {s}")
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Recommend Ollama models for this host (advisory; "
                    "never writes models.yaml).")
    p.add_argument("--headroom-gb", type=_nonnegative_float, default=DEFAULT_HEADROOM_GB,
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
    installed_digests = {d for m in installed if (d := m.get("digest"))}
    if installed_digests and tiers.worth_pulling:
        catalog_digests = {f.name: registry_digest(f.name)
                           for f in tiers.worth_pulling}
        tiers = drop_installed_digests(tiers, catalog_digests, installed_digests)
    sys.stdout.write(render_report(hw, tiers, headroom_gb=args.headroom_gb,
                                   ollama_note=ollama_note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
