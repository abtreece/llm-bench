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
