"""Configuration loading. The YAML file in configs/ is the single source of truth
for thresholds, feature settings, and split allocation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(p)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path from cfg['paths'][key] relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p
