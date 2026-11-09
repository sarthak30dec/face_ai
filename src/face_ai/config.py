"""Project paths and YAML configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("FACE_AI_ROOT", Path(__file__).resolve().parents[2]))

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = Path(os.environ.get("FACE_AI_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = Path(os.environ.get("FACE_AI_MODELS_DIR", PROJECT_ROOT / "models"))
PRETRAINED_DIR = MODELS_DIR / "pretrained"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
EXPORT_DIR = MODELS_DIR / "exports"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
VIS_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"
# Private enrollment galleries live outside data/ so they never mix with public datasets.
GALLERY_DIR = Path(os.environ.get("FACE_AI_GALLERY_DIR", PROJECT_ROOT / "gallery"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``."""
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MANIFEST_DIR, SPLITS_DIR, PRETRAINED_DIR,
              CHECKPOINT_DIR, EXPORT_DIR, METRICS_DIR, VIS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
