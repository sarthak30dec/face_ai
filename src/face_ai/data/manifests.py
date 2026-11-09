"""Dataset/model manifests: provenance, licensing, integrity and validation status as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from face_ai.config import MANIFEST_DIR


def _jsonable(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def manifest_path(name: str, directory: Path = MANIFEST_DIR) -> Path:
    return directory / f"{name}.json"


def load_manifest(name: str, directory: Path = MANIFEST_DIR) -> dict[str, Any]:
    p = manifest_path(name, directory)
    return json.loads(p.read_text()) if p.exists() else {}


def update_manifest(name: str, updates: dict[str, Any], directory: Path = MANIFEST_DIR) -> Path:
    """Merge ``updates`` into the manifest (top-level keys replaced) and stamp ``updated_at``."""
    directory.mkdir(parents=True, exist_ok=True)
    data = load_manifest(name, directory)
    data.update(updates)
    data["name"] = name
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    p = manifest_path(name, directory)
    p.write_text(json.dumps(data, indent=2, default=_jsonable))
    return p
