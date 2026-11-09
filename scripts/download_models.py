#!/usr/bin/env python
"""Download pretrained models listed in configs/models.yaml into models/pretrained/<model>/.

    python scripts/download_models.py               # all models
    python scripts/download_models.py --only yunet sface face_landmarker   # permissive-license stack only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from face_ai.config import MANIFEST_DIR, PRETRAINED_DIR, ensure_dirs, load_config
from face_ai.data.download import RemoteFile, fetch_all, human_bytes
from face_ai.data.manifests import update_manifest
from face_ai.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    log = setup_logging(log_file=Path("outputs/reports/download_models.log"))
    ensure_dirs()
    registry = load_config("models")["models"]
    failed = False
    for name in args.only or list(registry):
        spec = registry[name]
        out = PRETRAINED_DIR / name
        results, errors = fetch_all([RemoteFile.from_dict(f) for f in spec["files"]], out, out)
        update_manifest(f"model_{name}", {
            "description": spec["description"], "source": spec["homepage"], "license": spec["license"],
            "download_date": datetime.now(timezone.utc).isoformat(), "location": str(out),
            "files": results, "download_status": "downloaded" if not errors else "incomplete",
            "download_errors": errors,
        }, MANIFEST_DIR)
        log.info("%s: %s (%s)", name, "ok" if not errors else "FAILED", human_bytes(sum(r.size for r in results)))
        failed |= bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
