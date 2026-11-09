#!/usr/bin/env python
"""Download, verify and extract the selected datasets into data/raw/<dataset>/.

    python scripts/download_datasets.py                 # all datasets
    python scripts/download_datasets.py --only lfw      # one dataset
    python scripts/download_datasets.py --include-optional   # also WIDER test images (no labels)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from face_ai.config import RAW_DIR, ensure_dirs, load_config
from face_ai.data.download import RemoteFile, fetch_all, human_bytes
from face_ai.data.manifests import update_manifest
from face_ai.data.validate import quick_count
from face_ai.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="dataset keys from configs/datasets.yaml")
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--config", default="datasets")
    args = ap.parse_args()

    log = setup_logging(log_file=Path("outputs/reports/download_datasets.log"))
    ensure_dirs()
    registry = load_config(args.config)["datasets"]
    names = args.only or list(registry)
    failed = False
    for name in names:
        spec = registry[name]
        log.info("=== %s: %s", name, spec["description"])
        files = [RemoteFile.from_dict(f) for f in spec["files"]]
        ds_dir = RAW_DIR / name
        results, errors = fetch_all(files, ds_dir / "archives", ds_dir, include_optional=args.include_optional)
        counts = quick_count(name, ds_dir)
        status = "downloaded" if not errors else "incomplete"
        update_manifest(name, {
            "description": spec["description"],
            "source": spec["homepage"],
            "version": spec["version"],
            "license": spec["license"],
            "citation": spec.get("citation"),
            "download_date": datetime.now(timezone.utc).isoformat(),
            "location": str(ds_dir),
            "files": results,
            "archive_bytes": sum(r.size for r in results),
            "image_counts": counts,
            "expected": spec.get("expected", {}),
            "download_status": status,
            "download_errors": errors,
        })
        total = sum(r.size for r in results)
        log.info("%s: %s, %s across %d files, image counts %s", name, status, human_bytes(total), len(results), counts)
        for e in errors:
            log.error("  %s", e)
        failed |= bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
