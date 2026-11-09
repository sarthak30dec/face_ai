#!/usr/bin/env python
"""Convert annotations, build reproducible splits, and pre-detect LFW faces.

    python scripts/preprocess_datasets.py                       # everything, seed 42
    python scripts/preprocess_datasets.py --lfw-detectors yunet scrfd
"""

from __future__ import annotations

import argparse
import json
import sys

from face_ai.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, SPLITS_DIR, ensure_dirs, load_config
from face_ai.data.manifests import update_manifest
from face_ai.data.preprocess import detect_lfw_faces, lfw_openset_split, lfw_pairs_split, preprocess_wider
from face_ai.detection.detector import create_detector
from face_ai.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-wider", action="store_true")
    ap.add_argument("--skip-lfw", action="store_true")
    ap.add_argument("--lfw-detectors", nargs="*", default=["yunet"])
    args = ap.parse_args()
    log = setup_logging(log_file=REPORTS_DIR / "preprocess.log")
    ensure_dirs()
    summary = {}
    if not args.skip_wider:
        s = preprocess_wider(RAW_DIR / "wider_face", PROCESSED_DIR, SPLITS_DIR, seed=args.seed)
        log.info("WIDER: train_ft=%d val_ft=%d test=%d; excluded for leakage: %s", s["train_ft"], s["val_ft"],
                 s["test"], s["excluded_train_leakage"])
        update_manifest("wider_face", {"preprocessing": {k: v for k, v in s.items()}})
        summary["wider_face"] = s
    if not args.skip_lfw:
        s = {"pairs": lfw_pairs_split(RAW_DIR / "lfw", SPLITS_DIR),
             "openset": lfw_openset_split(RAW_DIR / "lfw", SPLITS_DIR, seed=args.seed)}
        det_cfg = load_config("detection")
        for backend in args.lfw_detectors:
            det = create_detector({**det_cfg, "backend": backend})
            r = detect_lfw_faces(RAW_DIR / "lfw", PROCESSED_DIR, det)
            log.info("LFW detection (%s): %s; fallback images: %d", backend, r["status_counts"], len(r["fallback_images"]))
            s[f"faces_{backend}"] = r
        update_manifest("lfw", {"preprocessing": s})
        summary["lfw"] = s
    (REPORTS_DIR / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
