#!/usr/bin/env python
"""Fine-tune a torchvision face detector on WIDER FACE (train_ft split), selecting on val_ft.

    python scripts/train_detector.py                                  # configs/detection.yaml -> training
    python scripts/train_detector.py --resume models/checkpoints/detector_ft/last.pt
    python scripts/train_detector.py --epochs 1 --max-train-images 200 --max-minutes 5   # smoke run

The official WIDER val split (our test set) is never used here; evaluate afterwards with
scripts/evaluate_detector.py --detectors torchvision.
"""

from __future__ import annotations

import argparse
import json
import sys

from face_ai.config import METRICS_DIR, ensure_dirs, load_config, resolve_path
from face_ai.detection.train import train
from face_ai.logging_config import setup_logging
from face_ai.utils.device import hardware_info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train-images", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device")
    ap.add_argument("--max-minutes", type=float, help="wall-clock budget; saves last.pt and stops")
    ap.add_argument("--checkpoint-dir")
    args = ap.parse_args()
    cfg = dict(load_config("detection")["training"])
    for k in ("epochs", "max_train_images", "batch_size", "device", "checkpoint_dir"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    setup_logging(log_file=resolve_path(cfg["checkpoint_dir"]) / "train.log")
    ensure_dirs()
    summary = train(cfg, resume=args.resume, max_minutes=args.max_minutes)
    summary["config"] = cfg
    summary["hardware"] = hardware_info()
    (METRICS_DIR / "detector_training.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
