#!/usr/bin/env python
"""Evaluate face detectors on WIDER FACE val (official protocol: AP easy/medium/hard).

    python scripts/evaluate_detector.py                         # detectors from configs/evaluation.yaml
    python scripts/evaluate_detector.py --detectors yunet --limit 300   # quick check on a subset
    python scripts/evaluate_detector.py --detectors torchvision --checkpoint models/checkpoints/detector_ft/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from face_ai.config import METRICS_DIR, OUTPUT_DIR, VIS_DIR, ensure_dirs, load_config, resolve_path
from face_ai.detection.detector import create_detector
from face_ai.detection.inference import predict_images
from face_ai.evaluation.benchmarks import PeakRSS, latency_stats, model_size_mb
from face_ai.evaluation.detection_metrics import (SETTINGS, evaluate_wider, load_wider_gt, operating_point,
                                                  write_wider_predictions)
from face_ai.logging_config import setup_logging
from face_ai.utils.device import hardware_info


def plot_pr(results: dict[str, dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, s in zip(axes, SETTINGS):
        for name, r in results.items():
            ax.plot(r[s]["recall_curve"], r[s]["precision_curve"], label=f"{name} AP={r[s]['ap']:.3f}")
        ax.set(title=f"WIDER FACE val - {s}", xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.01))
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detectors", nargs="*")
    ap.add_argument("--limit", type=int, help="evaluate only the first N images (AP then covers only those)")
    ap.add_argument("--checkpoint", help="torchvision checkpoint (for --detectors torchvision)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    log = setup_logging(log_file=OUTPUT_DIR / "reports" / "evaluate_detector.log")
    ensure_dirs()
    ecfg = load_config("evaluation")["detection"]
    dcfg = load_config("detection")
    split = json.loads(resolve_path(ecfg["split"]).read_text())
    image_root = Path(split["image_root"])
    images = split["images"][: args.limit] if args.limit else split["images"]
    gt = load_wider_gt(resolve_path(ecfg["gt_dir"]))
    if args.limit:
        keys = {str(Path(r).with_suffix("")) for r in images}
        gt.images = {k: v for k, v in gt.images.items() if k in keys}
    hw = hardware_info()
    all_results = {}
    for name in args.detectors or ecfg["detectors"]:
        cfg = {**dcfg, "backend": name}
        cfg[name] = {**dcfg.get(name, {}), "score_threshold": ecfg["eval_score_threshold"]}
        if args.checkpoint and name == "torchvision":
            cfg[name]["checkpoint"] = args.checkpoint
        t0 = time.perf_counter()
        det = create_detector(cfg)
        load_s = time.perf_counter() - t0
        with PeakRSS() as mem:
            preds, times, sizes, errors = predict_images(det, image_root, images, ecfg["warmup_images"], f"{name} WIDER val")
        res = evaluate_wider(preds, gt, ecfg["iou_threshold"])
        op_thr = ecfg["operating_thresholds"].get(name, 0.5)
        res["operating_point"] = {s: operating_point(preds, gt, op_thr, s, ecfg["iou_threshold"]) for s in ("easy", "hard")}
        model_files = [p for p in (cfg[name].get("model_path"), cfg[name].get("checkpoint")) if p]
        sz = np.array(sizes)
        res["performance"] = {
            "model_load_seconds": load_s,
            "model_size_mb": model_size_mb([resolve_path(p) for p in model_files]),
            "latency": latency_stats(times),
            "peak_rss_mb": mem.peak_mb,
            "image_resolution": {"median_wh": np.median(sz, 0).tolist(), "max_wh": sz.max(0).tolist()},
            "batch_size": 1,
            "warmup": f"{ecfg['warmup_images']} untimed images before the timed pass; timing covers detector.detect() only (no disk I/O)",
            "threads": "default (ONNX Runtime / OpenCV use all cores)",
        }
        res.update({"detector": name, "config": cfg[name], "images_evaluated": len(preds), "unreadable": errors,
                    "hardware": hw, "date": datetime.now(timezone.utc).isoformat(), "subset_limit": args.limit})
        tag = f"{name}{args.tag}" + (f"_first{args.limit}" if args.limit else "")
        write_wider_predictions(preds, OUTPUT_DIR / "predictions" / "wider_val" / tag)
        out = METRICS_DIR / f"detection_{tag}.json"
        out.write_text(json.dumps(res, indent=1))
        op = res["operating_point"]["hard"]
        log.info("%s: AP easy=%.4f medium=%.4f hard=%.4f | @%.2f hard: P=%.3f R=%.3f FD=%d missed=%d | p50=%.1fms p95=%.1fms -> %s",
                 name, res["easy"]["ap"], res["medium"]["ap"], res["hard"]["ap"], op_thr, op["precision"], op["recall"],
                 op["false_detections"], op["missed_faces"], res["performance"]["latency"]["p50_ms"],
                 res["performance"]["latency"]["p95_ms"], out)
        all_results[tag] = res
    if all_results:
        plot_pr(all_results, VIS_DIR / f"wider_pr{args.tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
