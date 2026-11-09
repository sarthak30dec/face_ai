#!/usr/bin/env python
"""Compile outputs/metrics/*.json into outputs/reports/summary.md (human-readable experiment report).

    python scripts/make_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from face_ai.config import MANIFEST_DIR, METRICS_DIR, REPORTS_DIR


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def detection_section() -> list[str]:
    rows = ["## Detection - WIDER FACE val (official protocol, IoU 0.5)", "",
            "| detector | AP easy | AP medium | AP hard | thr | P (hard) | R (hard) | false det. | missed | R <16px | R >=64px | p50 ms | p95 ms | img/s | load s | size MB | peak RSS MB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(METRICS_DIR.glob("detection_*.json")):
        r = _load(p)
        op = r["operating_point"]["hard"]
        perf = r["performance"]
        lat = perf["latency"]
        rs = op["recall_by_face_size"]
        rows.append(f"| {p.stem.removeprefix('detection_')} | {r['easy']['ap']:.4f} | {r['medium']['ap']:.4f} | {r['hard']['ap']:.4f} | "
                    f"{op['score_threshold']} | {op['precision']:.3f} | {op['recall']:.3f} | {op['false_detections']} | {op['missed_faces']} | "
                    f"{rs['tiny<16px']:.3f} | {rs['large>=64']:.3f} | {lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} | {lat['throughput_per_s']:.1f} | "
                    f"{perf['model_load_seconds']:.2f} | {perf['model_size_mb']:.2f} | {perf['peak_rss_mb']:.0f} |")
    return rows + [""]


def recognition_section() -> list[str]:
    rows = ["## Recognition - LFW", "",
            "| embedder | 10-fold acc | AUC | EER | TAR@FAR1e-3 | TAR@FAR1e-2 | thr (FPIR 1% calib) | FAR @thr | FRR @thr | rank-1 | DIR | FPIR | unknown reject | mis-ID | ms/face (b1 p50) | p95 | size MB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(METRICS_DIR.glob("recognition_*.json")):
        r = _load(p)
        v = r["verification"]
        thr = r["operating_threshold"]["value"]
        t = r["identification"][r["operating_threshold"]["strategy"]][f"fpir_{r['operating_threshold']['target_fpir']}"]["test"]
        lat = r["performance"]["latency_batch1_per_face"]
        rows.append(f"| {r['embedder']} | {v['ten_fold']['mean_accuracy']:.4f} ± {v['ten_fold']['std_accuracy']:.4f} | {v['auc']:.5f} | "
                    f"{v['eer']['eer']:.4f} | {v['tar_at_far'][0]['tar']:.4f} | {v['tar_at_far'][1]['tar']:.4f} | {thr:.3f} | "
                    f"{v['at_operating_threshold']['far']:.4f} | {v['at_operating_threshold']['frr']:.4f} | {t['rank1_accuracy_closed_set']:.4f} | "
                    f"{t['dir']:.4f} | {t['fpir']:.4f} | {t['unknown_rejection_rate']:.4f} | {t['misidentification_rate']:.4f} | "
                    f"{lat['p50_ms']:.2f} | {lat['p95_ms']:.2f} | {r['performance']['model_size_mb']:.1f} |")
    return rows + [""]


def training_section() -> list[str]:
    r = _load(METRICS_DIR / "detector_training.json")
    if not r:
        return []
    rows = ["## Detector fine-tuning", "", f"device {r['device']}, {r['total_minutes']:.1f} min, train {r['train_stats']}, val {r['val_stats']}", "",
            "| epoch | steps | train loss | val_ft AP | minutes |", "|---|---|---|---|---|"]
    for h in r["history"]:
        rows.append(f"| {h['epoch']} | {h['step']} | {h['train_loss']:.4f} | {h['val_ft_ap']:.4f} | {h['epoch_minutes']:.1f} |")
    return rows + [""]


def data_section() -> list[str]:
    rows = ["## Datasets", "", "| dataset | download | validation | archive bytes | counts |", "|---|---|---|---|---|"]
    for name in ("wider_face", "lfw"):
        m = _load(MANIFEST_DIR / f"{name}.json")
        if m:
            rows.append(f"| {name} | {m.get('download_status')} | {m.get('validation_status')} | {m.get('archive_bytes')} | {m.get('image_counts')} |")
    return rows + [""]


def main() -> int:
    hw = {}
    for p in sorted(METRICS_DIR.glob("*.json")):
        hw = _load(p).get("hardware") or hw
        if hw:
            break
    lines = ["# face_ai experiment summary", "", f"Hardware: `{json.dumps(hw)}`", ""]
    lines += data_section() + detection_section() + training_section() + recognition_section()
    exp = _load(METRICS_DIR / "export_report.json")
    if exp:
        lines += ["## Export", "", "```json", json.dumps(exp, indent=1), "```", ""]
    out = REPORTS_DIR / "summary.md"
    out.write_text("\n".join(lines))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
