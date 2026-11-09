#!/usr/bin/env python
"""Benchmark model loading and per-stage / end-to-end latency of the service on one image.

    python scripts/benchmark_pipeline.py                        # configs/evaluation.yaml -> benchmark
    python scripts/benchmark_pipeline.py --image photo.jpg --runs 300

Run it with nothing else using the CPU/GPU; numbers are only comparable on the same hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from face_ai.api.service import FaceAIService
from face_ai.config import METRICS_DIR, load_config, resolve_path
from face_ai.evaluation.benchmarks import PeakRSS, benchmark, model_size_mb
from face_ai.logging_config import setup_logging
from face_ai.utils.device import hardware_info
from face_ai.utils.image_io import load_image


def main() -> int:
    bcfg = load_config("evaluation")["benchmark"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=bcfg["image"])
    ap.add_argument("--runs", type=int, default=bcfg["runs"])
    ap.add_argument("--warmup", type=int, default=bcfg["warmup"])
    ap.add_argument("--detectors", nargs="*", default=["yunet", "scrfd"])
    ap.add_argument("--embedders", nargs="*", default=["sface", "arcface_r50"])
    args = ap.parse_args()
    log = setup_logging()
    img = load_image(resolve_path(args.image))
    dcfg, rcfg = load_config("detection"), load_config("recognition")
    out: dict = {"image": args.image, "resolution_wh": [img.shape[1], img.shape[0]], "runs": args.runs,
                 "warmup": args.warmup, "batch_size": 1, "hardware": hardware_info(),
                 "date": datetime.now(timezone.utc).isoformat(), "detectors": {}, "embedders": {}}

    for d in args.detectors:
        svc = FaceAIService.from_configs(detector_backend=d)
        t0 = time.perf_counter()
        _ = svc.detector
        load = time.perf_counter() - t0
        r = benchmark(svc.detect_faces, [img] * args.runs, warmup=args.warmup)
        out["detectors"][d] = {"model_load_seconds": load, "faces": len(svc.detect_faces(img)),
                               "model_size_mb": model_size_mb([resolve_path(dcfg[d]["model_path"])]), **r}
        log.info("detector %s: load %.3fs, p50 %.2f ms, p95 %.2f ms", d, load, r["p50_ms"], r["p95_ms"])

    svc = FaceAIService.from_configs()
    faces = svc.detect_faces(img)
    crops = [svc.align(img, f) for f in faces]
    for e in args.embedders:
        s = FaceAIService.from_configs(embedder_backend=e)
        t0 = time.perf_counter()
        _ = s.embedder
        load = time.perf_counter() - t0
        r1 = benchmark(lambda c: s.embedder.embed([c]), (crops * (args.runs // max(1, len(crops)) + 1))[: args.runs], warmup=args.warmup)
        rb = benchmark(lambda cs: s.embedder.embed(cs), [crops] * 50, warmup=3)
        out["embedders"][e] = {"model_load_seconds": load, "model_size_mb": model_size_mb([resolve_path(rcfg[e]["model_path"])]),
                               "per_face_batch1": r1, f"per_image_batch{len(crops)}": rb}
        log.info("embedder %s: load %.3fs, per face p50 %.2f ms; %d faces batched p50 %.2f ms", e, load, r1["p50_ms"], len(crops), rb["p50_ms"])

    t0 = time.perf_counter()
    _ = svc.landmarker
    lm_load = time.perf_counter() - t0
    lm = benchmark(lambda f: svc.extract_landmarks(img, f), (faces * (args.runs // max(1, len(faces)) + 1))[: args.runs], warmup=args.warmup)
    out["landmarks"] = {"model_load_seconds": lm_load, "per_face": lm,
                        "model_size_mb": model_size_mb([resolve_path(rcfg["landmarks"]["model_path"])])}
    log.info("landmarks: load %.3fs, per face p50 %.2f ms", lm_load, lm["p50_ms"])

    with PeakRSS() as mem:
        e2e = benchmark(lambda x: svc.analyze(x), [img] * max(20, args.runs // 5), warmup=3)
    e2e_vis = benchmark(lambda x: svc.render_overlay(x, faces), [img] * 50, warmup=3)
    out["end_to_end_default_stack"] = {"faces_in_image": len(faces), "analyze": e2e, "render_overlay": e2e_vis,
                                       "peak_rss_mb": mem.peak_mb,
                                       "stack": "yunet + sface + mediapipe landmarks (per face), CPU"}
    log.info("end-to-end analyze (%d faces): p50 %.1f ms, p95 %.1f ms, peak RSS %.0f MB", len(faces), e2e["p50_ms"], e2e["p95_ms"], mem.peak_mb)
    (METRICS_DIR / "benchmark_pipeline.json").write_text(json.dumps(out, indent=1))
    svc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
