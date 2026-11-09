#!/usr/bin/env python
"""Export models for on-device use and check numerical parity.

For each ONNX model: copy to models/exports/onnx/ with a JSON I/O spec, then try ONNX -> PyTorch
(onnx2torch) -> Core ML ML Program (coremltools) and compare Core ML outputs with ONNX Runtime on
real inputs. Results go to outputs/metrics/export_report.json.

    python scripts/export_models.py                      # yunet + sface (permissive licenses)
    python scripts/export_models.py --models yunet sface arcface_r50

Requires the optional export deps: pip install coremltools onnx onnx2torch
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from face_ai.config import EXPORT_DIR, METRICS_DIR, RAW_DIR, load_config, resolve_path
from face_ai.detection.detector import YuNetDetector
from face_ai.logging_config import setup_logging
from face_ai.recognition.alignment import align_face
from face_ai.utils.image_io import load_image

SPECS = {
    "yunet": {
        "source": ("detection", "yunet"),
        "license": "MIT",
        "input": {"name": "input", "shape": [1, 3, 640, 640], "layout": "NCHW", "color": "BGR",
                  "preprocessing": "raw 0-255 float, no normalization; pad/resize to the model input size"},
        "outputs": "cls_{8,16,32}, obj_{8,16,32}, bbox_{8,16,32}, kps_{8,16,32} (anchor-free per-stride heads)",
        "postprocessing": "score = sqrt(cls*obj); decode boxes/keypoints per stride (see OpenCV FaceDetectorYN); NMS 0.3",
    },
    "sface": {
        "source": ("recognition", "sface"),
        "license": "Apache-2.0",
        "input": {"name": "data", "shape": [1, 3, 112, 112], "layout": "NCHW", "color": "RGB",
                  "preprocessing": "112x112 crop aligned to the ArcFace 5-point template; raw 0-255 float"},
        "outputs": "fc1: 128-d embedding; L2-normalize, compare with cosine similarity",
    },
    "arcface_r50": {
        "source": ("recognition", "arcface_r50"),
        "license": "InsightFace non-commercial research only",
        "input": {"name": "input.1", "shape": [1, 3, 112, 112], "layout": "NCHW", "color": "RGB",
                  "preprocessing": "aligned 112x112; (pixel - 127.5) / 127.5"},
        "outputs": "512-d embedding; L2-normalize, compare with cosine similarity",
    },
}


def sample_inputs(name: str, n: int = 8) -> list[np.ndarray]:
    """Real NCHW float32 inputs built the same way the Python pipeline builds them."""
    det = YuNetDetector(load_config("detection")["yunet"]["model_path"])
    imgs = sorted((RAW_DIR / "lfw" / "lfw").glob("A*/*.jpg"))[:n]
    out = []
    for p in imgs:
        img = load_image(p)
        if name == "yunet":
            canvas = cv2.resize(img, (640, 640))
            out.append(canvas.transpose(2, 0, 1)[None].astype(np.float32))
        else:
            d = det.detect(img)[0]
            crop = align_face(img, d.keypoints)
            mean, scale = (127.5, 1 / 127.5) if name == "arcface_r50" else (0.0, 1.0)
            out.append(cv2.dnn.blobFromImage(crop, scale, (112, 112), (mean,) * 3, swapRB=True))
    return out


def export_coreml(name: str, onnx_path: Path, inputs: list[np.ndarray], out_dir: Path) -> dict:
    import coremltools as ct
    import onnxruntime as ort
    import torch
    from onnx2torch import convert

    rep: dict = {}
    t0 = time.time()
    tmodel = convert(str(onnx_path)).eval()
    example = torch.from_numpy(inputs[0])
    traced = torch.jit.trace(tmodel, example)
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(name="input", shape=tuple(example.shape))],
                         convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                         compute_precision=ct.precision.FLOAT16)
    path = out_dir / f"{name}.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))
    rep["mlpackage"] = str(path)
    rep["convert_seconds"] = time.time() - t0
    rep["size_mb"] = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in mlmodel.get_spec().description.output]
    diffs, cosines = [], []
    for x in inputs:
        ref = sess.run(None, {in_name: x})
        got = mlmodel.predict({"input": x})
        got_list = [np.asarray(got[k]) for k in out_names]
        if name == "yunet":
            ref_flat = np.concatenate([r.reshape(-1) for r in sorted(ref, key=lambda a: a.size)])
            got_flat = np.concatenate([g.reshape(-1) for g in sorted(got_list, key=lambda a: a.size)])
            diffs.append(float(np.abs(ref_flat - got_flat).max()))
        else:
            a, b = ref[0].reshape(-1), got_list[0].reshape(-1)
            cosines.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            diffs.append(float(np.abs(a - b).max()))
    rep["parity_vs_onnxruntime"] = {"samples": len(inputs), "max_abs_diff": max(diffs),
                                    **({"min_cosine": min(cosines)} if cosines else {}),
                                    "note": "Core ML run on macOS with FLOAT16 compute precision"}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=["yunet", "sface"])
    ap.add_argument("--skip-coreml", action="store_true")
    args = ap.parse_args()
    log = setup_logging()
    onnx_dir, cm_dir = EXPORT_DIR / "onnx", EXPORT_DIR / "coreml"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.models:
        spec = SPECS[name]
        cfg_name, key = spec["source"]
        src = resolve_path(load_config(cfg_name)[key]["model_path"])
        dst = onnx_dir / src.name
        shutil.copy2(src, dst)
        (onnx_dir / f"{name}.json").write_text(json.dumps({k: v for k, v in spec.items() if k != "source"}, indent=2))
        rep = {"onnx": str(dst), "onnx_size_mb": dst.stat().st_size / 2**20}
        if not args.skip_coreml:
            try:
                rep["coreml"] = export_coreml(name, src, sample_inputs(name), cm_dir)
                log.info("%s: Core ML ok %s", name, rep["coreml"]["parity_vs_onnxruntime"])
            except Exception as e:  # noqa: BLE001 - report and continue with the next model
                rep["coreml"] = {"error": f"{type(e).__name__}: {e}"}
                log.error("%s: Core ML conversion failed: %s", name, e)
        report[name] = rep
    report["mediapipe_face_landmarker"] = {
        "note": "Not converted: use the official MediaPipe Tasks iOS SDK (MediaPipeTasksVision) with the same "
                "face_landmarker.task file; connection topology is identical (FaceLandmarksConnections)."}
    (METRICS_DIR / "export_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
