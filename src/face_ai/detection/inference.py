"""Batch inference helpers for datasets and face cropping."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from face_ai.api.schemas import FaceDetection
from face_ai.detection.detector import FaceDetector
from face_ai.utils.image_io import load_image


def detections_to_array(dets: list[FaceDetection]) -> np.ndarray:
    """(N, 5) [x, y, w, h, score] - the WIDER prediction format."""
    if not dets:
        return np.zeros((0, 5), np.float32)
    return np.array([[*d.bbox.to_xywh(), d.score] for d in dets], dtype=np.float32)


def predict_images(detector: FaceDetector, image_root: Path, rel_paths: Iterable[str], warmup: int = 10,
                   desc: str = "detect") -> tuple[dict[str, np.ndarray], list[float], list[tuple[int, int]], list[str]]:
    """Run ``detector`` over images. Returns (preds keyed by path without extension, per-image
    seconds, image (w, h) sizes, unreadable files). Warm-up calls are excluded from timings."""
    rel_paths = list(rel_paths)
    for rel in rel_paths[:warmup]:
        detector.detect(load_image(image_root / rel))
    preds: dict[str, np.ndarray] = {}
    times: list[float] = []
    sizes: list[tuple[int, int]] = []
    errors: list[str] = []
    for rel in tqdm(rel_paths, desc=desc, mininterval=2.0):
        try:
            img = load_image(image_root / rel)
        except Exception:  # noqa: BLE001 - recorded
            errors.append(rel)
            continue
        t0 = time.perf_counter()
        dets = detector.detect(img)
        times.append(time.perf_counter() - t0)
        sizes.append((img.shape[1], img.shape[0]))
        preds[str(Path(rel).with_suffix(""))] = detections_to_array(dets)
    return preds, times, sizes, errors


def crop_faces(image: np.ndarray, detections: list[FaceDetection], margin: float = 0.2) -> list[np.ndarray]:
    """Rectangular crops (with margin) for each detection, clipped to the image."""
    h, w = image.shape[:2]
    out = []
    for d in detections:
        b = d.bbox.expand(1 + 2 * margin).clip(w, h)
        out.append(image[int(b.y1):int(b.y2), int(b.x1):int(b.x2)].copy())
    return out
