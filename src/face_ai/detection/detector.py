"""Face detector backends behind one interface.

* ``yunet``   - OpenCV YuNet (MIT). Default: 0.23 MB, fast on CPU, 5 keypoints, Core ML friendly.
* ``scrfd``   - InsightFace SCRFD-10GF from buffalo_l (non-commercial research license). Higher recall
                on small faces; used as an accuracy reference.
* ``torchvision`` - a torchvision detector fine-tuned by ``face_ai.detection.train`` (no keypoints).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from face_ai.api.schemas import BoundingBox, FaceDetection
from face_ai.config import resolve_path
from face_ai.utils.device import onnx_providers
from face_ai.utils.image_io import ensure_bgr

log = logging.getLogger(__name__)


class FaceDetector(Protocol):
    name: str
    score_threshold: float

    def detect(self, image: np.ndarray) -> list[FaceDetection]: ...


def _check_model(path: str | Path) -> Path:
    p = resolve_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"model not found: {p}. Run `python scripts/download_models.py`.")
    return p


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Plain NMS on (N, 4) x1y1x2y2 boxes; returns kept indices sorted by score."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


class YuNetDetector:
    """OpenCV ``FaceDetectorYN``. ``max_side`` downsizes large inputs for speed (boxes are mapped back)."""

    name = "yunet"

    def __init__(self, model_path: str | Path, score_threshold: float = 0.6, nms_threshold: float = 0.3,
                 top_k: int = 5000, max_side: int | None = None, min_face_size: float = 0.0):
        self.model_path = _check_model(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.max_side = max_side
        self.min_face_size = min_face_size
        self._det = cv2.FaceDetectorYN.create(str(self.model_path), "", (320, 320), score_threshold,
                                              nms_threshold, top_k)

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        image = ensure_bgr(image)
        h, w = image.shape[:2]
        scale = 1.0
        if self.max_side and max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
            image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        self._det.setScoreThreshold(self.score_threshold)
        self._det.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self._det.detect(image)
        out: list[FaceDetection] = []
        if faces is None:
            return out
        for f in faces:
            x, y, bw, bh = (f[:4] / scale).tolist()
            if min(bw, bh) < self.min_face_size:
                continue
            kps = (f[4:14].reshape(5, 2) / scale).astype(np.float32)
            out.append(FaceDetection(BoundingBox.from_xywh(x, y, bw, bh), float(f[14]), kps, detector=self.name))
        out.sort(key=lambda d: d.score, reverse=True)
        return out


class SCRFDDetector:
    """SCRFD (InsightFace) via ONNX Runtime. Letterboxes the image into ``input_size``."""

    name = "scrfd"
    _strides = (8, 16, 32)
    _num_anchors = 2

    def __init__(self, model_path: str | Path, score_threshold: float = 0.5, nms_threshold: float = 0.4,
                 input_size: int = 640, providers: str = "cpu", min_face_size: float = 0.0):
        import onnxruntime as ort

        self.model_path = _check_model(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.min_face_size = min_face_size
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(str(self.model_path), so, providers=onnx_providers(providers))
        self._input = self._sess.get_inputs()[0].name
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchors(self, fh: int, fw: int, stride: int) -> np.ndarray:
        key = (fh, fw, stride)
        if key not in self._anchor_cache:
            ys, xs = np.mgrid[:fh, :fw]
            centers = (np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride).astype(np.float32)
            self._anchor_cache[key] = np.repeat(centers, self._num_anchors, axis=0)
        return self._anchor_cache[key]

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        image = ensure_bgr(image)
        h, w = image.shape[:2]
        s = self.input_size
        scale = min(s / h, s / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(image, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (s, s), (127.5, 127.5, 127.5), swapRB=True)
        outs = self._sess.run(None, {self._input: blob})
        n = len(self._strides)
        all_boxes, all_scores, all_kps = [], [], []
        for i, stride in enumerate(self._strides):
            scores = outs[i].reshape(-1)
            bbox = outs[i + n].reshape(-1, 4) * stride
            kps = outs[i + 2 * n].reshape(-1, 10) * stride
            anchors = self._anchors(s // stride, s // stride, stride)
            keep = np.where(scores >= self.score_threshold)[0]
            if keep.size == 0:
                continue
            a = anchors[keep]
            b = bbox[keep]
            boxes = np.stack([a[:, 0] - b[:, 0], a[:, 1] - b[:, 1], a[:, 0] + b[:, 2], a[:, 1] + b[:, 3]], axis=1)
            k = kps[keep].reshape(-1, 5, 2) + a[:, None, :]
            all_boxes.append(boxes)
            all_scores.append(scores[keep])
            all_kps.append(k)
        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes) / scale
        scores = np.concatenate(all_scores)
        kps = np.concatenate(all_kps) / scale
        keep = nms(boxes, scores, self.nms_threshold)
        out = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            if min(x2 - x1, y2 - y1) < self.min_face_size:
                continue
            box = BoundingBox(float(x1), float(y1), float(x2), float(y2)).clip(w, h)
            out.append(FaceDetection(box, float(scores[i]), kps[i].astype(np.float32), detector=self.name))
        return out


def create_detector(cfg: dict[str, Any]) -> FaceDetector:
    """Build a detector from a ``configs/detection.yaml``-style dict (keys: backend, <backend>: {...})."""
    backend = cfg.get("backend", "yunet")
    params = dict(cfg.get(backend, {}))
    if "score_threshold" in cfg:
        params["score_threshold"] = cfg["score_threshold"]
    if backend == "yunet":
        return YuNetDetector(**params)
    if backend == "scrfd":
        return SCRFDDetector(**params)
    if backend == "torchvision":
        from face_ai.detection.train import TorchvisionFaceDetector

        return TorchvisionFaceDetector(**params)
    raise ValueError(f"unknown detector backend {backend!r}")
