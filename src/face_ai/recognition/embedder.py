"""Face embedding backends. Inputs are 112x112 BGR crops from ``alignment.align_face``.

* ``sface``  - OpenCV Zoo SFace (Apache-2.0), 128-d. Default (permissive license, 37 MB).
* ``arcface_r50`` - InsightFace ArcFace ResNet-50 / WebFace600K from buffalo_l, 512-d
  (non-commercial research license). Accuracy reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from face_ai.config import resolve_path
from face_ai.utils.device import onnx_providers


class FaceEmbedder(Protocol):
    name: str
    dim: int

    def embed(self, aligned_faces: list[np.ndarray]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


class OnnxEmbedder:
    """Generic 112x112 ONNX embedder: ``(pixel - mean) * scale`` on RGB input, L2-normalized output."""

    def __init__(self, name: str, model_path: str | Path, mean: float, scale: float, providers: str = "cpu",
                 flip_augment: bool = False, batch_size: int = 32):
        import onnxruntime as ort

        p = resolve_path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"model not found: {p}. Run `python scripts/download_models.py`.")
        self.name = name
        self.model_path = p
        self.mean = mean
        self.scale = scale
        self.flip_augment = flip_augment
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(str(p), so, providers=onnx_providers(providers))
        inp = self._sess.get_inputs()[0]
        self._input = inp.name
        # Some exports have a fixed batch of 1.
        self.batch_size = batch_size if not isinstance(inp.shape[0], int) else inp.shape[0]
        self.dim = int(self._sess.get_outputs()[0].shape[-1])

    def _blob(self, faces: list[np.ndarray]) -> np.ndarray:
        return cv2.dnn.blobFromImages(faces, self.scale, (112, 112), (self.mean,) * 3, swapRB=True)

    def _run(self, faces: list[np.ndarray]) -> np.ndarray:
        outs = []
        for i in range(0, len(faces), self.batch_size):
            outs.append(self._sess.run(None, {self._input: self._blob(faces[i:i + self.batch_size])})[0])
        return np.concatenate(outs).astype(np.float32)

    def embed(self, aligned_faces: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Return an (N, dim) array of L2-normalized embeddings."""
        if isinstance(aligned_faces, np.ndarray) and aligned_faces.ndim == 3:
            aligned_faces = [aligned_faces]
        faces = list(aligned_faces)
        if not faces:
            return np.zeros((0, self.dim), np.float32)
        feats = self._run(faces)
        if self.flip_augment:
            feats = feats + self._run([cv2.flip(f, 1) for f in faces])
        return l2_normalize(feats)


def create_embedder(cfg: dict[str, Any]) -> OnnxEmbedder:
    backend = cfg.get("backend", "sface")
    # the embedder section also holds pipeline options (e.g. refine_keypoints) read elsewhere
    params = {k: v for k, v in cfg.get(backend, {}).items()
              if k in ("model_path", "providers", "flip_augment", "batch_size")}
    if backend == "sface":
        # OpenCV FaceRecognizerSF feeds raw RGB pixel values (no mean/scale).
        return OnnxEmbedder("sface", mean=0.0, scale=1.0, **params)
    if backend == "arcface_r50":
        return OnnxEmbedder("arcface_r50", mean=127.5, scale=1.0 / 127.5, **params)
    raise ValueError(f"unknown embedder backend {backend!r}")
