"""MediaPipe Face Landmarker wrapper (478-point mesh).

Two modes:
* ``extract(image, detection)`` - crop around a box from *our* detector and run the landmarker on
  the crop, so mesh and identity refer to the same face.
* ``extract_all(image)`` - let MediaPipe's own BlazeFace find faces (useful for close-range video).

The mesh is a geometric visualization; it is not used for identity.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from face_ai.api.schemas import FaceDetection, FaceLandmarks
from face_ai.config import resolve_path


class FaceMeshLandmarker:
    def __init__(self, model_path: str | Path, num_faces: int = 4, min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5, crop_scale: float = 1.8, crop_size: int = 256):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        p = resolve_path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f"model not found: {p}. Run `python scripts/download_models.py`.")
        self._mp = mp
        self.crop_scale = crop_scale
        self.crop_size = crop_size

        def make(n: int):
            opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(p)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=n,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_presence_confidence,
            )
            return vision.FaceLandmarker.create_from_options(opts)

        self._single = make(1)
        self._multi = make(num_faces)

    def close(self) -> None:
        self._single.close()
        self._multi.close()

    def __enter__(self) -> "FaceMeshLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self, landmarker, rgb: np.ndarray):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return landmarker.detect(img).face_landmarks

    def extract(self, image: np.ndarray, detection: FaceDetection) -> FaceLandmarks | None:
        """Landmarks for the face in ``detection.bbox`` (pixel coords in ``image``), or None if not found."""
        h, w = image.shape[:2]
        box = detection.bbox.expand(self.crop_scale, square=True)
        x1, y1 = int(np.floor(box.x1)), int(np.floor(box.y1))
        side = int(np.ceil(box.width))
        if side < 8:
            return None
        # Pad so the square crop is always fully inside the canvas.
        pad = max(0, -x1, -y1, x1 + side - w, y1 + side - h)
        src = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else image
        crop = src[y1 + pad:y1 + pad + side, x1 + pad:x1 + pad + side]
        scale = self.crop_size / side
        crop = cv2.resize(crop, (self.crop_size, self.crop_size),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        faces = self._run(self._single, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not faces:
            return None
        pts = np.array([(lm.x, lm.y, lm.z) for lm in faces[0]], dtype=np.float32)
        pts[:, 0] = pts[:, 0] * self.crop_size / scale + x1
        pts[:, 1] = pts[:, 1] * self.crop_size / scale + y1
        pts[:, 2] = pts[:, 2] * self.crop_size / scale
        return FaceLandmarks(pts)

    def extract_all(self, image: np.ndarray) -> list[FaceLandmarks]:
        h, w = image.shape[:2]
        faces = self._run(self._multi, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        out = []
        for i, f in enumerate(faces):
            pts = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in f], dtype=np.float32)
            out.append(FaceLandmarks(pts, face_index=i))
        return out
