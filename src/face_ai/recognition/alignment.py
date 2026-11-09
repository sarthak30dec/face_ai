"""5-point similarity alignment to the canonical 112x112 ArcFace template."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from face_ai.api.schemas import FaceDetection

# Standard ArcFace/InsightFace destination points for a 112x112 crop
# (image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner).
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation) mapping src -> dst.

    Returns a 2x3 matrix. Umeyama (1991), without reflection.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[1, 1] = -1
    r = u @ s @ vt
    var_s = (sc ** 2).sum() / n
    scale = np.trace(np.diag(d) @ s) / var_s if var_s > 0 else 1.0
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def align_face(image: np.ndarray, keypoints: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp the face so its 5 keypoints land on the ArcFace template; returns a size x size BGR crop."""
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    m = umeyama_similarity(np.asarray(keypoints, dtype=np.float32).reshape(5, 2), dst)
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def crop_without_keypoints(image: np.ndarray, detection: FaceDetection, size: int = 112) -> np.ndarray:
    """Fallback when a detector gives no keypoints: square crop around the box (lower accuracy)."""
    h, w = image.shape[:2]
    b = detection.bbox.expand(1.0, square=True)
    x1, y1, x2, y2 = (int(round(v)) for v in (b.x1, b.y1, b.x2, b.y2))
    pad = max(0, -x1, -y1, x2 - w, y2 - h)
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else image
    crop = padded[y1 + pad:y2 + pad, x1 + pad:x2 + pad]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def needs_roll_fix(detection: FaceDetection, threshold_deg: float = 20.0) -> bool:
    """True if the keypoints show a strong in-plane tilt or an implausible layout (mouth not below the eyes)."""
    if detection.keypoints is None:
        return False
    k = np.asarray(detection.keypoints, np.float32)
    eye_vec = k[1] - k[0]
    iod = float(np.linalg.norm(eye_vec))
    if iod < 1e-6:
        return True
    roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
    down = np.array([-eye_vec[1], eye_vec[0]]) / iod  # perpendicular to the eye line, pointing chin-wards
    mouth_drop = float(np.dot((k[3] + k[4]) / 2 - (k[0] + k[1]) / 2, down)) / iod
    return abs(roll) > threshold_deg or mouth_drop < 0.4


@lru_cache(maxsize=1)
def _mesh_indices() -> tuple[list[int], list[int]]:
    from face_ai.landmarks import topology

    def verts(group: str) -> list[int]:
        return sorted({i for edge in topology.connections(group) for i in edge})

    return verts("right_eye"), verts("left_eye")


# MediaPipe face-mesh indices: nose tip, mouth corner on the subject's right, on the subject's left.
_MP_NOSE, _MP_MOUTH_R, _MP_MOUTH_L = 1, 61, 291


def mesh_five_points(points: np.ndarray) -> np.ndarray:
    """ArcFace-order 5 points (subject's right eye first, as on an upright face) from a 478-point mesh."""
    right_eye, left_eye = _mesh_indices()
    p = np.asarray(points)[:, :2]
    return np.stack([p[right_eye].mean(0), p[left_eye].mean(0), p[_MP_NOSE], p[_MP_MOUTH_R], p[_MP_MOUTH_L]]).astype(np.float32)


def align_detection(image: np.ndarray, detection: FaceDetection, size: int = 112) -> np.ndarray:
    if detection.keypoints is not None:
        return align_face(image, detection.keypoints, size)
    return crop_without_keypoints(image, detection, size)


class KeypointRefiner:
    """Re-estimate the 5 keypoints with SCRFD on a square crop around an existing detection.

    YuNet keypoints are slightly less precise than SCRFD's; alignment error directly costs
    recognition accuracy. Running SCRFD on a small crop (``input_size`` px) costs a few ms per face,
    far less than running it on the whole image. SCRFD weights are non-commercial research only.
    """

    def __init__(self, model_path: str = "models/pretrained/buffalo_l/det_10g.onnx", crop_scale: float = 2.0,
                 input_size: int = 192, min_iou: float = 0.3, providers: str = "cpu"):
        from face_ai.detection.detector import SCRFDDetector

        self.det = SCRFDDetector(model_path, score_threshold=0.3, input_size=input_size, providers=providers)
        self.crop_scale = crop_scale
        self.min_iou = min_iou

    def refine(self, image: np.ndarray, detection: FaceDetection) -> FaceDetection:
        from dataclasses import replace

        from face_ai.api.schemas import BoundingBox

        h, w = image.shape[:2]
        box = detection.bbox.expand(self.crop_scale, square=True)
        x1, y1 = int(np.floor(box.x1)), int(np.floor(box.y1))
        side = int(np.ceil(box.width))
        if side < 16:
            return detection
        pad = max(0, -x1, -y1, x1 + side - w, y1 + side - h)
        src = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else image
        crop = src[y1 + pad:y1 + pad + side, x1 + pad:x1 + pad + side]
        local = BoundingBox(detection.bbox.x1 - x1, detection.bbox.y1 - y1, detection.bbox.x2 - x1, detection.bbox.y2 - y1)
        cands = [d for d in self.det.detect(crop) if d.keypoints is not None]
        if not cands:
            return detection
        best = max(cands, key=lambda d: d.bbox.iou(local))
        if best.bbox.iou(local) < self.min_iou:
            return detection
        kps = best.keypoints + np.array([x1, y1], np.float32)
        return replace(detection, keypoints=kps.astype(np.float32))
