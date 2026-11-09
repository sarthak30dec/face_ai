"""Face image quality for recognition.

Low-quality faces (tiny, blurred, strongly turned, badly lit) produce unreliable embeddings: both
false matches and false rejections concentrate there. Instead of guessing, the matcher returns
``uncertain`` for faces below the quality bar.

All measures use the 5 detector keypoints and the 112x112 aligned crop, so they cost < 0.5 ms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from face_ai.api.schemas import FaceDetection


@dataclass
class QualityConfig:
    """Factor ranges: each factor is 0 at ``*_bad`` and 1 at ``*_good`` (linear in between)."""

    iod_bad: float = 12.0          # inter-ocular distance in source pixels
    iod_good: float = 32.0
    sharp_bad: float = 15.0        # variance of Laplacian on the aligned 112x112 crop
    sharp_good: float = 120.0
    yaw_good: float = 0.15         # |nose offset from eye midpoint| / inter-ocular distance
    yaw_bad: float = 0.55
    roll_good: float = 15.0        # degrees
    roll_bad: float = 45.0
    brightness_bad: float = 35.0   # mean gray level of the aligned crop (too dark / too bright)
    brightness_good: float = 70.0
    contrast_bad: float = 12.0     # std of gray level
    contrast_good: float = 30.0
    det_bad: float = 0.5
    det_good: float = 0.85
    min_quality: float = 0.35      # below this the face is "uncertain" for identification

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "QualityConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FaceQuality:
    """score in [0, 1] = the weakest factor; ``limiting`` names it; ``issues`` lists factors < 0.5."""

    score: float
    acceptable: bool
    limiting: str
    factors: dict[str, float]
    measures: dict[str, float]
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = round(self.score, 3)
        d["factors"] = {k: round(v, 3) for k, v in self.factors.items()}
        d["measures"] = {k: round(v, 3) for k, v in self.measures.items()}
        return d


_ISSUE_TEXT = {
    "size": "face too small",
    "sharpness": "blurred",
    "pose_yaw": "head turned sideways",
    "pose_roll": "head tilted",
    "brightness": "too dark or too bright",
    "contrast": "low contrast",
    "detection": "uncertain detection",
}


def _ramp(x: float, bad: float, good: float) -> float:
    if good == bad:
        return 1.0
    return float(np.clip((x - bad) / (good - bad), 0.0, 1.0))


def pose_from_keypoints(kps: np.ndarray) -> tuple[float, float, float]:
    """(inter-ocular distance px, yaw ratio, roll degrees) from 5 keypoints (ArcFace order)."""
    le, re, nose = kps[0], kps[1], kps[2]
    eye_vec = re - le
    iod = float(np.linalg.norm(eye_vec))
    if iod < 1e-6:
        return 0.0, 1.0, 90.0
    roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
    unit = eye_vec / iod
    yaw = float(np.dot(nose - (le + re) / 2, unit) / iod)
    return iod, yaw, roll


def assess_quality(detection: FaceDetection, aligned: np.ndarray, cfg: QualityConfig | None = None) -> FaceQuality:
    cfg = cfg or QualityConfig()
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    core = gray[16:104, 16:96]  # central face region, avoids warp borders and background
    sharp = float(cv2.Laplacian(core, cv2.CV_64F).var())
    mean, std = float(core.mean()), float(core.std())
    if detection.keypoints is not None:
        iod, yaw, roll = pose_from_keypoints(np.asarray(detection.keypoints, np.float32))
    else:  # no keypoints: approximate eye distance from box size, pose unknown (assume frontal)
        iod, yaw, roll = 0.4 * min(detection.bbox.width, detection.bbox.height), 0.0, 0.0
    bright_dev = abs(mean - 128.0)  # distance from mid-gray; 128-70=58 good, 128-35=93 bad
    factors = {
        "size": _ramp(iod, cfg.iod_bad, cfg.iod_good),
        "sharpness": _ramp(np.log(max(sharp, 1e-3)), np.log(cfg.sharp_bad), np.log(cfg.sharp_good)),
        "pose_yaw": _ramp(-abs(yaw), -cfg.yaw_bad, -cfg.yaw_good),
        "pose_roll": _ramp(-abs(roll), -cfg.roll_bad, -cfg.roll_good),
        "brightness": _ramp(-bright_dev, -(128 - cfg.brightness_bad), -(128 - cfg.brightness_good)),
        "contrast": _ramp(std, cfg.contrast_bad, cfg.contrast_good),
        "detection": _ramp(detection.score, cfg.det_bad, cfg.det_good),
    }
    limiting = min(factors, key=factors.get)
    score = factors[limiting]
    return FaceQuality(
        score=score,
        acceptable=score >= cfg.min_quality,
        limiting=limiting,
        factors=factors,
        measures={"inter_ocular_px": iod, "yaw_ratio": yaw, "roll_deg": roll, "sharpness": sharp,
                  "brightness": mean, "contrast": std, "det_score": float(detection.score)},
        issues=[_ISSUE_TEXT[k] for k, v in factors.items() if v < 0.5],
    )
