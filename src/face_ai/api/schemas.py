"""Typed result objects shared by every component and by future iOS bindings.

Coordinates are pixels in the original image, origin at the top-left, x to the
right, y down. All objects serialize to plain JSON via ``to_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

UNKNOWN = "Unknown"

# 5-point keypoint order used everywhere (image-left first, matching the ArcFace template):
KEYPOINT_NAMES = ("eye_image_left", "eye_image_right", "nose_tip", "mouth_image_left", "mouth_image_right")


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box ``[x1, y1, x2, y2)`` in pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        return cls(float(x), float(y), float(x + w), float(y + h))

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def iou(self, other: "BoundingBox") -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(min(max(self.x1, 0), width), min(max(self.y1, 0), height),
                           min(max(self.x2, 0), width), min(max(self.y2, 0), height))

    def expand(self, scale: float, square: bool = False) -> "BoundingBox":
        cx, cy = self.center
        w, h = self.width * scale, self.height * scale
        if square:
            w = h = max(w, h)
        return BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class FaceDetection:
    """One detected face.

    score:      detector confidence in [0, 1] that this box contains a face. Not an identity score.
    keypoints:  (5, 2) array in ``KEYPOINT_NAMES`` order, or None if the detector has no landmarks.
    track_id:   temporary per-video tracking id; never a persistent identity.
    """

    bbox: BoundingBox
    score: float
    keypoints: np.ndarray | None = None
    track_id: int | None = None
    detector: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "score": round(float(self.score), 5),
            "keypoints": None if self.keypoints is None else np.round(self.keypoints, 2).tolist(),
            "track_id": self.track_id,
            "detector": self.detector,
        }


@dataclass
class FaceLandmarks:
    """Dense face-mesh landmarks.

    points:  (N, 3) array: x, y in image pixels; z is MediaPipe's relative depth (same scale as x,
             negative is towards the camera). N = 478 for MediaPipe Face Landmarker (468 mesh + 10 iris).
    """

    points: np.ndarray
    source: str = "mediapipe_face_landmarker"
    face_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"points": np.round(self.points, 2).tolist(), "source": self.source, "face_index": self.face_index}


@dataclass
class FaceEmbedding:
    """L2-normalized identity embedding. Compare only embeddings from the same ``model``."""

    vector: np.ndarray
    model: str

    @property
    def dim(self) -> int:
        return int(self.vector.shape[-1])


MATCH, NO_MATCH, UNCERTAIN = "match", "unknown", "uncertain"
UNCERTAIN_LABEL = "Uncertain"


@dataclass
class IdentityMatch:
    """Open-set identification result (1:N against the gallery).

    decision:    "match" (person_id set), "unknown" (confidently not anyone enrolled) or "uncertain"
                 (low face quality, score between the reject and accept thresholds, or two enrolled
                 people too close to tell apart). Uncertain faces are never assigned an identity.
    person_id:   enrolled id when decision == "match", else None.
    similarity:  cosine similarity to the best enrolled template (identity score), None when the
                 gallery is empty. Unrelated to the detection score.
    threshold:   accept threshold applied; ``reject_threshold`` the lower bound of the uncertain band.
    margin:      best minus second-best similarity (None with fewer than two enrolled people).
    estimated_far: estimated probability that a stranger scores >= ``similarity`` against one
                 enrolled person (from calibration score distributions), when available.
    quality:     FaceQuality.to_dict() of the probe face, when assessed.
    reason:      short human-readable explanation of the decision.
    candidates:  top-k (person_id, similarity) pairs, best first, for diagnostics.
    """

    person_id: str | None
    similarity: float | None
    threshold: float
    candidates: list[tuple[str, float]] = field(default_factory=list)
    decision: str = NO_MATCH
    reject_threshold: float | None = None
    margin: float | None = None
    estimated_far: float | None = None
    quality: dict[str, Any] | None = None
    reason: str = ""
    display_label: str | None = None  # overrides ``label`` for display (e.g. verify mode)

    @property
    def is_known(self) -> bool:
        return self.decision == MATCH and self.person_id is not None

    @property
    def label(self) -> str:
        if self.display_label is not None:
            return self.display_label
        if self.is_known:
            return self.person_id  # type: ignore[return-value]
        return UNCERTAIN_LABEL if self.decision == UNCERTAIN else UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "decision": self.decision,
            "person_id": self.person_id,
            "is_known": self.is_known,
            "similarity": None if self.similarity is None else round(float(self.similarity), 5),
            "threshold": self.threshold,
            "reject_threshold": self.reject_threshold,
            "margin": None if self.margin is None else round(float(self.margin), 5),
            "estimated_far": self.estimated_far,
            "quality": self.quality,
            "reason": self.reason,
            "candidates": [(c, round(float(s), 5)) for c, s in self.candidates],
        }


@dataclass
class VerificationResult:
    """1:1 check of one face against one enrolled person ("is this person X?").

    decision: "match" (same person), "unknown" (different person) or "uncertain".
    """

    person_id: str
    similarity: float | None
    decision: str
    threshold: float
    reject_threshold: float
    estimated_far: float | None = None
    estimated_frr: float | None = None
    quality: dict[str, Any] | None = None
    reason: str = ""

    @property
    def is_match(self) -> bool:
        return self.decision == MATCH

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id, "decision": self.decision, "is_match": self.is_match,
            "similarity": None if self.similarity is None else round(float(self.similarity), 5),
            "threshold": self.threshold, "reject_threshold": self.reject_threshold,
            "estimated_far": self.estimated_far, "estimated_frr": self.estimated_frr,
            "quality": self.quality, "reason": self.reason,
        }


@dataclass
class FaceAnalysis:
    """Everything computed for one face in one image/frame."""

    detection: FaceDetection
    landmarks: FaceLandmarks | None = None
    identity: IdentityMatch | None = None
    embedding: FaceEmbedding | None = None
    quality: Any = None  # recognition.quality.FaceQuality when identification ran

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "detection": self.detection.to_dict(),
            "landmarks": None if self.landmarks is None else self.landmarks.to_dict(),
            "identity": None if self.identity is None else self.identity.to_dict(),
        }
        if include_embedding and self.embedding is not None:
            d["embedding"] = {"model": self.embedding.model, "vector": self.embedding.vector.tolist()}
        return d
