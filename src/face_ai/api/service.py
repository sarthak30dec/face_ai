"""Stable, UI-independent entry points (the surface a future iOS port mirrors).

    svc = FaceAIService.from_configs()
    detections = svc.detect_faces(image)
    emb = svc.extract_embedding(svc.align(image, detections[0]))
    svc.enroll_person("alice", ["a1.jpg", "a2.jpg"], consent=ConsentRecord(True, "alice", "door unlock"))
    match = svc.identify_face(image, detections[0])          # 1:N: match / unknown / uncertain
    check = svc.verify_face(image, detections[0], "alice")   # 1:1: is this alice?
    lm = svc.extract_landmarks(image, detections[0])
    vis = svc.render_overlay(image, detections, [lm], [match])
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from face_ai.api.schemas import (FaceAnalysis, FaceDetection, FaceEmbedding, FaceLandmarks, IdentityMatch,
                                 VerificationResult)
from face_ai.config import GALLERY_DIR, load_config, resolve_path
from face_ai.detection.detector import FaceDetector, create_detector
from face_ai.landmarks.landmarker import FaceMeshLandmarker
from face_ai.recognition.alignment import align_detection, mesh_five_points, needs_roll_fix
from face_ai.recognition.calibration import ScoreCalibration
from face_ai.recognition.embedder import OnnxEmbedder, create_embedder
from face_ai.recognition.enrollment import EnrollmentReport
from face_ai.recognition.enrollment import enroll_person as _enroll_person
from face_ai.recognition.gallery import ConsentRecord, Gallery
from face_ai.recognition.matcher import Matcher
from face_ai.recognition.quality import FaceQuality, QualityConfig, assess_quality
from face_ai.utils.image_io import ensure_bgr
from face_ai.visualization.overlay import OverlayOptions, render_overlay

log = logging.getLogger(__name__)


class FaceAIService:
    """Lazily constructs detector / embedder / landmarker from config dicts."""

    def __init__(self, detection_cfg: dict[str, Any], recognition_cfg: dict[str, Any],
                 gallery: Gallery | None = None):
        self.detection_cfg = detection_cfg
        self.recognition_cfg = recognition_cfg
        self._detector: FaceDetector | None = None
        self._embedder: OnnxEmbedder | None = None
        self._landmarker: FaceMeshLandmarker | None = None
        self._gallery = gallery
        self._refiner = None
        self.quality_cfg = QualityConfig.from_dict(recognition_cfg.get("quality"))
        self.use_quality = bool(recognition_cfg.get("quality", {}).get("enabled", True))
        align_cfg = recognition_cfg.get("alignment", {})
        self.roll_correction = bool(align_cfg.get("roll_correction", True))
        self.roll_threshold_deg = float(align_cfg.get("roll_threshold_deg", 20.0))
        self.matcher = build_matcher(recognition_cfg)

    @classmethod
    def from_configs(cls, detection: str | dict = "detection", recognition: str | dict = "recognition",
                     **overrides: Any) -> "FaceAIService":
        det = load_config(detection) if isinstance(detection, str) else detection
        rec = load_config(recognition) if isinstance(recognition, str) else recognition
        if overrides.get("detector_backend"):
            det = {**det, "backend": overrides["detector_backend"]}
        if overrides.get("embedder_backend"):
            rec = {**rec, "backend": overrides["embedder_backend"]}
        svc = cls(det, rec)
        if overrides.get("detection_threshold") is not None:
            svc.detector.score_threshold = float(overrides["detection_threshold"])
        if overrides.get("identity_threshold") is not None:
            svc.matcher.threshold = float(overrides["identity_threshold"])
        return svc

    # --- components ---------------------------------------------------------------------------
    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = create_detector(self.detection_cfg)
        return self._detector

    @property
    def embedder(self) -> OnnxEmbedder:
        if self._embedder is None:
            self._embedder = create_embedder(self.recognition_cfg)
        return self._embedder

    @property
    def landmarker(self) -> FaceMeshLandmarker:
        if self._landmarker is None:
            self._landmarker = FaceMeshLandmarker(**self.recognition_cfg.get("landmarks", {}))
        return self._landmarker

    @property
    def gallery(self) -> Gallery:
        if self._gallery is None:
            self._gallery = Gallery(self.embedder.name, self.embedder.dim)
        return self._gallery

    def load_gallery(self, directory: str | Path = GALLERY_DIR) -> Gallery:
        self._gallery = Gallery.load(directory, expected_model=self.embedder.name)
        return self._gallery

    def save_gallery(self, directory: str | Path = GALLERY_DIR) -> Path:
        return self.gallery.save(directory)

    # --- stable interface -----------------------------------------------------------------------
    def detect_faces(self, image: np.ndarray) -> list[FaceDetection]:
        return self.detector.detect(ensure_bgr(image))

    @property
    def keypoint_refiner(self):
        """Optional second-stage keypoint detector (``alignment.refine_keypoints`` in the config)."""
        cfg = self.recognition_cfg.get("alignment", {})
        backend_cfg = self.recognition_cfg.get(self.recognition_cfg.get("backend", "sface"), {})
        enabled = backend_cfg.get("refine_keypoints", cfg.get("refine_keypoints", False))
        if self._refiner is None and enabled:
            from face_ai.recognition.alignment import KeypointRefiner

            self._refiner = KeypointRefiner(**cfg.get("refiner", {}))
        return self._refiner

    def align(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        image = ensure_bgr(image)
        if self.keypoint_refiner is not None:
            detection = self.keypoint_refiner.refine(image, detection)
        if self.roll_correction and detection.keypoints is not None:
            detection = self.check_keypoints(image, detection)
        return align_detection(image, detection)

    def check_keypoints(self, image: np.ndarray, detection: FaceDetection) -> FaceDetection:
        """Cross-check detector keypoints against the face mesh (which stays accurate on tilted faces).

        Past ~45 degrees of tilt YuNet still returns an upright-looking layout, just misplaced, so its
        own keypoints can't reveal the problem. If the mesh shows a tilt above ``roll_threshold_deg`` or
        disagrees with the detector by more than 35 % of the eye distance, the face is re-aligned upright.
        Upright faces keep the detector keypoints, so the calibrated thresholds still apply.
        """
        lm = self.landmarker.extract(image, detection)
        if lm is None:
            return detection
        five = mesh_five_points(lm.points)
        eye = five[1] - five[0]
        iod = float(np.linalg.norm(eye)) or 1.0
        roll = abs(float(np.degrees(np.arctan2(eye[1], eye[0]))))
        disagree = float(np.linalg.norm(np.asarray(detection.keypoints) - five, axis=1).mean()) / iod
        if roll <= self.roll_threshold_deg and disagree <= 0.35 and not needs_roll_fix(detection, self.roll_threshold_deg):
            return detection
        return self.upright_keypoints(image, detection, five)

    def upright_keypoints(self, image: np.ndarray, detection: FaceDetection,
                          five: np.ndarray | None = None) -> FaceDetection:
        """Keypoints for a strongly tilted face.

        Detector keypoints degrade past ~45 degrees of in-plane tilt, while the face mesh stays
        accurate. The mesh gives the true tilt; the face crop is rotated upright, re-detected there
        and the keypoints are mapped back. If re-detection fails, 5 points taken from the mesh are used.
        """
        if five is None:
            lm = self.landmarker.extract(image, detection)
            if lm is None:
                return detection
            five = mesh_five_points(lm.points)
        right_eye, left_eye = five[0], five[1]  # subject's right eye sits image-left on an upright face
        angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))
        b = detection.bbox
        side = int(round(max(b.width, b.height) * 2.2))
        cx, cy = b.center
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        h, w = image.shape[:2]
        pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
        src = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else image
        crop = src[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
        rot = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
        upright = cv2.warpAffine(crop, rot, (side, side), borderMode=cv2.BORDER_REPLICATE)
        cands = [d for d in self.detector.detect(upright) if d.keypoints is not None]
        if cands:
            best = min(cands, key=lambda d: np.hypot(d.bbox.center[0] - side / 2, d.bbox.center[1] - side / 2))
            inv = cv2.invertAffineTransform(rot)
            kps = (inv[:, :2] @ best.keypoints.T).T + inv[:, 2] + np.array([x0, y0], np.float32)
        else:
            kps = five
        return dataclasses.replace(detection, keypoints=kps.astype(np.float32))

    def assess_quality(self, detection: FaceDetection, aligned: np.ndarray) -> FaceQuality | None:
        return assess_quality(detection, aligned, self.quality_cfg) if self.use_quality else None

    def extract_embedding(self, aligned_face: np.ndarray) -> FaceEmbedding:
        return FaceEmbedding(self.embedder.embed([aligned_face])[0], self.embedder.name)

    def enroll_person(self, person_id: str, images: Sequence[str | Path | np.ndarray],
                      consent: ConsentRecord, **kwargs: Any) -> EnrollmentReport:
        enroll_cfg = {**self.recognition_cfg.get("enrollment", {}), **kwargs}
        enroll_cfg.setdefault("quality_cfg", self.quality_cfg)
        if enroll_cfg.get("consistency_threshold", "auto") == "auto":
            # a reference that would be *rejected* as a different person in a 1:1 check does not belong
            enroll_cfg["consistency_threshold"] = self.matcher.verify_reject_threshold or 0.3
        refiner = self.keypoint_refiner
        detector = self.detector if refiner is None else _RefinedDetector(self.detector, refiner)
        return _enroll_person(person_id, list(images), detector, self.embedder, self.gallery, consent, **enroll_cfg)

    def embed_face(self, image: np.ndarray, detection: FaceDetection) -> tuple[FaceEmbedding, FaceQuality | None]:
        aligned = self.align(image, detection)
        return self.extract_embedding(aligned), self.assess_quality(detection, aligned)

    def identify_face(self, image: np.ndarray, detection: FaceDetection,
                      gallery: Gallery | None = None) -> IdentityMatch:
        emb, q = self.embed_face(image, detection)
        return self.matcher.identify(emb.vector, gallery or self.gallery, q)

    def identify_embedding(self, embedding: FaceEmbedding, gallery: Gallery | None = None,
                           quality: FaceQuality | None = None) -> IdentityMatch:
        return self.matcher.identify(embedding.vector, gallery or self.gallery, quality)

    def verify_face(self, image: np.ndarray, detection: FaceDetection, person_id: str,
                    gallery: Gallery | None = None) -> VerificationResult:
        """1:1 check: is the face in ``detection`` the enrolled person ``person_id``?"""
        emb, q = self.embed_face(image, detection)
        return self.matcher.verify_against(emb.vector, person_id, gallery or self.gallery, q)

    def extract_landmarks(self, image: np.ndarray, detection: FaceDetection) -> FaceLandmarks | None:
        return self.landmarker.extract(ensure_bgr(image), detection)

    def render_overlay(self, image: np.ndarray, detections: Sequence[FaceDetection],
                       landmarks: Sequence[FaceLandmarks | None] | None = None,
                       identities: Sequence[IdentityMatch | None] | None = None,
                       options: OverlayOptions | None = None) -> np.ndarray:
        return render_overlay(image, detections, landmarks, identities, options)

    def analyze(self, image: np.ndarray, with_landmarks: bool = True, with_identity: bool = True,
                keep_embeddings: bool = False) -> list[FaceAnalysis]:
        """Detection -> alignment -> embedding/identity -> landmarks for every face in ``image``."""
        image = ensure_bgr(image)
        dets = self.detect_faces(image)
        results = [FaceAnalysis(d) for d in dets]
        if with_identity and dets:
            crops = [self.align(image, d) for d in dets]
            vecs = self.embedder.embed(crops)
            quals = [self.assess_quality(d, c) for d, c in zip(dets, crops)]
            matches = self.matcher.identify_batch(vecs, self.gallery, quals)
            for r, v, m, q in zip(results, vecs, matches, quals):
                r.identity = m
                r.quality = q
                if keep_embeddings:
                    r.embedding = FaceEmbedding(v, self.embedder.name)
        if with_landmarks:
            for r in results:
                r.landmarks = self.extract_landmarks(image, r.detection)
        return results

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()


class _RefinedDetector:
    """Detector wrapper whose detections carry refined keypoints (used for enrollment)."""

    def __init__(self, detector: FaceDetector, refiner):
        self._d, self._r = detector, refiner
        self.name, self.score_threshold = detector.name, detector.score_threshold

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        return [self._r.refine(image, d) for d in self._d.detect(image)]


def build_matcher(recognition_cfg: dict[str, Any]) -> Matcher:
    """Matcher for the active embedder: thresholds come from its calibration file
    (``matching.profiles.<backend>.calibration``); explicit values in the profile override them."""
    backend = recognition_cfg.get("backend", "sface")
    m = recognition_cfg.get("matching", {})
    prof = dict(m.get("profiles", {}).get(backend, {}))
    calib = None
    if prof.get("calibration") and resolve_path(prof["calibration"]).exists():
        calib = ScoreCalibration.load(resolve_path(prof["calibration"]))
    elif prof.get("calibration"):
        log.warning("calibration file %s missing; using fallback thresholds", prof["calibration"])
    if calib is not None:
        bcfg = recognition_cfg.get(backend, {})
        refine = bcfg.get("refine_keypoints", recognition_cfg.get("alignment", {}).get("refine_keypoints", False))
        expected = f"{backend}{'_flip' if bcfg.get('flip_augment') else ''}_{'scrfd' if refine else 'yunet'}"
        if calib.variant != expected:
            log.warning("calibration %s was made for %s but the active pipeline is %s; thresholds may be off. "
                        "Re-run scripts/evaluate_recognition.py --update-config", prof["calibration"], calib.variant, expected)
    ident = calib.identification if calib else {}
    ver = calib.verification if calib else {}

    def pick(key: str, fallback: float | None) -> float | None:
        v = prof.get(key)
        return float(v) if v is not None else (float(fallback) if fallback is not None else None)

    accept = pick("accept", ident.get("accept", m.get("thresholds", {}).get(backend, 0.5)))
    return Matcher(
        threshold=accept,
        strategy=m.get("strategy", "mean"),
        top_k=int(m.get("top_k", 3)),
        reject_threshold=pick("reject", ident.get("reject", accept)),
        margin=float(m.get("margin", 0.0)),
        verify_threshold=pick("verify_accept", ver.get("accept", accept)),
        verify_reject_threshold=pick("verify_reject", ver.get("reject", accept)),
        calibration=calib,
    )


# Module-level convenience functions backed by a default service.
_default: FaceAIService | None = None


def _svc() -> FaceAIService:
    global _default
    if _default is None:
        _default = FaceAIService.from_configs()
    return _default


def detect_faces(image: np.ndarray) -> list[FaceDetection]:
    return _svc().detect_faces(image)


def extract_embedding(aligned_face: np.ndarray) -> FaceEmbedding:
    return _svc().extract_embedding(aligned_face)


def enroll_person(person_id: str, images: Sequence[str | Path | np.ndarray], consent: ConsentRecord) -> EnrollmentReport:
    return _svc().enroll_person(person_id, images, consent)


def identify_face(image: np.ndarray, face: FaceDetection, gallery: Gallery | None = None) -> IdentityMatch:
    return _svc().identify_face(image, face, gallery)


def verify_face(image: np.ndarray, face: FaceDetection, person_id: str) -> VerificationResult:
    return _svc().verify_face(image, face, person_id)


def extract_landmarks(image: np.ndarray, face: FaceDetection) -> FaceLandmarks | None:
    return _svc().extract_landmarks(image, face)


__all__ = [
    "FaceAIService", "detect_faces", "extract_embedding", "enroll_person", "identify_face", "verify_face",
    "extract_landmarks", "render_overlay", "build_matcher",
]
