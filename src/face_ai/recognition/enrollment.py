"""Enroll consenting people from a few reference photos. No model retraining is involved."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from face_ai.detection.detector import FaceDetector
from face_ai.recognition.alignment import align_detection
from face_ai.recognition.embedder import FaceEmbedder
from face_ai.recognition.gallery import ConsentRecord, Gallery, GalleryError
from face_ai.recognition.quality import QualityConfig, assess_quality
from face_ai.utils.image_io import ImageLoadError, load_image

log = logging.getLogger(__name__)


@dataclass
class EnrollmentReport:
    person_id: str
    accepted: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (image ref, reason)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.accepted > 0


def _ref(i: int, src: str | Path | np.ndarray) -> str:
    return f"image[{i}]" if isinstance(src, np.ndarray) else Path(src).name


def enroll_person(
    person_id: str,
    images: list[str | Path | np.ndarray],
    detector: FaceDetector,
    embedder: FaceEmbedder,
    gallery: Gallery,
    consent: ConsentRecord | None,
    min_face_size: int = 64,
    min_detection_score: float = 0.8,
    min_quality: float = 0.6,
    quality_cfg: QualityConfig | None = None,
    consistency_threshold: float | None = 0.3,
    recommended_references: int = 3,
) -> EnrollmentReport:
    """Detect the face in each reference image, embed it, and add it to ``gallery``.

    Reference photos decide how well a person is recognized later, so the bar is higher than for
    probes: images with no face, several comparably large faces, a small face, or quality below
    ``min_quality`` (blur, turned head, bad light) are rejected with a reason. With 3+ accepted
    photos, a photo whose embedding disagrees with the mean of the others (cosine below
    ``consistency_threshold``) is rejected as probably a different person. Requires a granted
    ConsentRecord unless the gallery is an ephemeral evaluation gallery.
    """
    if not gallery.ephemeral and (consent is None or not consent.granted):
        raise GalleryError(f"refusing to enroll {person_id!r}: no granted consent record")
    if gallery.model != embedder.name:
        raise GalleryError(f"gallery model {gallery.model!r} != embedder {embedder.name!r}")

    report = EnrollmentReport(person_id)
    crops: list[np.ndarray] = []
    refs: list[str] = []
    hashes: list[str] = []
    for i, src in enumerate(images):
        ref = _ref(i, src)
        try:
            img = src if isinstance(src, np.ndarray) else load_image(src)
        except ImageLoadError as e:
            report.rejected.append((ref, f"unreadable: {e}"))
            continue
        dets = [d for d in detector.detect(img) if d.score >= min_detection_score]
        if not dets:
            report.rejected.append((ref, "no face detected above the confidence threshold"))
            continue
        dets.sort(key=lambda d: d.bbox.area, reverse=True)
        if len(dets) > 1 and dets[1].bbox.area > 0.5 * dets[0].bbox.area:
            report.rejected.append((ref, f"{len(dets)} faces of similar size; use a photo with only this person"))
            continue
        face = dets[0]
        if min(face.bbox.width, face.bbox.height) < min_face_size:
            report.rejected.append((ref, f"face too small ({face.bbox.width:.0f}px < {min_face_size}px)"))
            continue
        aligned = align_detection(img, face)
        q = assess_quality(face, aligned, quality_cfg)
        if q.score < min_quality:
            report.rejected.append((ref, f"photo quality {q.score:.2f} < {min_quality} ({', '.join(q.issues) or q.limiting})"))
            continue
        if len(dets) > 1:
            report.warnings.append(f"{ref}: {len(dets)} faces, used the largest")
        crops.append(aligned)
        refs.append(ref)
        hashes.append(hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest())

    if crops:
        emb = embedder.embed(crops)
        if consistency_threshold is not None and len(emb) >= 3:
            keep = []
            for i in range(len(emb)):
                others = np.delete(emb, i, axis=0).mean(0)
                sim = float(emb[i] @ others / (np.linalg.norm(others) + 1e-10))
                if sim < consistency_threshold:
                    report.rejected.append((refs[i], f"does not look like the other photos (cos={sim:.2f}); different person?"))
                else:
                    keep.append(i)
            emb = emb[keep]
            hashes = [hashes[i] for i in keep]
    if crops and len(emb):
        if gallery.ephemeral is False and person_id in gallery:
            existing = gallery.get(person_id).embeddings.mean(0)
            sim = float(emb.mean(0) @ existing / (np.linalg.norm(emb.mean(0)) * np.linalg.norm(existing) + 1e-10))
            if sim < 0.3:
                report.warnings.append(f"new references are dissimilar to existing ones (cos={sim:.2f}); check the photos")
        gallery.add(person_id, emb, consent, hashes)
        report.accepted = len(emb)
        total = gallery.get(person_id).num_references
        if total < recommended_references:
            report.warnings.append(f"{total} reference photo(s) stored; {recommended_references}+ varied, sharp, frontal "
                                   "photos make recognition noticeably more reliable")
    log.debug("enrollment %s: accepted=%d rejected=%d", person_id, report.accepted, len(report.rejected))
    return report
