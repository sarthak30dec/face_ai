"""Cosine-similarity matching with an open-set decision: match / unknown / uncertain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from face_ai.api.schemas import MATCH, NO_MATCH, UNCERTAIN, IdentityMatch, VerificationResult
from face_ai.recognition.calibration import ScoreCalibration
from face_ai.recognition.gallery import Gallery
from face_ai.recognition.quality import FaceQuality


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of ``a`` (n, d) and ``b`` (m, d) -> (n, m). Inputs need not be normalized."""
    a = np.atleast_2d(a).astype(np.float32)
    b = np.atleast_2d(b).astype(np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-10)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-10)
    return a @ b.T


def score_against_gallery(embeddings: np.ndarray, gallery: Gallery, strategy: str = "mean") -> tuple[list[str], np.ndarray]:
    """Per-person similarity matrix (n_probes, n_people).

    strategy="mean": compare to each person's averaged template.
    strategy="max":  best similarity over each person's reference embeddings.
    """
    if strategy == "mean":
        ids, templates = gallery.templates()
        return ids, cosine_similarity(embeddings, templates) if ids else np.zeros((len(np.atleast_2d(embeddings)), 0))
    if strategy == "max":
        owners, refs = gallery.references()
        ids = gallery.people()
        sims = cosine_similarity(embeddings, refs) if owners else np.zeros((len(np.atleast_2d(embeddings)), 0))
        out = np.full((sims.shape[0], len(ids)), -1.0, dtype=np.float32)
        col = {pid: i for i, pid in enumerate(ids)}
        for j, pid in enumerate(owners):
            out[:, col[pid]] = np.maximum(out[:, col[pid]], sims[:, j])
        return ids, out
    raise ValueError(f"unknown strategy {strategy!r}")


@dataclass
class Matcher:
    """Decision rule (1:N):

    1. face quality below the bar           -> uncertain
    2. best similarity >= threshold and best - second >= margin -> match
    3. best similarity >= threshold but margin too small         -> uncertain (two people too alike)
    4. best similarity < reject_threshold                         -> unknown
    5. otherwise (between the two thresholds)                     -> uncertain

    ``threshold`` is calibrated for a target false-positive identification rate, ``reject_threshold``
    for a target false-negative rate; when the score distributions are well separated they coincide
    and there is no uncertain band.
    """

    threshold: float
    strategy: str = "mean"
    top_k: int = 3
    reject_threshold: float | None = None
    margin: float = 0.0
    verify_threshold: float | None = None
    verify_reject_threshold: float | None = None
    calibration: ScoreCalibration | None = None

    @property
    def reject(self) -> float:
        return min(self.threshold, self.reject_threshold) if self.reject_threshold is not None else self.threshold

    def _far(self, score: float | None) -> float | None:
        return None if self.calibration is None or score is None else self.calibration.far(score)

    def identify(self, embedding: np.ndarray, gallery: Gallery, quality: FaceQuality | None = None) -> IdentityMatch:
        return self.identify_batch(np.atleast_2d(embedding), gallery, [quality])[0]

    def identify_batch(self, embeddings: np.ndarray, gallery: Gallery,
                       qualities: Sequence[FaceQuality | None] | None = None) -> list[IdentityMatch]:
        embeddings = np.atleast_2d(embeddings)
        qualities = list(qualities) if qualities is not None else [None] * len(embeddings)
        if len(gallery) == 0:
            return [IdentityMatch(None, None, self.threshold, decision=NO_MATCH, reject_threshold=self.reject,
                                  quality=q.to_dict() if q else None, reason="nobody enrolled")
                    for q in qualities]
        ids, sims = score_against_gallery(embeddings, gallery, self.strategy)
        results = []
        for row, q in zip(sims, qualities):
            order = np.argsort(row)[::-1]
            cands = [(ids[i], float(row[i])) for i in order[: self.top_k]]
            best_id, best = cands[0]
            margin = float(row[order[0]] - row[order[1]]) if len(order) > 1 else None
            if q is not None and not q.acceptable:
                decision, pid = UNCERTAIN, None
                reason = f"low face quality ({', '.join(q.issues) or q.limiting})"
            elif best >= self.threshold:
                if margin is not None and margin < self.margin:
                    decision, pid = UNCERTAIN, None
                    reason = f"{cands[0][0]} and {cands[1][0]} are too similar (margin {margin:.3f} < {self.margin})"
                else:
                    decision, pid, reason = MATCH, best_id, "similarity above accept threshold"
            elif best < self.reject:
                decision, pid, reason = NO_MATCH, None, "not similar to anyone enrolled"
            else:
                decision, pid, reason = UNCERTAIN, None, "similarity between reject and accept thresholds"
            results.append(IdentityMatch(pid, best, self.threshold, cands, decision, self.reject, margin,
                                         self._far(best), q.to_dict() if q else None, reason))
        return results

    def verify_against(self, embedding: np.ndarray, person_id: str, gallery: Gallery,
                       quality: FaceQuality | None = None) -> VerificationResult:
        """1:1: is this face ``person_id``? Uses the 1:1 thresholds (calibrated for a target FAR)."""
        accept = self.verify_threshold if self.verify_threshold is not None else self.threshold
        reject = min(accept, self.verify_reject_threshold) if self.verify_reject_threshold is not None else accept
        if person_id not in gallery:
            raise KeyError(f"{person_id!r} is not enrolled")
        ids, sims = score_against_gallery(np.atleast_2d(embedding), gallery, self.strategy)
        s = float(sims[0, ids.index(person_id)])
        frr = None if self.calibration is None else self.calibration.frr(s)
        qd = quality.to_dict() if quality else None
        if quality is not None and not quality.acceptable:
            return VerificationResult(person_id, s, UNCERTAIN, accept, reject, self._far(s), frr, qd,
                                      f"low face quality ({', '.join(quality.issues) or quality.limiting})")
        if s >= accept:
            return VerificationResult(person_id, s, MATCH, accept, reject, self._far(s), frr, qd, "same person")
        if s < reject:
            return VerificationResult(person_id, s, NO_MATCH, accept, reject, self._far(s), frr, qd, "different person")
        return VerificationResult(person_id, s, UNCERTAIN, accept, reject, self._far(s), frr, qd,
                                  "similarity between reject and accept thresholds")

    def verify(self, a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
        """Raw embedding-to-embedding check with the 1:1 accept threshold."""
        s = float(cosine_similarity(a, b)[0, 0])
        return s >= (self.verify_threshold if self.verify_threshold is not None else self.threshold), s
