"""1:N open-set identification metrics.

Terminology (ISO/IEC 19795-1 / NIST FRVT):
* Rank-1 accuracy: closed-set; fraction of known probes whose top match is the right person.
* DIR(τ): detection & identification rate; known probe ranked correctly at rank 1 AND score >= τ.
* FPIR(τ): false-positive identification rate; unknown probe whose top score >= τ.
* Unknown rejection rate = 1 - FPIR.
* Misidentification rate: known probe accepted (score >= τ) as the WRONG person.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def top1(similarities: np.ndarray, gallery_ids: Sequence[str]) -> tuple[list[str], np.ndarray]:
    idx = similarities.argmax(1)
    return [gallery_ids[i] for i in idx], similarities[np.arange(len(idx)), idx]


def threshold_for_fpir(unknown_top_scores: np.ndarray, target_fpir: float) -> float:
    """Smallest threshold whose FPIR on these unknown probes is <= target."""
    s = np.sort(unknown_top_scores)[::-1]
    k = int(np.floor(target_fpir * len(s)))  # number of unknowns allowed above the threshold
    if k >= len(s):
        return float(s[-1])
    return float(np.nextafter(s[k], np.float32(np.inf)))


def open_set_metrics(known_sims: np.ndarray, known_labels: Sequence[str], unknown_sims: np.ndarray,
                     gallery_ids: Sequence[str], threshold: float) -> dict[str, Any]:
    pred_k, score_k = top1(known_sims, gallery_ids)
    _, score_u = top1(unknown_sims, gallery_ids)
    correct = np.array([p == t for p, t in zip(pred_k, known_labels)])
    accepted = score_k >= threshold
    fpir = float((score_u >= threshold).mean()) if len(score_u) else 0.0
    return {
        "threshold": float(threshold),
        "known_probes": int(len(known_labels)),
        "unknown_probes": int(len(score_u)),
        "rank1_accuracy_closed_set": float(correct.mean()) if len(correct) else 0.0,
        "dir": float((correct & accepted).mean()) if len(correct) else 0.0,
        "known_rejected_as_unknown": float((~accepted).mean()) if len(correct) else 0.0,
        "misidentification_rate": float((~correct & accepted).mean()) if len(correct) else 0.0,
        "fpir": fpir,
        "unknown_rejection_rate": 1.0 - fpir,
    }


def dir_fpir_curve(known_sims: np.ndarray, known_labels: Sequence[str], unknown_sims: np.ndarray,
                   gallery_ids: Sequence[str]) -> dict[str, list[float]]:
    """DIR and FPIR evaluated at every distinct top-1 score (exact step curve)."""
    pred_k, score_k = top1(known_sims, gallery_ids)
    _, score_u = top1(unknown_sims, gallery_ids)
    correct = np.array([p == t for p, t in zip(pred_k, known_labels)])
    ths = np.unique(np.concatenate([score_k, score_u]))
    ok = np.sort(score_k[correct])
    unk = np.sort(score_u)
    dir_ = (len(ok) - np.searchsorted(ok, ths, side="left")) / max(len(correct), 1)
    fpir = (len(unk) - np.searchsorted(unk, ths, side="left")) / max(len(unk), 1)
    return {"thresholds": ths.tolist(), "dir": dir_.tolist(), "fpir": fpir.tolist()}
