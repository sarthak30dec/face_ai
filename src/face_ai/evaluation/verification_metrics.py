"""1:1 verification metrics: ROC/AUC, FAR/FRR, TAR@FAR, EER and LFW 10-fold accuracy."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import auc, roc_curve


def roc(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    fpr, tpr, thr = roc_curve(labels.astype(int), scores)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thr, "auc": float(auc(fpr, tpr))}


def far_frr(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """FAR = impostor pairs accepted / impostor pairs; FRR = genuine pairs rejected / genuine pairs."""
    labels = labels.astype(bool)
    accept = scores >= threshold
    far = float(accept[~labels].mean()) if (~labels).any() else 0.0
    frr = float((~accept[labels]).mean()) if labels.any() else 0.0
    return {"threshold": float(threshold), "far": far, "frr": frr, "tar": 1 - frr,
            "accuracy": float((accept == labels).mean())}


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far_targets: Sequence[float]) -> list[dict[str, float]]:
    """Largest TAR whose FAR does not exceed each target (threshold chosen on this same data)."""
    r = roc(scores, labels)
    out = []
    for t in far_targets:
        ok = np.where(r["fpr"] <= t)[0]
        i = ok[-1]
        out.append({"far_target": t, "tar": float(r["tpr"][i]), "far_actual": float(r["fpr"][i]),
                    "threshold": float(r["thresholds"][i])})
    return out


def eer(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    r = roc(scores, labels)
    fnr = 1 - r["tpr"]
    i = int(np.nanargmin(np.abs(fnr - r["fpr"])))
    return {"eer": float((fnr[i] + r["fpr"][i]) / 2), "threshold": float(r["thresholds"][i])}


def best_accuracy_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    cand = np.unique(scores)
    mids = np.concatenate([[cand[0] - 1e-6], (cand[1:] + cand[:-1]) / 2, [cand[-1] + 1e-6]])
    labels = labels.astype(bool)
    accs = [((scores >= t) == labels).mean() for t in mids]
    return float(mids[int(np.argmax(accs))])


def kfold_accuracy(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    """LFW protocol: for each fold pick the accuracy-maximizing threshold on the other folds,
    then measure accuracy on the held-out fold. Thresholds never see their test fold."""
    accs, thrs = [], []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        t = best_accuracy_threshold(scores[tr], labels[tr])
        accs.append(float(((scores[te] >= t) == labels[te].astype(bool)).mean()))
        thrs.append(t)
    return {"mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
            "fold_accuracies": accs, "fold_thresholds": thrs}
