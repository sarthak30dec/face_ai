"""WIDER FACE evaluation, a faithful Python port of the official ``eval_tools`` (evaluation.m).

Protocol (official): IoU >= 0.5; scores min-max normalized over the whole prediction set; 1000
score thresholds; faces outside the setting's keep list (easy/medium/hard ``gt_list``) are
*ignored* (matching them is neither a TP nor a FP); AP is the VOC all-point interpolated area.

``operating_point`` additionally reports precision / recall / false detections / missed faces at a
fixed raw detector threshold, using the same matching rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

SETTINGS = ("easy", "medium", "hard")
THRESH_NUM = 1000
SIZE_BINS = ((0, 16, "tiny<16px"), (16, 32, "small16-32"), (32, 64, "medium32-64"), (64, 1e9, "large>=64"))


@dataclass
class WiderGroundTruth:
    """Per image: boxes (N,4 xywh) and 1-based keep indices for each setting."""

    images: dict[str, np.ndarray]                # "event/name" (no extension) -> boxes
    keep: dict[str, dict[str, np.ndarray]]       # setting -> image -> keep indices (1-based)


def load_wider_gt(gt_dir: str | Path) -> WiderGroundTruth:
    gt_dir = Path(gt_dir)
    main = loadmat(gt_dir / "wider_face_val.mat")
    events = [e[0][0] for e in main["event_list"]]
    images: dict[str, np.ndarray] = {}
    order: list[list[str]] = []
    for i, ev in enumerate(events):
        names = [f[0][0] for f in main["file_list"][i][0]]
        order.append(names)
        for j, n in enumerate(names):
            images[f"{ev}/{n}"] = main["face_bbx_list"][i][0][j][0].astype(np.float64)
    keep: dict[str, dict[str, np.ndarray]] = {}
    for s in SETTINGS:
        m = loadmat(gt_dir / f"wider_{s}_val.mat")
        keep[s] = {}
        for i, ev in enumerate(events):
            for j, n in enumerate(order[i]):
                keep[s][f"{ev}/{n}"] = m["gt_list"][i][0][j][0].reshape(-1).astype(np.int64)
    return WiderGroundTruth(images, keep)


def bbox_overlaps(boxes: np.ndarray, query: np.ndarray) -> np.ndarray:
    """IoU with the official +1 pixel convention; boxes are x1y1x2y2."""
    b = boxes[:, None, :]
    q = query[None, :, :]
    iw = np.minimum(b[..., 2], q[..., 2]) - np.maximum(b[..., 0], q[..., 0]) + 1
    ih = np.minimum(b[..., 3], q[..., 3]) - np.maximum(b[..., 1], q[..., 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area_b = (b[..., 2] - b[..., 0] + 1) * (b[..., 3] - b[..., 1] + 1)
    area_q = (q[..., 2] - q[..., 0] + 1) * (q[..., 3] - q[..., 1] + 1)
    union = area_b + area_q - inter
    return np.where(inter > 0, inter / union, 0.0)


def image_eval(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray, iou_thresh: float):
    """pred (P,5 xywhs sorted by score desc), gt (G,4 xywh), ignore (G,) 1 = kept, 0 = ignored.

    Returns cumulative recall per prediction, the proposal flags (1 counts as a proposal, -1 matched
    an ignored face) and the per-gt recall flags.
    """
    p = pred[:, :4].copy()
    g = gt.copy()
    p[:, 2] += p[:, 0]
    p[:, 3] += p[:, 1]
    g[:, 2] += g[:, 0]
    g[:, 3] += g[:, 1]
    overlaps = bbox_overlaps(p, g)
    pred_recall = np.zeros(len(p))
    recall_list = np.zeros(len(g))
    proposal_list = np.ones(len(p))
    for h in range(len(p)):
        idx = int(overlaps[h].argmax())
        if overlaps[h, idx] >= iou_thresh:
            if ignore[idx] == 0:
                recall_list[idx] = -1
                proposal_list[h] = -1
            elif recall_list[idx] == 0:
                recall_list[idx] = 1
        pred_recall[h] = (recall_list == 1).sum()
    return pred_recall, proposal_list, recall_list


def img_pr_info(scores: np.ndarray, proposal_list: np.ndarray, pred_recall: np.ndarray) -> np.ndarray:
    """Vectorized form of the official loop; ``scores`` must be sorted descending."""
    thresh = 1 - (np.arange(THRESH_NUM) + 1) / THRESH_NUM
    n_above = (scores[None, :] >= thresh[:, None]).sum(1)  # = last index with score >= thresh, + 1
    cum_prop = np.concatenate([[0], np.cumsum(proposal_list == 1)])
    cum_recall = np.concatenate([[0], pred_recall])
    return np.stack([cum_prop[n_above], cum_recall[n_above]], axis=1).astype(np.float64)


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def _sorted(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 5)
    return pred[np.argsort(-pred[:, 4], kind="stable")]


def evaluate_wider(preds: dict[str, np.ndarray], gt: WiderGroundTruth, iou_thresh: float = 0.5) -> dict[str, Any]:
    """preds: "event/name" (no extension) -> (P, 5) [x, y, w, h, score]. Missing images = no detections."""
    preds = {k: _sorted(v) for k, v in preds.items()}
    allscores = np.concatenate([v[:, 4] for v in preds.values() if len(v)]) if preds else np.zeros(0)
    lo, hi = (allscores.min(), allscores.max()) if allscores.size else (0.0, 1.0)
    span = hi - lo if hi > lo else 1.0
    normed = {k: np.column_stack([v[:, :4], (v[:, 4] - lo) / span]) if len(v) else v for k, v in preds.items()}
    results: dict[str, Any] = {"iou_threshold": iou_thresh, "protocol": "WIDER FACE official (eval_tools v2017-03-31), Python port"}
    for s in SETTINGS:
        count_face = 0
        pr_curve = np.zeros((THRESH_NUM, 2))
        for img, gt_boxes in gt.images.items():
            keep = gt.keep[s][img]
            count_face += len(keep)
            pred = normed.get(img, np.zeros((0, 5)))
            if len(gt_boxes) == 0 or len(pred) == 0:
                continue
            ignore = np.zeros(len(gt_boxes))
            if len(keep):
                ignore[keep - 1] = 1
            pred_recall, proposal_list, _ = image_eval(pred, gt_boxes, ignore, iou_thresh)
            pr_curve += img_pr_info(pred[:, 4], proposal_list, pred_recall)
        precision = np.divide(pr_curve[:, 1], pr_curve[:, 0], out=np.zeros(THRESH_NUM), where=pr_curve[:, 0] > 0)
        recall = pr_curve[:, 1] / max(count_face, 1)
        results[s] = {"ap": voc_ap(recall, precision), "faces": int(count_face),
                      "precision_curve": precision.tolist(), "recall_curve": recall.tolist()}
    return results


def operating_point(preds: dict[str, np.ndarray], gt: WiderGroundTruth, score_threshold: float,
                    setting: str = "hard", iou_thresh: float = 0.5) -> dict[str, Any]:
    """Counts at a fixed raw score threshold. A detection is a false detection if it matches no kept
    face (or a face already matched); a kept face with no match is a missed face."""
    tp = proposals = count_face = 0
    missed_by_size = {name: 0 for *_, name in SIZE_BINS}
    total_by_size = {name: 0 for *_, name in SIZE_BINS}
    for img, gt_boxes in gt.images.items():
        keep = gt.keep[setting][img]
        count_face += len(keep)
        pred = _sorted(preds.get(img, np.zeros((0, 5))))
        pred = pred[pred[:, 4] >= score_threshold]
        ignore = np.zeros(len(gt_boxes))
        if len(keep):
            ignore[keep - 1] = 1
        if len(gt_boxes) and len(pred):
            _, proposal_list, recall_list = image_eval(pred, gt_boxes, ignore, iou_thresh)
            proposals += int((proposal_list == 1).sum())
            tp += int((recall_list == 1).sum())
        else:
            proposals += len(pred)
            recall_list = np.zeros(len(gt_boxes))
        for k in keep:
            side = min(gt_boxes[k - 1, 2], gt_boxes[k - 1, 3])
            for lo_, hi_, name in SIZE_BINS:
                if lo_ <= side < hi_:
                    total_by_size[name] += 1
                    if recall_list[k - 1] != 1:
                        missed_by_size[name] += 1
    return {
        "setting": setting, "score_threshold": score_threshold, "faces": count_face,
        "detections_counted": proposals, "true_positives": tp,
        "false_detections": proposals - tp, "missed_faces": count_face - tp,
        "precision": tp / proposals if proposals else 0.0, "recall": tp / count_face if count_face else 0.0,
        "recall_by_face_size": {k: 1 - missed_by_size[k] / total_by_size[k] if total_by_size[k] else None
                                for k in total_by_size},
        "faces_by_size": total_by_size,
    }


def write_wider_predictions(preds: dict[str, np.ndarray], out_dir: str | Path) -> None:
    """Official submission format: <event>/<name>.txt with name, count, then 'x y w h score' rows."""
    out_dir = Path(out_dir)
    for img, p in preds.items():
        f = out_dir / f"{img}.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} {s:.5f}" for x, y, w, h, s in _sorted(p))
        f.write_text(f"{Path(img).name}\n{len(p)}\n{rows}\n")
