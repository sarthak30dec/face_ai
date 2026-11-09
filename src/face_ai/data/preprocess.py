"""Reproducible splits, annotation conversion and face pre-detection for LFW."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from face_ai.api.schemas import FaceDetection
from face_ai.data.annotations import iter_lfw_images, parse_lfw_pairs, parse_wider_bbx_gt, wider_to_coco
from face_ai.data.validate import inspect_images

log = logging.getLogger(__name__)


def _write(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
    return path


# --- WIDER FACE --------------------------------------------------------------------------------

def preprocess_wider(raw: Path, processed: Path, splits: Path, seed: int = 42, val_fraction: float = 0.1,
                     workers: int = 8) -> dict[str, Any]:
    """COCO conversion + fine-tuning splits.

    * ``train_ft`` / ``val_ft``: official train split divided per event class (stratified), seeded.
    * ``test``: official val split (official test labels are not public). Never used for training,
      threshold choice or early stopping.
    * Train images byte-identical to any val image are excluded from train_ft/val_ft (leakage).
    """
    summary: dict[str, Any] = {"seed": seed, "val_fraction": val_fraction}
    entries = {s: parse_wider_bbx_gt(raw / "wider_face_split" / f"wider_face_{s}_bbx_gt.txt") for s in ("train", "val")}
    records = {}
    for s, ents in entries.items():
        root = raw / f"WIDER_{s}" / "images"
        recs = inspect_images([root / e.rel_path for e in ents], workers, f"wider {s} sizes")
        records[s] = {str(Path(r["path"]).relative_to(root)): r for r in recs}
        sizes = {k: (r.get("width", 0), r.get("height", 0)) for k, r in records[s].items()}
        coco = wider_to_coco(ents, root, sizes)
        _write(processed / "wider_face" / f"{s}_coco.json", coco)
        ignored = defaultdict(int)
        for a in coco["annotations"]:
            if a["ignore_reason"]:
                ignored[a["ignore_reason"]] += 1
        failed = [k for k, r in records[s].items() if "error" in r]
        summary[f"{s}_coco"] = {"images": len(coco["images"]), "annotations": len(coco["annotations"]),
                                "flagged_ignore": dict(ignored), "unreadable_images": failed}

    val_hashes = {r["sha256"] for r in records["val"].values() if "sha256" in r}
    leaked = sorted(k for k, r in records["train"].items() if r.get("sha256") in val_hashes)
    unreadable = {k for k, r in records["train"].items() if "error" in r}

    rng = random.Random(seed)
    by_event: dict[str, list[str]] = defaultdict(list)
    for e in entries["train"]:
        if e.rel_path not in leaked and e.rel_path not in unreadable:
            by_event[e.event].append(e.rel_path)
    train_ft, val_ft = [], []
    for ev in sorted(by_event):
        items = sorted(by_event[ev])
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction))
        val_ft += items[:n_val]
        train_ft += items[n_val:]
    test = sorted(e.rel_path for e in entries["val"])
    base = {"dataset": "wider_face", "seed": seed}
    _write(splits / "wider_face" / "train_ft.json", {**base, "source_split": "train", "image_root": str(raw / "WIDER_train" / "images"),
                                                      "coco": str(processed / "wider_face" / "train_coco.json"), "images": sorted(train_ft)})
    _write(splits / "wider_face" / "val_ft.json", {**base, "source_split": "train", "image_root": str(raw / "WIDER_train" / "images"),
                                                    "coco": str(processed / "wider_face" / "train_coco.json"), "images": sorted(val_ft)})
    _write(splits / "wider_face" / "test.json", {**base, "source_split": "val", "image_root": str(raw / "WIDER_val" / "images"),
                                                  "coco": str(processed / "wider_face" / "val_coco.json"), "images": test})
    summary.update({"train_ft": len(train_ft), "val_ft": len(val_ft), "test": len(test),
                    "excluded_train_leakage": leaked, "excluded_train_unreadable": sorted(unreadable)})
    _write(splits / "wider_face" / "split_info.json", summary)
    return summary


# --- LFW ---------------------------------------------------------------------------------------

def lfw_pairs_split(raw: Path, splits: Path) -> dict[str, Any]:
    pairs = parse_lfw_pairs(raw / "archives" / "pairs.txt")
    out = {"protocol": "LFW View 2, unrestricted with labeled outside data; 10-fold cross-validation",
           "folds": [[{"a": a, "b": b, "same": s} for a, b, s in f] for f in pairs.folds]}
    _write(splits / "lfw" / "pairs_view2.json", out)
    return {"folds": len(pairs.folds), "pairs": sum(len(f) for f in pairs.folds)}


def lfw_openset_split(raw: Path, splits: Path, seed: int = 42, n_enroll: int = 2,
                      max_probes_per_identity: int = 10) -> dict[str, Any]:
    """Identity-disjoint open-set identification protocol.

    Identities are split 50/50 into ``calibration`` (threshold tuning only) and ``test`` (reported).
    Within each half:
      * identities with >= n_enroll + 1 images are *known*: ``n_enroll`` random images go to the
        gallery, up to ``max_probes_per_identity`` of the rest become known probes (enrollment
        images are never probes);
      * every other identity is *unknown*: one image becomes an unknown probe.
    The gallery is an ephemeral, in-memory evaluation fixture and is never persisted.
    """
    by_id: dict[str, list[str]] = defaultdict(list)
    for name, p in iter_lfw_images(raw / "lfw"):
        by_id[name].append(f"{name}/{p.name}")
    rng = random.Random(seed)
    ids = sorted(by_id)
    rng.shuffle(ids)
    halves = {"calibration": sorted(ids[: len(ids) // 2]), "test": sorted(ids[len(ids) // 2:])}
    out: dict[str, Any] = {"seed": seed, "n_enroll": n_enroll, "max_probes_per_identity": max_probes_per_identity,
                           "note": "Evaluation-only gallery built from public benchmark identities; never saved."}
    stats = {}
    for part, part_ids in halves.items():
        gallery, known, unknown = {}, [], []
        for pid in part_ids:
            imgs = sorted(by_id[pid])
            rng.shuffle(imgs)
            if len(imgs) >= n_enroll + 1:
                gallery[pid] = sorted(imgs[:n_enroll])
                known += [{"image": x, "identity": pid} for x in sorted(imgs[n_enroll:])[:max_probes_per_identity]]
            else:
                unknown.append({"image": imgs[0], "identity": pid})
        out[part] = {"identities": part_ids, "gallery": gallery, "known_probes": known, "unknown_probes": unknown}
        stats[part] = {"identities": len(part_ids), "enrolled": len(gallery), "known_probes": len(known),
                       "unknown_probes": len(unknown)}
    assert not set(halves["calibration"]) & set(halves["test"])
    out["stats"] = stats
    _write(splits / "lfw" / "openset.json", out)
    return stats


def _pick_lfw_face(dets: list[FaceDetection], w: int, h: int) -> FaceDetection | None:
    """LFW images are centered on the labeled face: prefer large, central detections."""
    if not dets:
        return None
    cx, cy = w / 2, h / 2

    def key(d: FaceDetection) -> float:
        dx, dy = d.bbox.center[0] - cx, d.bbox.center[1] - cy
        return d.bbox.area / (1.0 + (dx * dx + dy * dy) / (0.15 * w) ** 2)

    best = max(dets, key=key)
    bx, by = best.bbox.center
    return best if abs(bx - cx) < 0.3 * w and abs(by - cy) < 0.3 * h else None


def detect_lfw_faces(raw: Path, processed: Path, detector, low_threshold: float = 0.3) -> dict[str, Any]:
    """Detect + record 5 keypoints for every LFW image (cached per detector).

    If no central face is found even at ``low_threshold``, the mean keypoints of all successful
    detections are used (a fixed "average LFW face" alignment) and the image is flagged ``fallback``.
    """
    from face_ai.utils.image_io import load_image

    items = list(iter_lfw_images(raw / "lfw"))
    out: dict[str, Any] = {}
    orig = detector.score_threshold
    for name, p in tqdm(items, desc=f"lfw detect ({detector.name})", mininterval=2.0):
        rel = f"{name}/{p.name}"
        try:
            img = load_image(p)
        except Exception as e:  # noqa: BLE001
            out[rel] = {"status": "error", "error": str(e)}
            continue
        h, w = img.shape[:2]
        detector.score_threshold = orig
        dets = detector.detect(img)
        face = _pick_lfw_face(dets, w, h)
        status = "detected"
        if face is None:
            detector.score_threshold = low_threshold
            face = _pick_lfw_face(detector.detect(img), w, h)
            status = "detected_low_threshold"
        if face is None:
            out[rel] = {"status": "pending_fallback", "n_faces": len(dets)}
            continue
        out[rel] = {"status": status, "n_faces": len(dets), "score": face.score,
                    "bbox": [face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2],
                    "keypoints": None if face.keypoints is None else face.keypoints.tolist()}
    detector.score_threshold = orig
    kps = np.array([v["keypoints"] for v in out.values() if v.get("keypoints")], dtype=np.float32)
    mean_kps = kps.mean(0).tolist() if len(kps) else None
    for v in out.values():
        if v["status"] == "pending_fallback":
            v.update({"status": "fallback_mean_keypoints", "keypoints": mean_kps, "score": 0.0})
    counts: dict[str, int] = defaultdict(int)
    for v in out.values():
        counts[v["status"]] += 1
    payload = {"detector": detector.name, "score_threshold": orig, "low_threshold": low_threshold,
               "mean_keypoints": mean_kps, "status_counts": dict(counts), "faces": out}
    _write(processed / "lfw" / f"faces_{detector.name}.json", payload)
    return {"detector": detector.name, "status_counts": dict(counts),
            "fallback_images": sorted(k for k, v in out.items() if v["status"].startswith("fallback")),
            "errors": sorted(k for k, v in out.items() if v["status"] == "error")}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
