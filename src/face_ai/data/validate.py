"""Dataset validation: structure, decodability, duplicates, annotation sanity."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from face_ai.data.annotations import iter_lfw_images, parse_lfw_pairs, parse_wider_bbx_gt

log = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _count_images(d: Path) -> int:
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if d.exists() else 0


def quick_count(name: str, ds_dir: Path) -> dict[str, int]:
    """Cheap image counts by file extension, used right after download."""
    if name == "wider_face":
        return {s: _count_images(ds_dir / f"WIDER_{s}" / "images") for s in ("train", "val", "test")}
    if name == "lfw":
        root = ds_dir / "lfw"
        return {"images": _count_images(root),
                "identities": sum(1 for p in root.iterdir() if p.is_dir()) if root.exists() else 0}
    return {"images": _count_images(ds_dir)}


def inspect_image(path: Path) -> dict[str, Any]:
    """Decode an image and hash its bytes. Never raises; errors are returned in the record."""
    rec: dict[str, Any] = {"path": str(path)}
    try:
        data = path.read_bytes()
        rec["sha256"] = hashlib.sha256(data).hexdigest()
        rec["bytes"] = len(data)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            rec["error"] = "decode_failed"
        else:
            rec["height"], rec["width"] = img.shape[:2]
    except OSError as e:
        rec["error"] = f"read_failed: {e}"
    return rec


def inspect_images(paths: list[Path], workers: int = 8, desc: str = "inspect") -> list[dict[str, Any]]:
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(inspect_image, paths), total=len(paths), desc=desc, mininterval=2.0))


def _duplicates(records: list[dict[str, Any]], root: Path) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if "sha256" in r:
            by_hash[r["sha256"]].append(str(Path(r["path"]).relative_to(root)))
    return [sorted(v) for v in by_hash.values() if len(v) > 1]


def validate_wider(ds_dir: Path, workers: int = 8) -> dict[str, Any]:
    report: dict[str, Any] = {"dataset": "wider_face", "splits": {}, "problems": []}
    split_dir = ds_dir / "wider_face_split"
    if not split_dir.exists():
        report["problems"].append("wider_face_split/ missing")
    hashes_by_split: dict[str, set[str]] = {}
    for split in ("train", "val"):
        img_root = ds_dir / f"WIDER_{split}" / "images"
        gt_file = split_dir / f"wider_face_{split}_bbx_gt.txt"
        s: dict[str, Any] = {"image_root": str(img_root), "annotation_file": str(gt_file)}
        if not img_root.exists() or not gt_file.exists():
            s["status"] = "missing"
            report["splits"][split] = s
            report["problems"].append(f"{split}: images or annotations missing")
            continue
        entries = parse_wider_bbx_gt(gt_file)
        listed = {e.rel_path for e in entries}
        on_disk = {str(p.relative_to(img_root)) for p in img_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
        records = inspect_images([img_root / r for r in sorted(listed & on_disk)], workers, f"wider {split}")
        sizes = {str(Path(r["path"]).relative_to(img_root)): (r.get("width"), r.get("height")) for r in records}
        corrupt = [r["path"] for r in records if "error" in r]

        n_faces = degenerate = invalid_flag = out_of_bounds = tiny = 0
        oob_examples: list[str] = []
        for e in entries:
            w, h = sizes.get(e.rel_path, (None, None))
            for box, attr in zip(e.boxes, e.attrs):
                n_faces += 1
                x, y, bw, bh = box
                if bw <= 0 or bh <= 0:
                    degenerate += 1
                if attr[3] == 1:
                    invalid_flag += 1
                if 0 < min(bw, bh) < 4:
                    tiny += 1
                if w and (x < 0 or y < 0 or x + bw > w + 1 or y + bh > h + 1):
                    out_of_bounds += 1
                    if len(oob_examples) < 10:
                        oob_examples.append(f"{e.rel_path}: {box.tolist()} in {w}x{h}")
        hashes_by_split[split] = {r["sha256"] for r in records if "sha256" in r}
        widths = [r["width"] for r in records if "width" in r]
        heights = [r["height"] for r in records if "height" in r]
        s.update({
            "status": "ok",
            "annotated_images": len(entries),
            "images_on_disk": len(on_disk),
            "missing_images": sorted(listed - on_disk),
            "unannotated_images": sorted(on_disk - listed),
            "corrupt_images": corrupt,
            "duplicate_groups": _duplicates(records, img_root),
            "faces": n_faces,
            "images_without_faces": sum(1 for e in entries if len(e.boxes) == 0),
            "degenerate_boxes": degenerate,
            "invalid_flag_boxes": invalid_flag,
            "tiny_boxes_lt4px": tiny,
            "out_of_bounds_boxes": out_of_bounds,
            "out_of_bounds_examples": oob_examples,
            "width_range": [min(widths), max(widths)] if widths else None,
            "height_range": [min(heights), max(heights)] if heights else None,
            "events": len({e.event for e in entries}),
        })
        if s["missing_images"] or corrupt:
            s["status"] = "problems"
            report["problems"].append(f"{split}: {len(s['missing_images'])} missing, {len(corrupt)} corrupt images")
        report["splits"][split] = s

    if len(hashes_by_split) == 2:
        overlap = hashes_by_split["train"] & hashes_by_split["val"]
        report["train_val_identical_files"] = len(overlap)
        if overlap:
            report["problems"].append(f"{len(overlap)} byte-identical images appear in both train and val")

    # Evaluation ground truth (.mat) must agree with the text annotations for val.
    gt_dir = ds_dir / "eval_tools" / "ground_truth"
    try:
        from scipy.io import loadmat

        mat = loadmat(gt_dir / "wider_face_val.mat")
        mat_faces = sum(int(b[0].shape[0]) for ev in mat["face_bbx_list"] for b in ev[0])
        report["eval_ground_truth"] = {
            "dir": str(gt_dir),
            "files": sorted(p.name for p in gt_dir.glob("*.mat")),
            "val_faces_in_mat": mat_faces,
            "val_faces_in_txt": report["splits"].get("val", {}).get("faces"),
        }
        if mat_faces != report["splits"].get("val", {}).get("faces"):
            report["problems"].append("val face count differs between wider_face_val.mat and the txt annotations")
    except Exception as e:  # noqa: BLE001
        report["problems"].append(f"eval ground truth unreadable: {e}")
    extras = [p.name for p in ds_dir.iterdir() if p.name == "__MACOSX"]
    if extras:
        report["notes"] = ["__MACOSX/ is macOS archive metadata from eval_tools.zip; ignored"]
    report["status"] = "valid" if not report["problems"] else "valid_with_warnings"
    return report


def validate_lfw(ds_dir: Path, workers: int = 8) -> dict[str, Any]:
    root = ds_dir / "lfw"
    report: dict[str, Any] = {"dataset": "lfw", "problems": []}
    items = list(iter_lfw_images(root))
    records = inspect_images([p for _, p in items], workers, "lfw")
    per_id: dict[str, int] = defaultdict(int)
    for name, _ in items:
        per_id[name] += 1
    corrupt = [r["path"] for r in records if "error" in r]
    sizes = {(r.get("width"), r.get("height")) for r in records if "width" in r}
    bad_names = [str(p.relative_to(root)) for name, p in items if not p.stem.startswith(name + "_")]
    counts = np.array(list(per_id.values()))
    report.update({
        "images": len(items),
        "identities": len(per_id),
        "identities_with_ge2_images": int((counts >= 2).sum()),
        "identities_with_ge3_images": int((counts >= 3).sum()),
        "identities_with_ge5_images": int((counts >= 5).sum()),
        "max_images_per_identity": int(counts.max()) if len(counts) else 0,
        "image_sizes": sorted(f"{w}x{h}" for w, h in sizes),
        "corrupt_images": corrupt,
        "misnamed_files": bad_names,
        "duplicate_groups": _duplicates(records, root),
    })
    pairs_file = ds_dir / "archives" / "pairs.txt"
    try:
        pairs = parse_lfw_pairs(pairs_file)
        flat = [p for fold in pairs.folds for p in fold]
        missing = sorted({x for a, b, _ in flat for x in (a, b) if not (root / x).exists()})
        report["pairs"] = {
            "file": str(pairs_file), "folds": len(pairs.folds), "pairs": len(flat),
            "positive": sum(1 for *_, same in flat if same), "negative": sum(1 for *_, same in flat if not same),
            "missing_referenced_images": missing,
        }
        if missing:
            report["problems"].append(f"pairs.txt references {len(missing)} missing images")
    except Exception as e:  # noqa: BLE001
        report["problems"].append(f"pairs.txt unreadable: {e}")
    if corrupt:
        report["problems"].append(f"{len(corrupt)} corrupt images")
    if report["duplicate_groups"]:
        report["notes"] = [f"{len(report['duplicate_groups'])} groups of byte-identical images (kept; reported)"]
    report["status"] = "valid" if not report["problems"] else "valid_with_warnings"
    return report
