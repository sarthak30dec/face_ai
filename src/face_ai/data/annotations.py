"""Annotation parsers and format conversion (source files are never modified)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

WIDER_ATTRS = ("blur", "expression", "illumination", "invalid", "occlusion", "pose")


@dataclass
class WiderImage:
    """One image entry from ``wider_face_<split>_bbx_gt.txt``.

    boxes: (N, 4) x, y, w, h in pixels (as published).
    attrs: (N, 6) integer attributes in ``WIDER_ATTRS`` order.
    """

    rel_path: str  # e.g. "0--Parade/0_Parade_marchingband_1_465.jpg"
    boxes: np.ndarray
    attrs: np.ndarray

    @property
    def event(self) -> str:
        return self.rel_path.split("/", 1)[0]


def parse_wider_bbx_gt(path: str | Path) -> list[WiderImage]:
    """Parse the WIDER FACE text ground truth.

    Format per image: relative path, face count N, then max(N, 1) lines of
    ``x y w h blur expression illumination invalid occlusion pose`` (images with N = 0 have one
    all-zero placeholder line, which is skipped).
    """
    lines = Path(path).read_text().splitlines()
    out: list[WiderImage] = []
    i = 0
    while i < len(lines):
        rel = lines[i].strip()
        if not rel:
            i += 1
            continue
        n = int(lines[i + 1].strip())
        rows = [list(map(int, lines[i + 2 + k].split()[:10])) for k in range(max(n, 1))]
        i += 2 + max(n, 1)
        arr = np.array(rows[:n], dtype=np.int64).reshape(-1, 10)
        out.append(WiderImage(rel, arr[:, :4].astype(np.float32), arr[:, 4:10]))
    return out


def iter_lfw_images(root: str | Path) -> Iterator[tuple[str, Path]]:
    """Yield (identity, path) for every image under ``<root>/<Identity>/<Identity>_NNNN.jpg``."""
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            for p in sorted(d.glob("*.jpg")):
                yield d.name, p


def lfw_image_name(name: str, idx: int) -> str:
    return f"{name}/{name}_{idx:04d}.jpg"


@dataclass
class LfwPairs:
    """LFW View 2 pairs: ``folds[k]`` is a list of (rel_path_a, rel_path_b, same_identity)."""

    folds: list[list[tuple[str, str, bool]]] = field(default_factory=list)


def parse_lfw_pairs(path: str | Path) -> LfwPairs:
    """Parse ``pairs.txt``: header ``<folds> <n>``, then per fold n matched lines
    ``name i j`` followed by n mismatched lines ``name1 i name2 j``."""
    lines = Path(path).read_text().splitlines()
    n_folds, n = map(int, lines[0].split())
    rows = [l.split() for l in lines[1:] if l.strip()]
    if len(rows) != n_folds * 2 * n:
        raise ValueError(f"pairs.txt: expected {n_folds * 2 * n} pairs, found {len(rows)}")
    pairs = LfwPairs()
    for f in range(n_folds):
        fold = []
        for r in rows[f * 2 * n:(f + 1) * 2 * n]:
            if len(r) == 3:
                fold.append((lfw_image_name(r[0], int(r[1])), lfw_image_name(r[0], int(r[2])), True))
            elif len(r) == 4:
                fold.append((lfw_image_name(r[0], int(r[1])), lfw_image_name(r[2], int(r[3])), False))
            else:
                raise ValueError(f"pairs.txt: malformed line {r}")
        pairs.folds.append(fold)
    return pairs


def wider_to_coco(images: list[WiderImage], image_root: Path, sizes: dict[str, tuple[int, int]],
                  min_face_px: float = 0.0) -> dict[str, Any]:
    """Convert to COCO detection JSON, keeping every source box and its attributes.

    Boxes that are degenerate (w or h <= 0), flagged ``invalid`` by WIDER, or smaller than
    ``min_face_px`` get ``iscrowd=1`` + ``ignore_reason`` so trainers can skip them explicitly;
    nothing is dropped.
    """
    coco: dict[str, Any] = {
        "info": {"description": "WIDER FACE converted by face_ai", "source_format": "wider_face_bbx_gt.txt"},
        "categories": [{"id": 1, "name": "face"}],
        "images": [],
        "annotations": [],
    }
    ann_id = 1
    for img_id, im in enumerate(images, start=1):
        w, h = sizes.get(im.rel_path, (0, 0))
        coco["images"].append({"id": img_id, "file_name": im.rel_path, "width": w, "height": h,
                               "event": im.event, "image_root": str(image_root)})
        for box, attr in zip(im.boxes, im.attrs):
            x, y, bw, bh = (float(v) for v in box)
            reason = None
            if bw <= 0 or bh <= 0:
                reason = "degenerate"
            elif attr[3] == 1:
                reason = "wider_invalid_flag"
            elif min(bw, bh) < min_face_px:
                reason = f"smaller_than_{min_face_px:g}px"
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1, "bbox": [x, y, bw, bh],
                "area": max(bw, 0) * max(bh, 0), "iscrowd": int(reason is not None), "ignore_reason": reason,
                "attributes": dict(zip(WIDER_ATTRS, map(int, attr))),
            })
            ann_id += 1
    return coco
