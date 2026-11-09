"""Low-level OpenCV drawing primitives."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from face_ai.api.schemas import BoundingBox

Color = tuple[int, int, int]


def draw_box(img: np.ndarray, box: BoundingBox, color: Color, thickness: int = 2) -> None:
    cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, thickness, cv2.LINE_AA)


def draw_label(img: np.ndarray, lines: list[str], anchor: tuple[int, int], color: Color,
               font_scale: float = 0.5, thickness: int = 1, above: bool = True) -> None:
    """Stacked text lines on a filled background, anchored at the box's top-left (or below it)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    lh = max(s[1] for s in sizes) + 6
    bw = max(s[0] for s in sizes) + 6
    bh = lh * len(lines)
    x, y = anchor
    y0 = y - bh if above and y - bh >= 0 else y
    x = int(np.clip(x, 0, max(0, img.shape[1] - bw)))
    y0 = int(np.clip(y0, 0, max(0, img.shape[0] - bh)))
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y0), (x + bw, y0 + bh), color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, dst=img)
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + 3, y0 + (i + 1) * lh - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_connections(img: np.ndarray, points: np.ndarray, edges: Iterable[tuple[int, int]], color: Color,
                     thickness: int = 1) -> None:
    pts = np.round(points[:, :2]).astype(np.int32)
    n = len(pts)
    segs = [np.array([pts[a], pts[b]]) for a, b in edges if a < n and b < n]
    if segs:
        cv2.polylines(img, segs, False, color, thickness, cv2.LINE_AA)


def draw_points(img: np.ndarray, points: np.ndarray, color: Color, radius: int = 2) -> None:
    for x, y in np.round(points[:, :2]).astype(int):
        cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
