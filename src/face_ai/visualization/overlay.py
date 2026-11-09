"""Compose boxes, wireframe mesh and labels into one annotated image."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from face_ai.api.schemas import UNCERTAIN, FaceDetection, FaceLandmarks, IdentityMatch
from face_ai.landmarks import topology
from face_ai.visualization.drawing import draw_box, draw_connections, draw_label, draw_points

KNOWN_COLOR = (80, 200, 60)
UNKNOWN_COLOR = (40, 140, 255)
UNCERTAIN_COLOR = (160, 160, 160)
NEUTRAL_COLOR = (255, 180, 0)


@dataclass
class OverlayOptions:
    show_boxes: bool = True
    show_mesh: bool = True
    show_keypoints: bool = False
    show_detection_score: bool = True
    show_identity: bool = True
    show_track_id: bool = True
    show_index: bool = False  # prefix labels with "#1", "#2", ... (matches result tables in the UI)
    mesh_groups: Sequence[str] = topology.DEFAULT_GROUPS
    mesh_colors: dict[str, tuple[int, int, int]] = field(default_factory=lambda: dict(topology.DEFAULT_COLORS))
    box_thickness: int = 0  # 0 = scale with image size
    mesh_thickness: int = 1
    font_scale: float = 0.0  # 0 = scale with image size


def render_overlay(
    image: np.ndarray,
    detections: Sequence[FaceDetection],
    landmarks: Sequence[FaceLandmarks | None] | None = None,
    identities: Sequence[IdentityMatch | None] | None = None,
    options: OverlayOptions | None = None,
) -> np.ndarray:
    """Return a copy of ``image`` with the overlay. Lists are aligned by index with ``detections``.

    Labels keep the two scores apart: ``det`` is detector confidence that a face is present,
    ``sim`` is cosine similarity to the best enrolled identity. Box colours: green = match,
    orange = Unknown, grey = Uncertain (low quality / ambiguous), blue = no identification run.
    """
    opt = options or OverlayOptions()
    out = image.copy()
    n = len(detections)
    landmarks = list(landmarks) if landmarks is not None else [None] * n
    identities = list(identities) if identities is not None else [None] * n
    diag = float(np.hypot(*out.shape[:2]))
    bt = opt.box_thickness or max(1, int(round(diag / 600)))
    fs = opt.font_scale or float(np.clip(diag / 1800, 0.4, 1.2))

    if opt.show_mesh:
        for lm in landmarks:
            if lm is None:
                continue
            for g in opt.mesh_groups:
                draw_connections(out, lm.points, topology.connections(g),
                                 opt.mesh_colors.get(g, NEUTRAL_COLOR), opt.mesh_thickness)

    for i, (det, ident) in enumerate(zip(detections, identities)):
        if ident is None:
            color = NEUTRAL_COLOR
        elif ident.is_known:
            color = KNOWN_COLOR
        else:
            color = UNCERTAIN_COLOR if ident.decision == UNCERTAIN else UNKNOWN_COLOR
        if opt.show_boxes:
            draw_box(out, det.bbox, color, bt)
        if opt.show_keypoints and det.keypoints is not None:
            draw_points(out, det.keypoints, (0, 0, 255), max(1, bt))
        lines = []
        if opt.show_identity and ident is not None:
            lines.append(f"#{i + 1} {ident.label}" if opt.show_index else ident.label)
            if ident.similarity is not None:
                lines.append(f"sim {ident.similarity:.2f} (thr {ident.threshold:.2f})")
        elif opt.show_index:
            lines.append(f"#{i + 1}")
        if opt.show_detection_score:
            lines.append(f"det {det.score:.2f}")
        if opt.show_track_id and det.track_id is not None:
            lines.append(f"track #{det.track_id}")
        if lines:
            draw_label(out, lines, (int(det.bbox.x1), int(det.bbox.y1) - 2), color, fs, max(1, bt // 2))
    return out
