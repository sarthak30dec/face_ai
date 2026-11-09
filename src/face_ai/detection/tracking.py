"""Greedy IoU tracker giving faces *temporary* per-video track ids.

Track ids exist only for the lifetime of one video stream and carry no identity; identity comes
solely from gallery matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from face_ai.api.schemas import BoundingBox, FaceDetection


@dataclass
class _Track:
    track_id: int
    bbox: BoundingBox
    misses: int = 0
    hits: int = 1


@dataclass
class IoUTracker:
    iou_threshold: float = 0.3
    max_misses: int = 10
    _tracks: list[_Track] = field(default_factory=list)
    _next_id: int = 1

    def update(self, detections: list[FaceDetection]) -> list[FaceDetection]:
        """Assign ``track_id`` to each detection in place (and return them)."""
        pairs = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(detections):
                iou = t.bbox.iou(d.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self._tracks[ti]
            t.bbox, t.misses, t.hits = detections[di].bbox, 0, t.hits + 1
            detections[di].track_id = t.track_id
        for ti, t in enumerate(self._tracks):
            if ti not in used_t:
                t.misses += 1
        for di, d in enumerate(detections):
            if di not in used_d:
                self._tracks.append(_Track(self._next_id, d.bbox))
                d.track_id = self._next_id
                self._next_id += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        return detections

    def active_ids(self) -> set[int]:
        """Track ids still alive (seen within the last ``max_misses`` frames)."""
        return {t.track_id for t in self._tracks}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


def smooth_scores(history: dict[int, list[float]], track_id: int, value: float, window: int = 5) -> float:
    """Running mean of the last ``window`` identity similarities for one track (display only)."""
    h = history.setdefault(track_id, [])
    h.append(value)
    del h[:-window]
    return float(np.mean(h))
