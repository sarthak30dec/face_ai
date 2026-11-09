"""Per-track embedding aggregation for video.

One frame can be blurred or turned; averaging the embeddings of the same tracked face over recent
frames (weighted by quality) gives a much more stable identity than deciding frame by frame.
Track ids are temporary, so the aggregated identity only lives as long as the track.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from face_ai.recognition.embedder import l2_normalize
from face_ai.recognition.quality import FaceQuality


@dataclass
class TrackIdentityAggregator:
    window: int = 15            # frames kept per track
    min_frames: int = 1
    _buf: dict[int, deque] = field(default_factory=dict)

    def update(self, track_id: int, embedding: np.ndarray, quality: FaceQuality) -> tuple[np.ndarray, FaceQuality, int]:
        """Add this frame and return (aggregated embedding, quality to gate on, frames used).

        Only acceptable-quality frames are averaged; until one exists, the current frame is returned
        with its own (low) quality so the matcher reports ``uncertain``.
        """
        buf = self._buf.setdefault(track_id, deque(maxlen=self.window))
        if quality.acceptable:
            buf.append((np.asarray(embedding, np.float32), float(quality.score), quality))
        if len(buf) < self.min_frames or not buf:
            return np.asarray(embedding, np.float32), quality, 0
        embs = np.stack([b[0] for b in buf])
        w = np.array([b[1] for b in buf], np.float32)
        agg = l2_normalize((embs * w[:, None]).sum(0))
        best_q = max((b[2] for b in buf), key=lambda q: q.score)
        return agg, best_q, len(buf)

    def prune(self, active_track_ids: set[int]) -> None:
        for tid in list(self._buf):
            if tid not in active_track_ids:
                del self._buf[tid]
