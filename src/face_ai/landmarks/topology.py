"""MediaPipe face-mesh topology (official connection lists shipped with mediapipe.tasks)."""

from __future__ import annotations

from functools import lru_cache

# BGR colors per region for the wireframe overlay.
DEFAULT_COLORS: dict[str, tuple[int, int, int]] = {
    "tesselation": (190, 190, 190),
    "face_oval": (255, 255, 255),
    "left_eye": (48, 255, 48),
    "right_eye": (48, 48, 255),
    "left_eyebrow": (48, 255, 48),
    "right_eyebrow": (48, 48, 255),
    "left_iris": (0, 255, 255),
    "right_iris": (0, 255, 255),
    "nose": (255, 200, 0),
    "lips": (180, 105, 255),
}

# Groups drawn by default: contour, eyes, eyebrows, nose, lips (tessellation is optional/dense).
DEFAULT_GROUPS = ("face_oval", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose", "lips")

_MP_NAMES = {
    "tesselation": "FACE_LANDMARKS_TESSELATION",
    "contours": "FACE_LANDMARKS_CONTOURS",
    "face_oval": "FACE_LANDMARKS_FACE_OVAL",
    "left_eye": "FACE_LANDMARKS_LEFT_EYE",
    "right_eye": "FACE_LANDMARKS_RIGHT_EYE",
    "left_eyebrow": "FACE_LANDMARKS_LEFT_EYEBROW",
    "right_eyebrow": "FACE_LANDMARKS_RIGHT_EYEBROW",
    "left_iris": "FACE_LANDMARKS_LEFT_IRIS",
    "right_iris": "FACE_LANDMARKS_RIGHT_IRIS",
    "nose": "FACE_LANDMARKS_NOSE",
    "lips": "FACE_LANDMARKS_LIPS",
}


@lru_cache(maxsize=None)
def connections(group: str) -> tuple[tuple[int, int], ...]:
    """Edge list (start, end) for a named group, taken from ``FaceLandmarksConnections``."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections

    if group not in _MP_NAMES:
        raise KeyError(f"unknown mesh group {group!r}; options: {sorted(_MP_NAMES)}")
    conns = getattr(FaceLandmarksConnections, _MP_NAMES[group])
    return tuple((int(c.start), int(c.end)) for c in conns)


@lru_cache(maxsize=1)
def triangles() -> tuple[tuple[int, int, int], ...]:
    """Mesh faces (index triples) recovered from the official tessellation edges.

    Every 3-cycle in the edge graph is a triangle of the canonical face mesh; used to draw the
    faceted ("shard") overlay. Order is stable.
    """
    edges = connections("tesselation")
    nbrs: dict[int, set[int]] = {}
    for a, b in edges:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    tris = set()
    for a, b in edges:
        for c in nbrs[a] & nbrs[b]:
            tris.add(tuple(sorted((a, b, c))))
    return tuple(sorted(tris))


def available_groups() -> list[str]:
    return sorted(_MP_NAMES)
