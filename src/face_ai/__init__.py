"""face_ai: local face detection, recognition and face-mesh visualization."""

from face_ai.api.schemas import (
    UNKNOWN,
    BoundingBox,
    FaceAnalysis,
    FaceDetection,
    FaceEmbedding,
    FaceLandmarks,
    IdentityMatch,
)

__version__ = "0.1.0"

__all__ = [
    "UNKNOWN", "BoundingBox", "FaceAnalysis", "FaceDetection", "FaceEmbedding", "FaceLandmarks",
    "IdentityMatch", "__version__",
]
