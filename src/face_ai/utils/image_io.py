"""Image loading/saving. Images are BGR uint8 numpy arrays throughout (OpenCV convention)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


class ImageLoadError(ValueError):
    pass


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8, applying EXIF orientation. Raises ImageLoadError on failure."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"not a file: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ImageLoadError(f"could not decode image: {path}")
    return img


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {image.dtype}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"unsupported image shape {image.shape}")


def save_image(path: str | Path, image: np.ndarray, quality: int = 95) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise IOError(f"could not encode image for {path}")
    buf.tofile(str(path))
    return path


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def list_images(directory: str | Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and is_image(p))
