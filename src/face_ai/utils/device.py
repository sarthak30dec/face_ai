"""Hardware inspection and device selection."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any

import psutil


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_info(path: str = ".") -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.mac_ver()[0] or platform.version(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "disk_free_gb": round(shutil.disk_usage(path).free / 2**30, 1),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps_available"] = bool(torch.backends.mps.is_available())
        info["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        info["torch"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except ImportError:
        pass
    return info


def torch_device(preference: str = "auto"):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def onnx_providers(preference: str = "cpu") -> list[str]:
    """ONNX Runtime providers. CPU is the default: it is deterministic and, for these small
    models, usually as fast as CoreML once session-creation cost is included."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if preference == "coreml" and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if preference == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
