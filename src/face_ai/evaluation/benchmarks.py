"""Latency, throughput, memory and model-size measurement."""

from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import psutil


class PeakRSS:
    """Samples this process's resident memory in a background thread."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.peak = 0
        self.start = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self) -> "PeakRSS":
        gc.collect()
        self.start = self.peak = self.proc.memory_info().rss
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._t:
            self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / 2**20

    @property
    def delta_mb(self) -> float:
        return (self.peak - self.start) / 2**20


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    out: dict[str, float] = {}
    t0 = time.perf_counter()
    yield out
    out["seconds"] = time.perf_counter() - t0


def latency_stats(times_s: Sequence[float]) -> dict[str, float]:
    a = np.asarray(times_s) * 1000
    return {"n": int(a.size), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "max_ms": float(a.max()),
            "throughput_per_s": float(1000.0 / a.mean())}


def benchmark(fn: Callable[[Any], Any], inputs: Sequence[Any], warmup: int = 10, repeats: int = 1) -> dict[str, Any]:
    """Call ``fn`` on each input (after ``warmup`` untimed calls) and summarize per-call latency."""
    for x in list(inputs)[:warmup]:
        fn(x)
    times = []
    with PeakRSS() as mem:
        for _ in range(repeats):
            for x in inputs:
                t0 = time.perf_counter()
                fn(x)
                times.append(time.perf_counter() - t0)
    return {**latency_stats(times), "warmup_calls": min(warmup, len(inputs)), "peak_rss_mb": mem.peak_mb,
            "rss_increase_mb": mem.delta_mb}


def model_size_mb(paths: Sequence[str | Path]) -> float:
    return sum(Path(p).stat().st_size for p in paths if Path(p).exists()) / 2**20
