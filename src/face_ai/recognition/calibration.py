"""Score calibration: decision thresholds and false-match / false-reject estimates.

Built by ``scripts/evaluate_recognition.py`` from the *calibration* half of the LFW open-set split:
* impostor scores: every unknown probe vs every enrolled template (~1.1 M comparisons), which
  resolves false-match rates down to ~1e-6;
* genuine scores: every known probe vs its own template.

Stored in ``models/calibration/<variant>.json`` (scores only; no images or embeddings).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _curve(scores: np.ndarray, levels: np.ndarray, upper: bool) -> list[list[float]]:
    """(score, rate) pairs. upper=True: rate = P(s >= score) (FAR); False: rate = P(s < score) (FRR)."""
    s = np.sort(scores)
    n = len(s)
    out = []
    for r in levels:
        k = int(np.clip(round(r * n), 1, n))
        thr = s[n - k] if upper else s[k - 1]
        out.append([float(thr), float(r)])
    return out


@dataclass
class ScoreCalibration:
    embedder: str
    variant: str
    far_curve: list[list[float]]            # [score, FAR] with FAR decreasing as score increases
    frr_curve: list[list[float]]            # [score, FRR] with FRR increasing as score increases
    n_impostor: int
    n_genuine: int
    identification: dict[str, float] = field(default_factory=dict)  # accept / reject thresholds (1:N)
    verification: dict[str, float] = field(default_factory=dict)    # accept / reject thresholds (1:1)
    source: str = ""

    @classmethod
    def build(cls, embedder: str, variant: str, impostor: np.ndarray, genuine: np.ndarray, **kw: Any) -> "ScoreCalibration":
        n_i, n_g = len(impostor), len(genuine)
        far_levels = np.unique(np.concatenate([np.logspace(np.log10(1.0 / n_i), 0, 120), [1e-4, 1e-3, 1e-2, 0.05, 0.1]]))
        frr_levels = np.unique(np.concatenate([np.logspace(np.log10(1.0 / n_g), 0, 80), [1e-3, 1e-2, 0.05, 0.1]]))
        return cls(embedder, variant, _curve(impostor, far_levels, True), _curve(genuine, frr_levels, False), n_i, n_g, **kw)

    def far(self, score: float) -> float:
        """Estimated probability that a stranger scores >= ``score`` against one template.
        Values at ``1 / n_impostor`` are upper bounds (no calibration impostor scored that high)."""
        pts = np.array(sorted(self.far_curve))  # ascending score
        if score >= pts[-1, 0]:
            return 1.0 / self.n_impostor
        if score <= pts[0, 0]:
            return 1.0
        return float(10 ** np.interp(score, pts[:, 0], np.log10(pts[:, 1])))

    def frr(self, score: float) -> float:
        """Estimated probability that the genuine person scores below ``score``."""
        pts = np.array(sorted(self.frr_curve))
        if score <= pts[0, 0]:
            return 1.0 / self.n_genuine
        if score >= pts[-1, 0]:
            return 1.0
        return float(10 ** np.interp(score, pts[:, 0], np.log10(pts[:, 1])))

    def threshold_at_far(self, far: float) -> float:
        pts = np.array(sorted(self.far_curve, key=lambda p: p[1]))  # ascending FAR
        return float(np.interp(np.log10(far), np.log10(pts[:, 1]), pts[:, 0]))

    def threshold_at_frr(self, frr: float) -> float:
        pts = np.array(sorted(self.frr_curve, key=lambda p: p[1]))
        return float(np.interp(np.log10(frr), np.log10(pts[:, 1]), pts[:, 0]))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ScoreCalibration":
        return cls(**json.loads(Path(path).read_text()))


def format_rate(rate: float | None, bound: float | None = None) -> str:
    """'1 in 52,000' style text; '< 1 in N' when the rate is the calibration's resolution limit."""
    if rate is None:
        return "n/a"
    if rate >= 0.5:
        return f"{rate:.0%}"
    n = round(1 / rate) if rate > 0 else 0
    prefix = "< " if bound is not None and rate <= bound * 1.0001 else ""
    return f"{prefix}1 in {n:,}"
