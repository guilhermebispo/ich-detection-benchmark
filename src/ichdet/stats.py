"""Patient-level paired bootstrap, equivalence test (TOST) and precision projection.

The bootstrap resamples exams (one exam per patient in CQ500 and CT-ICH) with replacement, within each
stratum (cross-validation fold), and gives each exam a weight equal to the number of times it was drawn.
Both detectors are evaluated on the same draws, so the difference between them is paired.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm


def bootstrap_weights(strata_sizes: Sequence[int], n_resamples: int, seed: int):
    """Yield, for each resample, one weight array per stratum (counts of a draw with replacement)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        yield [rng.multinomial(n, np.full(n, 1.0 / n)).astype(float) for n in strata_sizes]


def paired_bootstrap(statistic: Callable[[list[np.ndarray]], Sequence[float]], strata_sizes: Sequence[int],
                     n_resamples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate and bootstrap replicates of `statistic(weights_per_stratum)`.

    `statistic` returns a fixed-length sequence (for example AP of RetinaNet, AP of YOLO, difference).
    Returns (point, replicates) with replicates shaped (n_resamples, len(point)).
    """
    point = np.asarray(statistic([np.ones(n) for n in strata_sizes]), dtype=float)
    reps = np.array([statistic(w) for w in bootstrap_weights(strata_sizes, n_resamples, seed)], dtype=float)
    return point, reps.reshape(n_resamples, -1)


def percentile_ci(replicates: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1 - level) / 2
    lo, hi = np.nanpercentile(replicates, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def tost(point_diff: float, diff_replicates: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests by the confidence-interval rule.

    Equivalence is concluded when the (1 - 2*alpha) percentile interval of the difference lies entirely
    inside (-margin, +margin); with alpha = 0.05 this is the 90% interval.
    """
    lo, hi = percentile_ci(diff_replicates, 1 - 2 * alpha)
    lo95, hi95 = percentile_ci(diff_replicates, 0.95)
    return {
        "difference": float(point_diff),
        "ci90": [lo, hi],
        "ci95": [lo95, hi95],
        "margin": margin,
        "equivalent": bool(-margin < lo and hi < margin),
        "bootstrap_se": float(np.nanstd(diff_replicates, ddof=1)),
        "n_resamples": int(np.sum(~np.isnan(diff_replicates))),
    }


def projected_half_width(pilot_se: float, pilot_exams: int, planned_exams: int, alpha: float = 0.05) -> float:
    """Expected half-width of the (1 - 2*alpha) interval with `planned_exams`, scaling the pilot SE by 1/sqrt(n)."""
    return float(norm.ppf(1 - alpha) * pilot_se * np.sqrt(pilot_exams / planned_exams))


def required_exams(pilot_se: float, pilot_exams: int, margin: float, expected_diff: float = 0.0,
                   alpha: float = 0.05) -> int:
    """Exams needed so that |expected difference| + half-width of the 90% interval stays below the margin."""
    room = margin - abs(expected_diff)
    if room <= 0:
        raise ValueError("expected difference is not smaller than the margin")
    return int(np.ceil(pilot_exams * (norm.ppf(1 - alpha) * pilot_se / room) ** 2))
