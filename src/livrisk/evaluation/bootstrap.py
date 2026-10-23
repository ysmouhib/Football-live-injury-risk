"""Player-cluster bootstrap (docs/02_MATH.md §9: whole players resampled, 1,000 times).

Statistics that are ratios of row sums (means, O/E, recall, flag share) are resampled exactly
from per-cluster sums and a matrix of resampling counts; with the same clusters, ``n_boot`` and
seed, every such statistic sees the same resamples, so differences of statistics are paired.
Other statistics (PR-AUC) resample rows through ``cluster_bootstrap_delta``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Interval:
    """Point estimate with a percentile bootstrap interval."""

    point: float
    lo: float
    hi: float

    def excludes(self, value: float = 0.0) -> bool:
        """True when ``value`` lies outside [lo, hi]."""
        return self.lo > value or self.hi < value


def _codes(clusters: ArrayLike) -> tuple[IntArray, int]:
    codes, uniques = pd.factorize(np.asarray(clusters, dtype=object), sort=True)
    return np.asarray(codes, dtype=np.int64), len(uniques)


def resample_counts(n_clusters: int, n_boot: int, seed: int) -> FloatArray:
    """(n_boot, n_clusters) times each cluster is drawn in each resample."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    out = np.zeros((n_boot, n_clusters), dtype=np.float64)
    for b in range(n_boot):
        out[b] = np.bincount(draws[b], minlength=n_clusters)
    return out


def ratio_draws(
    num: ArrayLike, den: ArrayLike, clusters: ArrayLike, n_boot: int = 1000, seed: int = 0
) -> tuple[float, FloatArray]:
    """Point value and bootstrap draws of sum(num) / sum(den) over whole-cluster resamples."""
    codes, n_c = _codes(clusters)
    s_num = np.bincount(codes, weights=np.asarray(num, dtype=np.float64), minlength=n_c)
    s_den = np.bincount(codes, weights=np.asarray(den, dtype=np.float64), minlength=n_c)
    counts = resample_counts(n_c, n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (counts @ s_num) / (counts @ s_den)
    return float(s_num.sum() / s_den.sum()), np.asarray(draws, dtype=np.float64)


def percentile_interval(point: float, draws: ArrayLike, level: float = 0.95) -> Interval:
    """Percentile interval of the finite draws at ``level``."""
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(d, [tail, 1.0 - tail]) if d.size else (np.nan, np.nan)
    return Interval(point=point, lo=float(lo), hi=float(hi))


def mean_interval(
    values: ArrayLike, clusters: ArrayLike, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[Interval, FloatArray]:
    """Cluster-bootstrap interval of the mean of per-row ``values``, and its draws."""
    v = np.asarray(values, dtype=np.float64)
    point, draws = ratio_draws(v, np.ones_like(v), clusters, n_boot, seed)
    return percentile_interval(point, draws, level), draws


def bootstrap_p_value(draws: ArrayLike, null: float = 0.0) -> float:
    """Two-sided bootstrap p-value of ``null`` from the draws (with the +1 correction)."""
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    below = (1 + int(np.sum(d <= null))) / (n + 1)
    above = (1 + int(np.sum(d >= null))) / (n + 1)
    return float(min(1.0, 2.0 * min(below, above)))


def cluster_bootstrap_delta(
    metric: Callable[[FloatArray, FloatArray], float],
    y: ArrayLike,
    p_a: ArrayLike,
    p_b: ArrayLike,
    clusters: ArrayLike,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) of metric(y, p_b) - metric(y, p_a), resampling whole clusters (players).

    95 % percentile interval; rows of a drawn cluster enter as often as it is drawn.
    """
    yy = np.asarray(y, dtype=np.float64)
    a = np.asarray(p_a, dtype=np.float64)
    b = np.asarray(p_b, dtype=np.float64)
    codes, n_c = _codes(clusters)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=n_c)
    starts = np.cumsum(counts) - counts
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        drawn = rng.integers(0, n_c, size=n_c)
        c = counts[drawn]
        offsets = np.repeat(np.cumsum(c) - c, c)
        rows = order[np.repeat(starts[drawn], c) + np.arange(int(c.sum())) - offsets]
        deltas[k] = metric(yy[rows], b[rows]) - metric(yy[rows], a[rows])
    interval = percentile_interval(metric(yy, b) - metric(yy, a), deltas)
    return interval.point, interval.lo, interval.hi
