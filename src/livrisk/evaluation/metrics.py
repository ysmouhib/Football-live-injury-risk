"""Evaluation metrics (docs/05_VALIDATION.md; docs/02_MATH.md §9)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def spell_table(person_minutes: pd.DataFrame, pred_col: str, spell_min: int = 15) -> pd.DataFrame:
    """Aggregate minute-level hazards to 15-min spells: y_spell, p_spell = 1 - exp(-sum lambda)."""
    raise NotImplementedError("planned for M10")


def log_loss(y: ArrayLike, p: ArrayLike) -> float:
    raise NotImplementedError("planned for M10")


def brier(y: ArrayLike, p: ArrayLike) -> float:
    raise NotImplementedError("planned for M10")


def pr_auc(y: ArrayLike, p: ArrayLike) -> float:
    raise NotImplementedError("planned for M10")


def calibration_slope_intercept(y: ArrayLike, p: ArrayLike) -> tuple[float, float]:
    """Fit cloglog(P(y=1)) = c0 + c1 * cloglog(p); return (c1 slope, c0 intercept)."""
    raise NotImplementedError("planned for M10")


def cluster_bootstrap_delta(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: ArrayLike,
    p_a: ArrayLike,
    p_b: ArrayLike,
    clusters: ArrayLike,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) of metric(y, p_b) - metric(y, p_a), resampling whole clusters (players)."""
    raise NotImplementedError("planned for M10")
