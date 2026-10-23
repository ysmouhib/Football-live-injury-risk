"""Decision curves at spell level (docs/05_VALIDATION.md §2: p* from 0.001 to 0.02 per 15 min).

Net benefit of acting on spells whose probability reaches the threshold p*, in true positives
per spell: NB(p*) = TP / N - FP / N * p* / (1 - p*). The threshold is the action threshold of
docs/02_MATH.md §7 (``riskmath.action_threshold``: p* = C / (C + B)), so each p* stands for one
cost-benefit ratio of a cheap action such as a physio's look.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Default thresholds p*, per 15-minute spell.
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(np.round(np.linspace(0.001, 0.02, 20), 6))


def net_benefit(y: ArrayLike, p: ArrayLike, thresholds: Sequence[float]) -> FloatArray:
    """Net benefit per threshold of acting when p >= p* (true positives per spell)."""
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(p, dtype=np.float64)
    n = yy.size
    out = np.empty(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        if not 0.0 < t < 1.0:
            raise ValueError("thresholds must be in (0, 1)")
        act = pp >= t
        tp = float(np.sum(act & (yy > 0.5)))
        fp = float(np.sum(act & (yy < 0.5)))
        out[i] = tp / n - fp / n * t / (1.0 - t)
    return out


def decision_curve(
    spells: pd.DataFrame,
    pred_cols: Sequence[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Net benefit of each prediction column and of acting on every spell (``all``); acting on
    none scores 0 everywhere. Columns: threshold, then one per model."""
    y = spells["y"].to_numpy(dtype=np.float64)
    table: dict[str, object] = {"threshold": list(thresholds)}
    for col in pred_cols:
        table[col] = net_benefit(y, spells[col].to_numpy(dtype=np.float64), thresholds)
    table["all"] = net_benefit(y, np.ones_like(y), thresholds)
    return pd.DataFrame(table)
