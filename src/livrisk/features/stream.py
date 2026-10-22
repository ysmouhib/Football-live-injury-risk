"""Stream feature engine (docs/02_MATH.md §10).

Rule: every function takes ``as_of_t_s`` and only reads samples with ``t_s <= as_of_t_s - delta``.
The online engine must reproduce ``compute_features_offline`` exactly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from livrisk.config import Config
from livrisk.contracts import FeatureSnapshot, MatchEvent, TrackingSample

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Effort:
    """A contiguous run above a speed threshold (docs/02_MATH.md §10)."""

    start_s: float
    end_s: float
    peak_mps: float
    distance_m: float


def smooth_speed(
    t_s: FloatArray, x_m: FloatArray, y_m: FloatArray, window_s: float, polyorder: int
) -> FloatArray:
    """Savitzky-Golay smoothing of positions, then speed = |dp/dt| in m/s.

    Speeds above the configured plausibility limit are returned as NaN by the caller's policy.
    Handles irregular sampling by resampling to the median sampling interval first.
    """
    raise NotImplementedError("planned for M3")


def detect_efforts(
    t_s: FloatArray,
    speed_mps: FloatArray,
    threshold_mps: float,
    min_duration_s: float,
    merge_gap_s: float,
) -> list[Effort]:
    """Efforts with speed >= threshold lasting >= min_duration_s; gaps < merge_gap_s merged."""
    raise NotImplementedError("planned for M3")


def compute_features_offline(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    player_id: str,
    as_of_t_s: float,
    vmax_mps: float | None,
    cfg: Config,
) -> FeatureSnapshot:
    """Batch reference: raw features F1-F6 (+ secondary), coverage and identity confidence for one
    player at ``as_of_t_s``. ``z`` is left empty (filled by norms in M4)."""
    raise NotImplementedError("planned for M3")


class StreamFeatureEngine:
    """Online engine with per-player ring buffers; ``tick`` every ``features.feature_tick_s``."""

    def __init__(self, cfg: Config, match_id: str, vmax_by_player: dict[str, float]) -> None:
        raise NotImplementedError("planned for M3")

    def push_samples(self, samples: Iterable[TrackingSample]) -> None:
        raise NotImplementedError("planned for M3")

    def push_event(self, event: MatchEvent) -> None:
        raise NotImplementedError("planned for M3")

    def tick(self, as_of_t_s: float) -> list[FeatureSnapshot]:
        """One snapshot per player currently on the pitch."""
        raise NotImplementedError("planned for M3")
