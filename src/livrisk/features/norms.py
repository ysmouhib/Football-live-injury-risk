"""Player, position and team norms (docs/02_MATH.md §3).

Norms are built only from matches strictly before ``as_of_date``. Past matches are weighted by
``riskmath.recency_weights``; shrinkage (``riskmath.shrink_mean``, ``riskmath.shrink_sd``) uses
the Kish effective number of matches, never the number of windows. The spread uses
non-overlapping windows (``norms.spread_windows_per_band`` per band and match).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from livrisk.config import Config
from livrisk.contracts import PositionGroup


@dataclass(frozen=True)
class NormEntry:
    """Shrunk norm for one player x band x feature, on the transformed scale."""

    mean: float
    sd: float
    n_matches: int
    #: Kish effective number of matches after recency weighting (and any reset cap).
    n_effective: float
    #: True while a norm reset is active (docs/02_MATH.md §3); quality reason ``norm_reset``.
    reset: bool = False


class NormTable:
    """Lookup of shrunk norms; falls back to the position norm when the player has no history."""

    def get(self, player_id: str, position: PositionGroup, band: int, feature: str) -> NormEntry:
        raise NotImplementedError("planned for M4")


def build_norms(
    history: pd.DataFrame,
    positions: dict[str, PositionGroup],
    as_of_date: date,
    cfg: Config,
) -> NormTable:
    """``history`` columns: player_id, match_date, band, feature, value (raw).

    Rows with ``match_date >= as_of_date`` are ignored (no look-ahead)."""
    raise NotImplementedError("planned for M4")


def reset_active(
    days_since_return: float | None,
    days_since_last_match: float | None,
    matches_since_return: int,
    cfg: Config,
) -> bool:
    """Norm reset rule: after >= ``reset_after_absence_days`` lost to injury, or more than
    ``reset_after_no_match_days`` without a match, for the next ``reset_matches`` matches."""
    raise NotImplementedError("planned for M4")


def occupancy_drift(recent: Sequence[float], norm: Sequence[float]) -> float:
    """Total-variation distance between recent and usual speed-zone occupancy (diagnostic only;
    use ``markov.tv_distance``)."""
    raise NotImplementedError("planned for M4")


def z_for(
    table: NormTable,
    player_id: str,
    position: PositionGroup,
    band: int,
    feature: str,
    raw_value: float | None,
    cfg: Config,
) -> float | None:
    """Transform per config (log1p / log / none), then z against the shrunk norm."""
    raise NotImplementedError("planned for M4")


def team_relative(z_by_player: dict[str, float | None]) -> dict[str, float | None]:
    """z_i minus the mean z of team-mates on the pitch (excluding i and missing values)."""
    raise NotImplementedError("planned for M4")
