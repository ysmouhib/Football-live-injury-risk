"""Lineup-constrained identity assignment for video tracks.

Input: per-frame tracks from an external pipeline (columns: frame_t_s, period, track_id,
team_guess, jersey_read, jersey_conf, x_m, y_m, detected). Output: TrackingSample columns with
``player_id`` and ``identity_confidence``. Method: per-window Hungarian assignment with
``scipy.optimize.linear_sum_assignment`` against the 11 on-pitch players per team; substitutions
from events; swap detection. Do not train vision models here.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from livrisk.contracts import Lineup, MatchEvent


def assign_identities(
    tracks: pd.DataFrame,
    lineups: Sequence[Lineup],
    events: Sequence[MatchEvent],
    window_s: float = 1.0,
) -> pd.DataFrame:
    raise NotImplementedError("planned for M13")
