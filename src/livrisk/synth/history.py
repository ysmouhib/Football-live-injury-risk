"""Norm history of a synthetic match (M2 full path), for demos and the CLI.

``feature_history`` evaluates the offline feature reference at the ends of three 5-minute
windows inside each regular band of one past match, the shape ``build_norms`` consumes. It
exists so a synthetic match can stand in for a player's history when no recorded data is
available (``livrisk replay --synthetic``); the values come from the same batch computation the
stream engine reproduces, so norms built from them are consistent with live features.
"""

from __future__ import annotations

import pandas as pd

from livrisk.config import Config
from livrisk.contracts import PERIOD_START_S
from livrisk.features.stream import compute_features_offline
from livrisk.synth.generator import SynthMatch, match_tables

#: (band, period, band start minute) of the six regular bands.
_REGULAR_BANDS: tuple[tuple[int, int, float], ...] = (
    (0, 1, 0.0),
    (1, 1, 15.0),
    (2, 1, 30.0),
    (4, 2, 45.0),
    (5, 2, 60.0),
    (6, 2, 75.0),
)
#: Window ends within each band, minutes from the band start.
_WINDOW_ENDS_MIN: tuple[float, ...] = (5.0, 10.0, 15.0)


def feature_history(match: SynthMatch, cfg: Config) -> pd.DataFrame:
    """Raw per-band feature values of one past match, one row per player x band x window x feature.

    Columns: player_id, match_date, band, feature, value (raw, untransformed). A row is kept
    only when the window's coverage meets ``quality.min_coverage``. Every evaluation is stamped
    ``as_of`` after the window end plus the blanking gap, so no sample later than a window end
    contributes to it. Windows stop at 90 minutes; stoppage bands carry no history.
    """
    events = match_tables(match)["events"]
    p1_end = float(
        events.loc[(events["type"] == "period_end") & (events["period"] == 1), "t_s"].iloc[0]
    )
    starts = {1: 0.0, 2: p1_end}
    day = match.lineups[0].kickoff_utc.date()
    rows: list[tuple[str, object, int, str, float]] = []
    for pid, frame in match.samples.groupby("player_id"):
        for band, period, band_start_min in _REGULAR_BANDS:
            for window_min in _WINDOW_ENDS_MIN:
                end_u = starts[period] + (band_start_min + window_min) * 60.0
                end_u -= PERIOD_START_S[period]
                snap = compute_features_offline(
                    frame,
                    events,
                    str(pid),
                    end_u + cfg.engine.blanking_delta_s,
                    match.vmax_mps.get(str(pid)),
                    cfg,
                )
                for feature, value in snap.features.items():
                    if value is not None and snap.coverage[feature] >= cfg.quality.min_coverage:
                        rows.append((str(pid), day, band, feature, value))
    return pd.DataFrame(rows, columns=["player_id", "match_date", "band", "feature", "value"])
