"""Train/test splits and the training-data policy (docs/05_VALIDATION.md §2; docs/02_MATH.md §6).

Splits work on person-minute tables (one row per player-minute at risk, the schema of
``synth.generate_person_minutes``). A fold is a pair of row masks; every row of a player-match
falls on the same side, so no match is split across training and test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from livrisk.config import Config

BoolArray = NDArray[np.bool_]

#: Sources of person-minute rows (docs/02 §6): historical archives, shadow mode, live mode.
SOURCES: tuple[str, ...] = ("historical", "shadow", "live")


@dataclass(frozen=True)
class Fold:
    """One train/test split of a person-minute table, as row masks of equal length."""

    name: str
    train: BoolArray
    test: BoolArray

    def __post_init__(self) -> None:
        if self.train.shape != self.test.shape:
            raise ValueError("train and test masks must have the same length")
        if bool(np.any(self.train & self.test)):
            raise ValueError(f"fold {self.name}: a row is in both train and test")


def _ordered_levels(col: pd.Series[Any]) -> list[object]:
    """Levels in their natural order: category order for an ordered categorical, else sorted."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        present = set(col.unique().tolist())
        return [c for c in col.cat.categories.tolist() if c in present]
    return sorted(col.dropna().unique().tolist(), key=str)


def time_splits(person_minutes: pd.DataFrame, column: str = "season") -> list[Fold]:
    """Expanding window: for every season after the first, train on all earlier seasons and test
    on that one (docs/05 §2: train on earlier seasons, test on the next)."""
    col = person_minutes[column]
    levels = _ordered_levels(col)
    if len(levels) < 2:
        raise ValueError(f"a time split needs at least two values of {column!r}")
    values = col.to_numpy()
    folds: list[Fold] = []
    for i in range(1, len(levels)):
        train = np.isin(values, np.asarray(levels[:i], dtype=object))
        test = values == levels[i]
        folds.append(Fold(f"{column} {levels[i]}", train, np.asarray(test, dtype=np.bool_)))
    return folds


def club_splits(person_minutes: pd.DataFrame, column: str = "club_id") -> list[Fold]:
    """Leave one club out: one fold per club, trained on all the others."""
    col = person_minutes[column]
    levels = _ordered_levels(col)
    if len(levels) < 2:
        raise ValueError(f"a leave-one-out split needs at least two values of {column!r}")
    values = col.to_numpy()
    return [
        Fold(f"{column} {level}", np.asarray(values != level), np.asarray(values == level))
        for level in levels
    ]


def make_splits(person_minutes: pd.DataFrame, kind: str) -> list[Fold]:
    """``kind`` is ``time`` (by season) or ``club`` (leave one club out)."""
    if kind == "time":
        return time_splits(person_minutes)
    if kind == "club":
        return club_splits(person_minutes)
    raise ValueError(f"unknown split {kind!r}; expected 'time' or 'club'")


@dataclass(frozen=True)
class TrainingRows:
    """Rows allowed into a fit and what the policy took out (player-minutes per source)."""

    keep: BoolArray
    excluded_minutes: dict[str, int]

    @property
    def n_excluded(self) -> int:
        return sum(self.excluded_minutes.values())


def training_policy(person_minutes: pd.DataFrame, cfg: Config) -> TrainingRows:
    """Training-data policy of docs/02_MATH.md §6 (D-015) as a row mask.

    Optional columns: ``source`` (historical, shadow or live; absent means historical) and
    ``first_alert_t_min`` (minute of the first alert shown to staff for that player-match, NaN
    when none was shown). Rows of ``evaluation.training_sources`` enter unchanged. Live-mode
    rows (even when listed there) enter only when ``competition.live_data_allowed``, each
    player-match censored at its
    first shown alert (minutes from the alert on are excluded). Any other source is excluded.
    The excluded player-minutes are counted per source for the report.
    """
    n = person_minutes.shape[0]
    if "source" in person_minutes.columns:
        source = person_minutes["source"].astype(str).to_numpy()
    else:
        source = np.full(n, "historical", dtype=object)
    # Live rows always go through the censoring rule, even if listed as a training source.
    allowed: Sequence[str] = [s for s in cfg.evaluation.training_sources if s != "live"]
    keep = np.isin(source, np.asarray(list(allowed), dtype=object))
    live = source == "live"
    if cfg.competition.live_data_allowed:
        if "first_alert_t_min" in person_minutes.columns:
            alert = person_minutes["first_alert_t_min"].to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            alert = np.full(n, math.nan)
        t_min = person_minutes["t_min"].to_numpy(dtype=np.float64)
        before_alert = np.isnan(alert) | (t_min < alert)
        keep = keep | (live & before_alert)
    excluded: dict[str, int] = {}
    for name in sorted(set(source[~keep].tolist())):
        excluded[str(name)] = int(np.sum(~keep & (source == name)))
    return TrainingRows(keep=np.asarray(keep, dtype=np.bool_), excluded_minutes=excluded)
