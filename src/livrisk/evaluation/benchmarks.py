"""Benchmarks and exploratory comparators for the Gate C report (docs/05_VALIDATION.md §2).

Every comparator has the signature ``(train, test, feature_names, cfg) -> per-minute hazards of
the test rows``. It fits on the training rows only, keeps the M1 offset and band levels, and
replaces the live layer with its own covariates.

Benchmarks (from public methods, descriptive only):
- ``ewma_acute_chronic``: acute minus chronic exponentially weighted load (tau 5 and 15 min) of
  F1 and F2 within the player-match.
- ``knn_players_like_him``: M1 times the smoothed observed/expected of the 200 most similar
  training spells (spell-mean live features).
- ``threshold_rule``: one fitted multiplier when F1 or F2 exceeds a staff-style limit of 2 SD.

Exploratory comparators (research track, docs/02_MATH.md §14; Holm-adjusted, never adopted on
the data that suggested them): ``ewma_features`` (R4) and ``team_relative`` (R5) run on
person-minute tables; R1-R3 need tracking or event-time inputs and are reported as not run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from livrisk.config import Config, config_hash
from livrisk.evaluation.metrics import spell_index, spell_table
from livrisk.evaluation.models import designs, fit_models, predict_models, with_covariates
from livrisk.features.norms import team_relative
from livrisk.model.live_hazard import Priors, fit_map, predict_hazard

FloatArray = NDArray[np.float64]
Comparator = Callable[[pd.DataFrame, pd.DataFrame, Sequence[str], Config], FloatArray]

#: Staff-style limit of the threshold rule, SD units of the player-relative features.
LIMIT_SD: float = 2.0
#: Acute and chronic time constants of the EWMA benchmark, minutes.
EWMA_TAU_MIN: tuple[float, float] = (5.0, 15.0)
#: Neighbours of the kNN benchmark (spells).
KNN_NEIGHBOURS: int = 200
#: F1-F4, the features of the team-relative variant (docs/02_MATH.md §3, §14 R5).
TEAM_RELATIVE_FEATURES: tuple[str, ...] = ("hsr_5m", "spr_5m", "recovery_10m", "hir_drop_15m")

_GROUP = ["match_id", "player_id"]


def _load_pair(feature_names: Sequence[str]) -> list[str]:
    """F1 and F2 when fitted, else the first two features."""
    pair = [f for f in ("hsr_5m", "spr_5m") if f in feature_names]
    return pair if len(pair) == 2 else list(feature_names[:2])


def ewma(person_minutes: pd.DataFrame, columns: Sequence[str], tau_min: float) -> FloatArray:
    """Exponentially weighted mean of each column within the player-match, causal in time.

    s_t = s_(t-1) + (1 - exp(-1 / tau)) (x_t - s_(t-1)), starting at the first minute's value;
    minute t uses rows up to t only. Returns an array of shape (rows, columns), in row order.
    """
    pm = person_minutes
    match = pm["match_id"].astype(str).to_numpy()
    player = pm["player_id"].astype(str).to_numpy()
    order = np.lexsort(
        (
            pm["t_min"].to_numpy(dtype=np.float64),
            pm["period"].to_numpy(dtype=np.int64),
            player,
            match,
        )
    )
    alpha = 1.0 - float(np.exp(-1.0 / tau_min))
    values = pm[list(columns)].to_numpy(dtype=np.float64)[order]
    # One row per player-match, one column per minute: the recursion runs over minutes for
    # all player-matches at once.
    new_group = np.ones(order.size, dtype=np.bool_)
    if order.size:
        new_group[1:] = (match[order][1:] != match[order][:-1]) | (
            player[order][1:] != player[order][:-1]
        )
    group = np.cumsum(new_group) - 1
    starts = np.flatnonzero(new_group)
    pos = np.arange(order.size) - starts[group]
    n_groups, length = int(starts.size), int(np.max(pos)) + 1 if order.size else 0
    grid = np.zeros((n_groups, length, values.shape[1]), dtype=np.float64)
    grid[group, pos] = values
    smooth = np.empty_like(grid)
    if length:
        smooth[:, 0] = grid[:, 0]
    for t in range(1, length):
        smooth[:, t] = smooth[:, t - 1] + alpha * (grid[:, t] - smooth[:, t - 1])
    out = np.empty_like(values)
    out[order] = smooth[group, pos]
    return out


def _fit_predict_covariates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: Sequence[str],
    cfg: Config,
    names: Sequence[str],
    x_train: FloatArray,
    x_test: FloatArray,
    version: str,
) -> FloatArray:
    """Fit the cloglog model with the M1 offset and bands plus ``names`` covariates; score test."""
    _, d_train = designs(train, feature_names, cfg)
    _, d_test = designs(test, feature_names, cfg)
    fit = fit_map(
        with_covariates(d_train, names, x_train),
        Priors.from_config(cfg),
        config_hash=config_hash(cfg),
        version=version,
    )
    return predict_hazard(fit, with_covariates(d_test, names, x_test))


def benchmark_ewma(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> FloatArray:
    """Acute minus chronic EWMA load of F1 and F2 (``EWMA_TAU_MIN``)."""
    pair = _load_pair(feature_names)
    acute, chronic = EWMA_TAU_MIN

    def cov(rows: pd.DataFrame) -> FloatArray:
        return ewma(rows, pair, acute) - ewma(rows, pair, chronic)

    names = [f"acwr_{f}" for f in pair]
    return _fit_predict_covariates(
        train, test, feature_names, cfg, names, cov(train), cov(test), "bench-ewma"
    )


def benchmark_threshold(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> FloatArray:
    """One multiplier when F1 or F2 exceeds ``LIMIT_SD``."""
    pair = _load_pair(feature_names)

    def cov(rows: pd.DataFrame) -> FloatArray:
        z = rows[pair].to_numpy(dtype=np.float64)
        return np.asarray(np.any(z > LIMIT_SD, axis=1), dtype=np.float64)[:, None]

    return _fit_predict_covariates(
        train, test, feature_names, cfg, ["over_limit"], cov(train), cov(test), "bench-limit"
    )


def benchmark_knn(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> FloatArray:
    """M1 times the smoothed O/E of the ``KNN_NEIGHBOURS`` nearest training spells.

    Spells are compared on their mean live features; the O/E of the neighbours uses the M1
    spell probabilities of ``metrics.spell_table`` with one pseudo-event of smoothing.
    """
    names = list(feature_names)
    fits = fit_models(train, feature_names, cfg, "bench-knn")
    m1_train = predict_models(fits, train, feature_names, cfg)["m1"]
    m1_test = predict_models(fits, test, feature_names, cfg)["m1"]
    sp_tr = spell_table(train.assign(_m1=m1_train), "_m1", mean_cols=names)
    sp_te = spell_table(test.assign(_m1=m1_test), "_m1", mean_cols=names)
    x_tr = sp_tr[names].to_numpy(dtype=np.float64)
    observed = sp_tr["y"].to_numpy(dtype=np.float64)
    expected = sp_tr["_m1"].to_numpy(dtype=np.float64)
    k = min(KNN_NEIGHBOURS, x_tr.shape[0])
    _, idx = NearestNeighbors(n_neighbors=k).fit(x_tr).kneighbors(sp_te[names].to_numpy())
    rr = (observed[idx].sum(axis=1) + 1.0) / (expected[idx].sum(axis=1) + 1.0)
    return np.asarray(m1_test * rr[spell_index(test)], dtype=np.float64)


def comparator_ewma_features(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> FloatArray:
    """R4: F1 and F2 as EWMAs (``research.ewma_tau_min``) instead of rectangular windows."""
    pair = _load_pair(feature_names)
    rest = [f for f in feature_names if f not in pair]
    taus = [float(t) for t in cfg.research.ewma_tau_min]

    def cov(rows: pd.DataFrame) -> FloatArray:
        parts = [ewma(rows, pair, tau) for tau in taus]
        parts.append(rows[rest].to_numpy(dtype=np.float64))
        return np.concatenate(parts, axis=1)

    names = [f"{f}_ewma{t:g}" for t in taus for f in pair] + rest
    return _fit_predict_covariates(
        train, test, feature_names, cfg, names, cov(train), cov(test), "explore-ewma"
    )


def _team_relative_columns(rows: pd.DataFrame, features: Sequence[str]) -> FloatArray:
    """F1-F4 minus the mean of the team-mates on the pitch in the same minute
    (``norms.team_relative``; without team-mates a player keeps his own value)."""
    out = rows[list(features)].to_numpy(dtype=np.float64).copy()
    keys = ["match_id", "team_id", "period", "t_min"]
    pids = rows["player_id"].astype(str).to_numpy()
    groups = list(rows.groupby(keys, sort=False, observed=True).indices.values())
    for j in range(len(features)):
        values = out[:, j].copy()
        for idx in groups:
            rel = team_relative({pids[i]: float(values[i]) for i in idx})
            out[idx, j] = [rel[pids[i]] for i in idx]
    return out


def comparator_team_relative(
    train: pd.DataFrame, test: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> FloatArray:
    """R5: F1-F4 replaced by their team-relative variants (needs ``team_id``)."""
    rel = [f for f in TEAM_RELATIVE_FEATURES if f in feature_names]
    rest = [f for f in feature_names if f not in rel]

    def cov(rows: pd.DataFrame) -> FloatArray:
        return np.concatenate(
            [_team_relative_columns(rows, rel), rows[rest].to_numpy(dtype=np.float64)], axis=1
        )

    names = [f"{f}_team" for f in rel] + rest
    return _fit_predict_covariates(
        train, test, feature_names, cfg, names, cov(train), cov(test), "explore-team"
    )


BENCHMARKS: dict[str, Comparator] = {
    "ewma_acute_chronic": benchmark_ewma,
    "knn_players_like_him": benchmark_knn,
    "threshold_rule": benchmark_threshold,
}

#: Exploratory comparators by ``evaluation.exploratory`` id, with the columns each one needs.
EXPLORATORY: dict[str, tuple[Comparator, tuple[str, ...]]] = {
    "ewma_features": (comparator_ewma_features, ()),
    "team_relative": (comparator_team_relative, ("team_id",)),
}

#: Why an exploratory id cannot run on a person-minute table.
NOT_RUNNABLE: dict[str, str] = {
    "decayed_disrupt": "needs event times (R2)",
    "movement_pattern": "needs zone transitions from tracking (R3)",
    "fatigue_hmm": "needs a frozen hidden semi-Markov model on z (R1)",
}
