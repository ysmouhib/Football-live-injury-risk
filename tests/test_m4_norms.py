"""M4 acceptance tests: player, position and team norms (docs/02_MATH.md §3).

Histories are small synthetic frames built inline with columns player_id, match_date, band,
feature, value (raw). Under the default config, hsr_5m is log1p-transformed, recovery_10m is
log-transformed and disrupt_5m is untransformed.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from livrisk import riskmath
from livrisk.app.wording import QUALITY_REASON_TEXT
from livrisk.config import load_config
from livrisk.contracts import PositionGroup
from livrisk.features.norms import (
    ResetInputs,
    build_norms,
    occupancy_drift,
    reset_active,
    team_relative,
    z_for,
)
from livrisk.markov import tv_distance

pytestmark = [pytest.mark.milestone_m4]

MF = PositionGroup.MF
FEAT = "disrupt_5m"  # identity transform in the default config
BAND = 0
AS_OF = date(2026, 3, 1)


def _history(rows: list[tuple[str, date, int, str, float]]) -> pd.DataFrame:
    """History frame from (player_id, match_date, band, feature, raw value) rows."""
    return pd.DataFrame(rows, columns=["player_id", "match_date", "band", "feature", "value"])


def _level_history(levels: dict[str, float], days: list[date], feature: str = FEAT) -> pd.DataFrame:
    """One row per player and match date, at the player's constant raw level, band 0."""
    rows = [(p, d, BAND, feature, level) for p, level in levels.items() for d in days]
    return _history(rows)


def test_team_relative_removes_shared_shift() -> None:
    z = {"a": 1.0, "b": 1.0, "c": 1.0, "d": None}
    out = team_relative(z)
    assert out["a"] == pytest.approx(0.0)
    assert out["d"] is None


def test_team_relative_without_valued_team_mates_keeps_z() -> None:
    out = team_relative({"a": 1.5, "b": None, "c": None})
    assert out == {"a": 1.5, "b": None, "c": None}
    assert team_relative({"a": -0.5}) == {"a": -0.5}


def test_norms_ignore_future_matches(config_path: Path) -> None:
    cfg = load_config(config_path)
    days = [AS_OF - timedelta(days=7 * (k + 1)) for k in range(8)]
    history = _level_history({"P1": 2.0, "P2": 4.0, "P3": 3.0}, days)
    positions = {"P1": MF, "P2": MF, "P3": MF}
    base = build_norms(history, positions, AS_OF, cfg)

    leak_rows = [
        (p, d, BAND, FEAT, 1.0e9) for p in ("P1", "P2") for d in (AS_OF, AS_OF + timedelta(days=1))
    ]
    leaked = build_norms(pd.concat([history, _history(leak_rows)]), positions, AS_OF, cfg)

    for pid in ("P1", "P2", "P3"):
        assert base.get(pid, MF, BAND, FEAT) == leaked.get(pid, MF, BAND, FEAT)
    # the position norm behind a player without history is unchanged too
    assert base.get("P9", MF, BAND, FEAT) == leaked.get("P9", MF, BAND, FEAT)


def test_new_player_gets_position_norm_and_veteran_gets_own(config_path: Path) -> None:
    cfg = load_config(config_path)
    nc = cfg.norms
    days = [AS_OF - timedelta(days=4 * (k + 1)) for k in range(20)]
    history = _level_history({"VET": 10.0, "QA": 20.0, "QB": 20.0}, days)
    positions = {"VET": MF, "QA": MF, "QB": MF, "NEW": MF}
    table = build_norms(history, positions, AS_OF, cfg)

    # same dates for all three players: the position mean is the plain mean of the levels
    pos_mean = (10.0 + 20.0 + 20.0) / 3.0
    new = table.get("NEW", MF, BAND, FEAT)
    assert new.n_matches == 0
    assert new.n_effective == 0.0
    assert not new.reset
    assert new.mean == pytest.approx(pos_mean)

    vet = table.get("VET", MF, BAND, FEAT)
    w = riskmath.recency_weights(np.arange(20, dtype=np.float64), nc.recency_half_life_matches)
    n_eff = riskmath.kish_effective_n(w)
    assert vet.n_matches == 20
    assert vet.n_effective == pytest.approx(n_eff)
    assert vet.mean == pytest.approx(riskmath.shrink_mean(n_eff, 10.0, nc.kappa_matches, pos_mean))
    # his own level dominates the shrinkage; the position level dominates the fallback
    assert abs(vet.mean - 10.0) < abs(vet.mean - pos_mean)
    assert vet.sd < new.sd  # a constant own history adds no spread of its own
    # identity transform: z of a value equal to the shrunk mean is 0
    assert z_for(table, "VET", MF, BAND, FEAT, vet.mean, cfg) == pytest.approx(0.0)


def test_recency_weighting_and_effective_matches(config_path: Path) -> None:
    cfg = load_config(config_path)
    nc = cfg.norms
    n_matches = 20
    days = [AS_OF - timedelta(days=4 * (n_matches - k)) for k in range(n_matches)]
    levels = np.array([100.0] * 15 + [200.0] * 5)  # level shift in the last 5 matches
    rows = [
        (p, days[k], BAND, FEAT, float(levels[k]) if p == "P1" else 150.0)
        for p in ("P1", "P2")
        for k in range(n_matches)
    ]
    table = build_norms(_history(rows), {"P1": MF, "P2": MF}, AS_OF, cfg)

    ages = np.arange(n_matches - 1, -1, -1, dtype=np.float64)  # match k has age 19 - k
    w = riskmath.recency_weights(ages, nc.recency_half_life_matches)
    xw, sw = riskmath.weighted_mean_sd(levels, w)
    n_eff = riskmath.kish_effective_n(w)
    # P2 is flat at 150 on the same dates, so it contributes 150 with the same weights
    pos_mean = (xw + 150.0) / 2.0
    _, s_pos = riskmath.weighted_mean_sd(
        np.concatenate([levels, np.full(20, 150.0)]), np.concatenate([w, w])
    )

    entry = table.get("P1", MF, BAND, FEAT)
    assert entry.n_matches == n_matches
    assert entry.n_effective == pytest.approx(n_eff)
    assert entry.n_effective < n_matches  # Kish n of the recency weights, not a count
    assert entry.mean == pytest.approx(riskmath.shrink_mean(n_eff, xw, nc.kappa_matches, pos_mean))
    assert entry.sd == pytest.approx(riskmath.shrink_sd(n_eff, sw, nc.kappa_matches, s_pos))
    # the shrunk mean moves toward the recent level more than an unweighted mean would
    unweighted = riskmath.shrink_mean(n_eff, float(levels.mean()), nc.kappa_matches, pos_mean)
    assert 200.0 - entry.mean < 200.0 - unweighted


def test_spread_uses_non_overlapping_windows(config_path: Path) -> None:
    cfg = load_config(config_path)
    days = [AS_OF - timedelta(days=7 * (k + 1)) for k in range(5)]
    snap_rows: list[tuple[str, date, int, str, float]] = []
    compact_rows: list[tuple[str, date, int, str, float]] = []
    for d in days:
        for value in (1.0, 2.0, 3.0):
            snap_rows.extend(("P1", d, BAND, FEAT, value) for _ in range(30))
        compact_rows.extend(("P1", d, BAND, FEAT, v) for v in (1.0, 2.0, 3.0))
        snap_rows.extend(("P2", d, BAND, FEAT, v) for v in (2.0, 2.0, 2.0))
        compact_rows.extend(("P2", d, BAND, FEAT, v) for v in (2.0, 2.0, 2.0))
    positions = {"P1": MF, "P2": MF}
    snap = build_norms(_history(snap_rows), positions, AS_OF, cfg)
    compact = build_norms(_history(compact_rows), positions, AS_OF, cfg)

    # 90 ten-second snapshots per band and match carry the same evidence as the 3 window
    # means: identical norm, and n_effective counts matches, never windows
    assert snap.get("P1", MF, BAND, FEAT) == compact.get("P1", MF, BAND, FEAT)
    entry = snap.get("P1", MF, BAND, FEAT)
    assert entry.n_matches == 5
    w = riskmath.recency_weights(
        np.arange(5, dtype=np.float64), cfg.norms.recency_half_life_matches
    )
    assert entry.n_effective == pytest.approx(riskmath.kish_effective_n(w))
    assert entry.n_effective != 3 * 5  # not the number of windows


def test_norm_reset_after_absence(config_path: Path) -> None:
    cfg = load_config(config_path)
    nc = cfg.norms

    assert reset_active(14.0, None, 0, cfg)
    assert reset_active(14.0, None, 2, cfg)
    assert not reset_active(14.0, None, 3, cfg)
    assert not reset_active(13.9, None, 0, cfg)
    assert reset_active(None, 43.0, 0, cfg)
    assert not reset_active(None, 42.0, 0, cfg)
    assert not reset_active(None, None, 0, cfg)

    days = [AS_OF - timedelta(days=60 + 4 * k) for k in range(10)]
    history = _level_history({"P1": 100.0, "P2": 200.0}, days)
    positions = {"P1": MF, "P2": MF, "P3": MF}
    w = riskmath.recency_weights(np.arange(10, dtype=np.float64), nc.recency_half_life_matches)
    n_uncapped = riskmath.kish_effective_n(w)
    pos_mean = 150.0

    plain = build_norms(history, positions, AS_OF, cfg).get("P1", MF, BAND, FEAT)
    assert not plain.reset
    assert plain.quality_reasons == ()
    assert plain.n_effective == pytest.approx(n_uncapped)
    assert plain.mean == pytest.approx(
        riskmath.shrink_mean(n_uncapped, 100.0, nc.kappa_matches, pos_mean)
    )

    reset_inputs = {"P1": ResetInputs(days_since_return=14.0, matches_since_return=0)}
    table = build_norms(history, positions, AS_OF, cfg, reset_inputs=reset_inputs)
    entry = table.get("P1", MF, BAND, FEAT)
    assert entry.reset
    assert entry.quality_reasons == ("norm_reset",)
    assert entry.n_matches == 10
    assert entry.n_effective == pytest.approx(nc.reset_own_weight_matches)
    assert entry.mean == pytest.approx(
        riskmath.shrink_mean(nc.reset_own_weight_matches, 100.0, nc.kappa_matches, pos_mean)
    )
    assert entry.mean > plain.mean  # capped at one effective match: mostly the position norm
    assert "norm_reset" in QUALITY_REASON_TEXT

    # the z path uses the reset (position-dominated) norm
    z = z_for(table, "P1", MF, BAND, FEAT, 100.0, cfg)
    assert z == pytest.approx((100.0 - entry.mean) / entry.sd)

    # a returning player without usable history still carries the reset on the fallback
    back = build_norms(
        history,
        positions,
        AS_OF,
        cfg,
        reset_inputs={"P3": ResetInputs(days_since_return=20.0, matches_since_return=1)},
    )
    fallback = back.get("P3", MF, BAND, FEAT)
    assert fallback.reset
    assert fallback.n_matches == 0
    assert fallback.n_effective == 0.0


def test_z_at_shrunk_mean_is_zero_and_transforms_applied(config_path: Path) -> None:
    cfg = load_config(config_path)
    nc = cfg.norms
    days = [AS_OF - timedelta(days=7 * (k + 1)) for k in range(6)]
    positions = {"P1": MF, "P2": MF}
    w = riskmath.recency_weights(np.arange(6, dtype=np.float64), nc.recency_half_life_matches)
    n_eff = riskmath.kish_effective_n(w)

    table = build_norms(
        _level_history({"P1": 50.0, "P2": 70.0}, days, "hsr_5m"), positions, AS_OF, cfg
    )
    entry = table.get("P1", MF, BAND, "hsr_5m")
    pos_mean = (math.log1p(50.0) + math.log1p(70.0)) / 2.0
    assert entry.mean == pytest.approx(
        riskmath.shrink_mean(n_eff, math.log1p(50.0), nc.kappa_matches, pos_mean)
    )
    assert z_for(table, "P1", MF, BAND, "hsr_5m", math.expm1(entry.mean), cfg) == pytest.approx(0.0)
    z = z_for(table, "P1", MF, BAND, "hsr_5m", 50.0, cfg)
    assert z == pytest.approx((math.log1p(50.0) - entry.mean) / entry.sd)

    log_table = build_norms(
        _level_history({"P1": 20.0, "P2": 40.0}, days, "recovery_10m"), positions, AS_OF, cfg
    )
    log_entry = log_table.get("P1", MF, BAND, "recovery_10m")
    pos_log = (math.log(20.0) + math.log(40.0)) / 2.0
    assert log_entry.mean == pytest.approx(
        riskmath.shrink_mean(n_eff, math.log(20.0), nc.kappa_matches, pos_log)
    )
    assert z_for(
        log_table, "P1", MF, BAND, "recovery_10m", math.exp(log_entry.mean), cfg
    ) == pytest.approx(0.0)

    # missing values and unknown norms give None
    assert z_for(table, "P1", MF, BAND, "hsr_5m", None, cfg) is None
    assert z_for(table, "P9", PositionGroup.GK, BAND, "hsr_5m", 50.0, cfg) is None
    with pytest.raises(KeyError):
        table.get("P9", PositionGroup.GK, BAND, "hsr_5m")


def test_occupancy_drift_is_total_variation() -> None:
    recent = [0.1, 0.2, 0.4, 0.3]
    usual = [0.25, 0.25, 0.25, 0.25]
    assert occupancy_drift(recent, usual) == pytest.approx(tv_distance(recent, usual))
    assert occupancy_drift(usual, usual) == pytest.approx(0.0)
