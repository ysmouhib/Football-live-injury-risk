from __future__ import annotations

import math
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from livrisk import riskmath
from livrisk.config import Config, load_config
from livrisk.contracts import (
    FeatureSnapshot,
    Mode,
    PositionGroup,
    PrematchBand,
    PreMatchRisk,
    Quality,
    RiskSnapshot,
)
from livrisk.engine.risk_engine import MatchState, RiskEngine
from livrisk.features.norms import NormTable, ResetInputs, build_norms
from livrisk.features.stream import FEATURE_NAMES
from livrisk.model.live_hazard import AuditReport, FittedHazard, Priors

pytestmark = [pytest.mark.milestone_m7]

P7: float = 0.0245  # docs/02_MATH.md §12 reference weekly probability
P7_LO, P7_HI = 0.01, 0.04
MATCH_DATE = date(2026, 9, 26)
HISTORY_DATES = (date(2026, 9, 5), date(2026, 9, 12), date(2026, 9, 19))
N_PARAMS = 1 + riskmath.N_BANDS + len(FEATURE_NAMES)


def _fitted(
    *,
    beta: tuple[float, ...] = (0.0,) * 6,
    theta: tuple[float, ...] = (0.0,) * 8,
    alpha: float = 0.0,
    cov_scale: float = 0.02,
    version: str = "test-v0",
) -> FittedHazard:
    """A hand-built fit: independent normal posterior with covariance ``cov_scale * I``.

    The band levels are normalised per docs/02_MATH.md §2 with the level absorbed into alpha,
    the invariant ``fit_map`` guarantees for real fits.
    """
    names = (
        "alpha",
        *(f"theta_{b}" for b in range(riskmath.N_BANDS)),
        *(f"beta_{f}" for f in FEATURE_NAMES),
    )
    point = np.asarray([alpha, *theta, *beta], dtype=np.float64)
    level = math.log(float(np.mean(np.exp(point[[1 + b for b in riskmath.REGULAR_BANDS]]))))
    point[0] += level
    point[1 : 1 + riskmath.N_BANDS] -= level
    cov = np.eye(len(names), dtype=np.float64) * cov_scale
    return FittedHazard(
        param_names=names,
        point=point,
        cov=cov,
        feature_names=FEATURE_NAMES,
        priors=Priors(),
        config_hash="test",
        version=version,
        n_rows=0,
        n_events=0,
    )


def _positions(players: list[str], gk: str | None = None) -> dict[str, PositionGroup]:
    out = {pid: PositionGroup.MF for pid in players}
    if gk is not None:
        out[gk] = PositionGroup.GK
    return out


def _norms(
    cfg: Config, players: list[str], reset: tuple[str, ...] = (), seed: int = 0
) -> NormTable:
    """Norms from three synthetic past matches per player (all before the match date)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in players:
        for played in HISTORY_DATES:
            for band in range(riskmath.N_BANDS):
                for feat in FEATURE_NAMES:
                    for value in rng.uniform(40.0, 160.0, size=6):
                        rows.append((pid, played, band, feat, float(value)))
    history = pd.DataFrame(rows, columns=["player_id", "match_date", "band", "feature", "value"])
    reset_inputs = {
        pid: ResetInputs(days_since_return=21.0, matches_since_return=0) for pid in reset
    }
    return build_norms(history, _positions(players), MATCH_DATE, cfg, reset_inputs=reset_inputs)


def _prematch(players: list[str], interval: bool = False) -> dict[str, PreMatchRisk]:
    return {
        pid: PreMatchRisk(
            player_id=pid,
            as_of_date=date(2026, 9, 25),
            p7=P7,
            p7_lo=P7_LO if interval else None,
            p7_hi=P7_HI if interval else None,
            model_version="weekly-v0",
        )
        for pid in players
    }


def _snapshot(
    pid: str,
    *,
    period: int = 2,
    value: float | None = 100.0,
    coverage: float | dict[str, float] = 1.0,
    identity: float = 1.0,
    quality: Quality = Quality.GOOD,
    reasons: tuple[str, ...] = (),
) -> FeatureSnapshot:
    cov = {name: coverage for name in FEATURE_NAMES} if isinstance(coverage, float) else coverage
    return FeatureSnapshot(
        match_id="m1",
        player_id=pid,
        period=period,
        as_of_t_s=3600.0,
        cutoff_t_s=3540.0,
        features={name: value for name in FEATURE_NAMES},
        z={},
        coverage=cov,
        identity_confidence=identity,
        quality=quality,
        quality_reasons=reasons,
    )


def test_beta_zero_flat_profile_equals_riskmath(config_path: Path) -> None:
    cfg = load_config(config_path)
    players = ["P001"]
    engine = RiskEngine(
        cfg, _fitted(), _norms(cfg, players), _prematch(players), _positions(players)
    )
    state = MatchState(match_id="m1", period=2, t_s=3600.0)  # 60:00 on the match clock
    snap = _snapshot("P001")
    out = engine.update([snap], state, {"P001": 8.0})
    (row,) = out.risks
    lam = riskmath.weekly_to_match_cumhaz(P7, cfg.prematch.match_share_prior)
    flat = riskmath.flat_time_profile()
    assert row.r_rest == riskmath.projected_probability(lam, flat, 1.0, 60.0, 2, None)
    assert row.p15 == riskmath.projected_probability(lam, flat, 1.0, 60.0, 2, 15.0)
    assert row.r_usual == row.r_rest
    assert row.rr == 1.0
    assert row.prematch_band == PrematchBand.WATCH  # 0.012 <= 0.0147728659 < 0.020
    assert row.quality == Quality.GOOD
    assert row.mode == Mode.RISK
    assert row.model_version == "test-v0"
    assert row.period == 2 and row.as_of_t_s == 3600.0
    # Announced stoppage extends the projection through the same riskmath schedule.
    (row_s,) = engine.update([snap], replace(state, stoppage_2h_min=5.0), {"P001": 8.0}).risks
    assert row_s.r_rest == riskmath.projected_probability(
        lam, flat, 1.0, 60.0, 2, None, stoppage_2h_min=5.0
    )
    # Deterministic: identical inputs give identical rows (seeded posterior draws).
    assert engine.update([snap], state, {"P001": 8.0}).risks == out.risks
    (enriched,) = out.features
    assert enriched.player_id == "P001"
    assert all(v is not None for v in enriched.z.values())


def test_quality_rules(config_path: Path) -> None:
    cfg = load_config(config_path)
    beta = (math.log(2.0), 0.0, 0.0, 0.0, 0.0, 0.0)  # only hsr_5m carries signal
    players = ["P001", "P002", "P003", "P004", "P005", "P006", "P007"]
    norms = _norms(cfg, players)
    prematch = _prematch([p for p in players if p != "P007"])  # P007 has no pre-match row
    engine = RiskEngine(cfg, _fitted(beta=beta), norms, prematch, _positions(players, gk="P006"))
    state = MatchState(match_id="m1", period=2, t_s=3600.0)
    low_cov = {name: 1.0 for name in FEATURE_NAMES}
    low_cov["hsr_5m"] = 0.5  # below quality.min_coverage (0.7)
    snaps = [
        _snapshot("P001", value=2000.0),  # valid extreme load: rr hits the rr_max bound
        _snapshot(
            "P002",
            value=2000.0,
            coverage=low_cov,
            quality=Quality.DEGRADED,
            reasons=("low_coverage",),
        ),
        _snapshot("P003", value=None, coverage=0.0, quality=Quality.PREMATCH_ONLY),
        _snapshot("P004", value=2000.0),
        _snapshot(
            "P005",
            value=2000.0,
            identity=0.5,
            quality=Quality.DEGRADED,
            reasons=("identity_low",),
        ),
        _snapshot("P006", value=2000.0),  # goalkeeper: excluded from risk outputs
        _snapshot("P007", value=2000.0),  # no pre-match row: skipped
    ]
    ages = {p: 8.0 for p in players}
    ages["P004"] = 600.0  # older than quality.stale_hide_s (180)
    out = engine.update(snaps, state, ages)
    rows = {r.player_id: r for r in out.risks}
    assert set(rows) == {"P001", "P002", "P003", "P004", "P005"}
    assert rows["P001"].quality == Quality.GOOD
    assert rows["P001"].rr == cfg.engine.rr_max
    assert rows["P001"].top_reason_code == "HSR_HIGH"
    # Low coverage zeroes the feature's z*: the live multiplier falls back to 1.
    assert rows["P002"].quality == Quality.DEGRADED
    assert rows["P002"].rr == 1.0
    # No valid feature at all: pre-match fallback.
    assert rows["P003"].quality == Quality.PREMATCH_ONLY
    assert rows["P003"].rr == 1.0
    assert rows["P003"].r_rest == rows["P003"].r_usual
    # Stale: the live signal is zeroed, the row is marked stale (never shown as live).
    assert rows["P004"].quality == Quality.STALE
    assert rows["P004"].rr == 1.0
    assert rows["P004"].data_age_s == 600.0
    assert rows["P004"].r_rest == rows["P003"].r_rest  # same fallback for the same p7
    # Low identity gates the tracking features but not the event-feed one: degraded, not lost.
    assert rows["P005"].quality == Quality.DEGRADED
    assert rows["P005"].rr == 1.0
    # Quality reasons propagate to the enriched feature snapshot.
    feats = {f.player_id: f for f in out.features}
    assert "low_coverage" in feats["P002"].quality_reasons
    assert "identity_low" in feats["P005"].quality_reasons


def test_norm_reasons_reach_the_feature_snapshot(config_path: Path) -> None:
    cfg = load_config(config_path)
    players = ["P001", "P002"]
    norms = _norms(cfg, players, reset=("P001",))
    all_players = [*players, "P003"]  # P003 has no history: position norm, few effective matches
    engine = RiskEngine(cfg, _fitted(), norms, _prematch(all_players), _positions(all_players))
    state = MatchState(match_id="m1", period=2, t_s=3600.0)
    snaps = [_snapshot(p) for p in all_players]
    out = engine.update(snaps, state, {p: 8.0 for p in all_players})
    feats = {f.player_id: f for f in out.features}
    assert "norm_reset" in feats["P001"].quality_reasons
    assert "norm_reset" not in feats["P002"].quality_reasons
    assert "few_matches" in feats["P003"].quality_reasons
    assert "few_matches" not in feats["P002"].quality_reasons
    # z values are filled by the engine from the norm table.
    assert all(v is not None for v in feats["P001"].z.values())


def test_intervals_contain_point_and_widen_with_cov(config_path: Path) -> None:
    cfg = load_config(config_path)
    players = ["P001"]
    norms = _norms(cfg, players)
    prematch = _prematch(players)
    state = MatchState(match_id="m1", period=2, t_s=3600.0)
    snap = _snapshot("P001")
    theta = (0.20, 0.10, 0.0, -0.10, 0.15, -0.05, -0.30, 0.0)

    def row(cov_scale: float) -> RiskSnapshot:
        engine = RiskEngine(
            cfg,
            _fitted(
                beta=(0.3, -0.1, 0.2, 0.0, 0.1, 0.0),
                theta=theta,
                alpha=-0.1,
                cov_scale=cov_scale,
            ),
            norms,
            prematch,
            _positions(players),
        )
        return engine.update([snap], state, {"P001": 8.0}).risks[0]

    base = row(0.02)
    wide = row(0.5)
    tiny = row(1e-12)
    for r in (base, wide, tiny):
        assert r.p15_lo <= r.p15 <= r.p15_hi
        assert r.r_rest_lo <= r.r_rest <= r.r_rest_hi
    # Point estimates are draw-free: identical across covariance scales.
    assert base.p15 == wide.p15 == tiny.p15
    assert base.r_rest == wide.r_rest == tiny.r_rest
    # Same seed, same standard-normal draws: a larger covariance widens the intervals.
    assert wide.p15_hi - wide.p15_lo > base.p15_hi - base.p15_lo
    assert wide.r_rest_hi - wide.r_rest_lo > base.r_rest_hi - base.r_rest_lo
    # A near-degenerate covariance collapses the interval onto the point estimate.
    assert tiny.p15_hi - tiny.p15_lo < 1e-6
    assert tiny.r_rest_hi - tiny.r_rest_lo < 1e-6


def test_monitor_only_mode(config_path: Path) -> None:
    cfg = load_config(config_path)
    monitor = cfg.model_copy(
        update={"engine": cfg.engine.model_copy(update={"mode": "monitor_only"})}
    )
    players = ["P001", "P002", "P003"]
    fitted = _fitted(
        beta=(0.4, -0.3, 0.2, 0.0, 0.1, 0.0),
        theta=(0.2, 0.0, 0.0, 0.0, 0.1, 0.0, -0.3, 0.0),
        alpha=-0.2,
    )
    norms = _norms(monitor, players)
    engine = RiskEngine(monitor, fitted, norms, _prematch(players), _positions(players))
    state = MatchState(match_id="m1", period=1, t_s=1500.0)  # 25:00
    snaps = [_snapshot(p, period=1, value=2000.0) for p in players]
    out = engine.update(snaps, state, {p: 5.0 for p in players})
    lam = riskmath.weekly_to_match_cumhaz(P7, monitor.prematch.match_share_prior) * math.exp(
        fitted.alpha
    )
    want = riskmath.projected_probability(lam, fitted.g_levels(), 1.0, 25.0, 1, None)
    assert len(out.risks) == 3
    for row in out.risks:
        assert row.mode == Mode.MONITOR_ONLY
        assert row.rr == 1.0
        assert row.r_rest == row.r_usual == want
        assert row.r_rest_lo <= row.r_rest <= row.r_rest_hi


def test_update_speed_30_players(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert cfg.engine.interval_draws == 1000
    players = [f"P{i:03d}" for i in range(1, 31)]
    engine = RiskEngine(
        cfg,
        _fitted(
            beta=(0.3, 0.2, -0.1, 0.05, 0.1, 0.0),
            theta=(0.1, 0.05, 0.0, 0.0, 0.1, 0.0, -0.15, 0.0),
            alpha=-0.05,
            cov_scale=0.05,
        ),
        _norms(cfg, players, seed=1),
        _prematch(players),
        _positions(players),
    )
    state = MatchState(match_id="m1", period=2, t_s=3600.0)
    snaps = [_snapshot(p) for p in players]
    ages = {p: 8.0 for p in players}
    engine.update(snaps, state, ages)  # warm-up
    t0 = time.perf_counter()
    out = engine.update(snaps, state, ages)
    elapsed = time.perf_counter() - t0
    assert len(out.risks) == 30
    assert elapsed < 0.5


def test_prematch_interval_widens_displayed_interval(config_path: Path) -> None:
    cfg = load_config(config_path)
    players = ["P001"]
    norms = _norms(cfg, players)
    fitted = _fitted(beta=(0.3, 0.0, 0.0, 0.0, 0.0, 0.0), cov_scale=0.02)
    state = MatchState(match_id="m1", period=2, t_s=3600.0)
    snap = _snapshot("P001")
    ages = {"P001": 8.0}

    def row(interval: bool) -> RiskSnapshot:
        engine = RiskEngine(
            cfg, fitted, norms, _prematch(players, interval=interval), _positions(players)
        )
        return engine.update([snap], state, ages).risks[0]

    base = row(False)
    spread = row(True)
    # docs/02_MATH.md §12 reference SD for this interval.
    assert riskmath.log_cumhaz_sd_from_interval(P7_LO, P7_HI, 0.90) == pytest.approx(
        0.42606, abs=1e-5
    )
    # Point estimates are draw-free: unchanged.
    assert spread.p15 == base.p15
    assert spread.r_rest == base.r_rest
    # Same parameter draws plus the ln-Lambda jitter: strictly wider intervals.
    assert spread.p15_lo < base.p15_lo
    assert spread.p15_hi > base.p15_hi
    assert spread.r_rest_lo < base.r_rest_lo
    assert spread.r_rest_hi > base.r_rest_hi


def test_failed_audit_ships_weighted_draws(config_path: Path) -> None:
    cfg = load_config(config_path)
    players = ["P001"]
    fitted = _fitted(cov_scale=0.02)
    norms = _norms(cfg, players)
    prematch = _prematch(players)
    positions = _positions(players)
    state = MatchState(match_id="m1", period=2, t_s=3600.0)
    snap = _snapshot("P001")
    ages = {"P001": 8.0}
    n = 2000
    draws = np.zeros((n, N_PARAMS), dtype=np.float64)
    draws[:, 0] = np.linspace(-0.8, 0.8, n)  # spread in alpha only
    weights_high = np.exp(2.0 * draws[:, 0])

    def audit(passed: bool, weights: NDArray[np.float64]) -> AuditReport:
        return AuditReport(
            passed=passed,
            khat=0.9,
            ess=100.0,
            n_draws=n,
            tail_draws=50,
            battery_cases=0,
            max_rel_diff=0.5,
            proposal_draws=draws,
            weights=weights,
        )

    flat = RiskEngine(cfg, fitted, norms, prematch, positions, audit=audit(False, np.ones(n)))
    high = RiskEngine(cfg, fitted, norms, prematch, positions, audit=audit(False, weights_high))
    row_flat = flat.update([snap], state, ages).risks[0]
    row_high = high.update([snap], state, ages).risks[0]
    # The stored draws are shared; the importance weights shift the percentiles toward high alpha.
    assert row_high.p15_lo > row_flat.p15_lo
    assert row_high.p15_hi > row_flat.p15_hi
    assert row_high.r_rest_lo > row_flat.r_rest_lo
    assert row_high.r_rest_hi > row_flat.r_rest_hi
    # Point estimates are draw-free: identical.
    assert row_high.p15 == row_flat.p15
    # A passing audit ships no weights: the Gaussian path is used, identical to no audit.
    plain = RiskEngine(cfg, fitted, norms, prematch, positions)
    passed = RiskEngine(cfg, fitted, norms, prematch, positions, audit=audit(True, weights_high))
    assert passed.update([snap], state, ages).risks == plain.update([snap], state, ages).risks
