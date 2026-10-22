from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from livrisk import riskmath
from livrisk.config import load_config
from livrisk.model.live_hazard import (
    FittedHazard,
    Priors,
    _expansion_matrix,
    _log_posterior_draws,
    build_design,
    fit_map,
    interval_audit,
    neg_log_posterior,
    neg_log_posterior_grad,
)
from livrisk.synth.generator import SynthTruth, generate_person_minutes
from tests.conftest import DEV_SEEDS

pytestmark = [pytest.mark.milestone_m6]

TRUTH = SynthTruth(p7_mean=0.10)
N_PLAYER_MATCHES = 7000  # ~400 events at p7_mean = 0.10

#: Fits are deterministic given (seed, beta); shared between the coverage and profile tests.
_FIT_CACHE: dict[tuple[int, tuple[float, ...]], FittedHazard] = {}


def _fit(seed: int, truth: SynthTruth) -> FittedHazard:
    key = (seed, truth.beta)
    if key not in _FIT_CACHE:
        df = generate_person_minutes(
            N_PLAYER_MATCHES, seed=seed, truth=truth, informative_censoring=False
        )
        design = build_design(df, truth.feature_names, truth.match_share)
        _FIT_CACHE[key] = fit_map(design, Priors(), config_hash="test", version="test")
    return _FIT_CACHE[key]


def _beta_sd(f: FittedHazard) -> NDArray[np.float64]:
    k0 = len(f.param_names) - len(f.beta)
    return np.asarray(np.sqrt(np.diag(np.asarray(f.cov))[k0:]), dtype=np.float64)


def test_parameter_recovery_coverage() -> None:
    inside = np.zeros(len(TRUTH.beta))
    seeds = DEV_SEEDS
    for s in seeds:
        f = _fit(s, TRUTH)
        sd = _beta_sd(f)
        lo, hi = np.asarray(f.beta) - 1.645 * sd, np.asarray(f.beta) + 1.645 * sd
        inside += (lo <= np.asarray(TRUTH.beta)) & (np.asarray(TRUTH.beta) <= hi)
    cover = inside / len(seeds)
    assert np.all((cover >= 0.85) & (cover <= 0.95)), cover


def test_null_signal_shrinks_to_zero() -> None:
    null = SynthTruth(p7_mean=0.10, beta=(0.0,) * 6)
    fits = [_fit(s, null) for s in DEV_SEEDS]
    betas = np.array([np.asarray(f.beta) for f in fits])
    assert np.all(np.abs(betas.mean(axis=0)) < 0.1)
    # 90 % Laplace intervals contain 0 in >= 85 % of seeds per coefficient.
    inside = np.zeros(len(null.beta))
    for f in fits:
        sd = _beta_sd(f)
        inside += (np.asarray(f.beta) - 1.645 * sd <= 0.0) & (
            np.asarray(f.beta) + 1.645 * sd >= 0.0
        )
    assert np.all(inside / len(fits) >= 0.85), inside / len(fits)


def test_time_profile_recovery() -> None:
    true_g = riskmath.normalise_time_profile(TRUTH.theta)
    gs = np.array([_fit(s, TRUTH).g_levels() for s in DEV_SEEDS[:10]])
    med = np.median(gs, axis=0)
    regular = list(riskmath.REGULAR_BANDS)
    assert np.all(np.abs(med[regular] / true_g[regular] - 1) < 0.2)


def test_interval_audit_passes_when_correct_and_flags_shrunk_covariance(
    config_path: Path,
) -> None:
    cfg = load_config(config_path).model.interval_audit
    df = generate_person_minutes(3000, seed=5, truth=TRUTH, informative_censoring=False)
    design = build_design(df, TRUTH.feature_names, TRUTH.match_share)
    fitted = fit_map(design, Priors(), config_hash="test", version="test")

    report = interval_audit(fitted, design, cfg, seed=123)
    assert report.khat < cfg.khat_max
    assert report.max_rel_diff <= cfg.quantile_rel_tol
    assert report.passed

    shrunk = replace(fitted, cov=np.asarray(fitted.cov) * 0.25)
    bad = interval_audit(shrunk, design, cfg, seed=123)
    assert not bad.passed


def test_analytic_gradient_matches_finite_differences() -> None:
    df = generate_person_minutes(400, seed=7, truth=TRUTH, informative_censoring=False)
    design = build_design(df, TRUTH.feature_names, TRUTH.match_share)
    priors = Priors()
    rng = np.random.default_rng(0)
    n_free = len(design.feature_names) + 8
    weights = rng.uniform(0.2, 2.0, size=design.n_rows)
    for phi, w in ((np.zeros(n_free), None), (rng.normal(0.0, 0.2, size=n_free), weights)):
        grad = neg_log_posterior_grad(phi, design, priors, w)
        num = np.empty(n_free)
        for i in range(n_free):
            step = np.zeros(n_free)
            step[i] = 1e-6
            num[i] = (
                neg_log_posterior(phi + step, design, priors, w)
                - neg_log_posterior(phi - step, design, priors, w)
            ) / 2e-6
        np.testing.assert_allclose(grad, num, rtol=1e-5, atol=1e-6)


def test_draw_log_posterior_matches_fit_objective() -> None:
    # The audit scores draws in the stored (full) parameterisation; the fit objective lives in
    # the free one. A point mapped by the expansion matrix must score identically in both.
    df = generate_person_minutes(60, seed=9, truth=TRUTH, informative_censoring=False)
    design = build_design(df, TRUTH.feature_names, TRUTH.match_share)
    priors = Priors()
    rng = np.random.default_rng(1)
    n_free = 8 + len(TRUTH.feature_names)
    phis = rng.normal(0.0, 0.15, size=(5, n_free))
    psis = (_expansion_matrix(len(TRUTH.feature_names)) @ phis.T).T
    got = _log_posterior_draws(psis, design, priors)
    want = np.array([-neg_log_posterior(phi, design, priors) for phi in phis])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_fit_scales_to_a_million_person_minutes() -> None:
    df = generate_person_minutes(14_000, seed=11, truth=TRUTH, informative_censoring=False)
    assert len(df) >= 1_000_000
    t0 = time.perf_counter()
    design = build_design(df, TRUTH.feature_names, TRUTH.match_share)
    fitted = fit_map(design, Priors(), config_hash="test", version="test")
    elapsed = time.perf_counter() - t0
    assert fitted.n_rows == len(df)
    assert elapsed < 60.0


def test_fitted_hazard_json_roundtrip(tmp_path: Path) -> None:
    df = generate_person_minutes(300, seed=13, truth=TRUTH, informative_censoring=False)
    design = build_design(df, TRUTH.feature_names, TRUTH.match_share)
    fitted = fit_map(design, Priors(), config_hash="abc123", version="v0")
    # The fast path has no stoppage minutes: bands 3 and 7 stay at the prior mean and get the
    # same level shift, so their g levels are equal.
    assert fitted.theta[3] == fitted.theta[7]
    assert np.isclose(fitted.g_levels()[list(riskmath.REGULAR_BANDS)].mean(), 1.0)

    path = tmp_path / "fit.json"
    fitted.save(path)
    back = FittedHazard.load(path)
    assert back.param_names == fitted.param_names
    assert back.config_hash == "abc123" and back.version == "v0"
    assert back.n_rows == fitted.n_rows and back.n_events == fitted.n_events
    np.testing.assert_allclose(back.point, fitted.point, rtol=0, atol=0)
    np.testing.assert_allclose(back.cov, fitted.cov, rtol=0, atol=0)
    assert back.priors == fitted.priors


def test_build_design_offset_uses_riskmath(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[float, float]] = []

    def fake(p7: float, match_share: float = 0.6) -> float:
        calls.append((p7, match_share))
        return 0.123

    df = generate_person_minutes(20, seed=3, truth=TRUTH, informative_censoring=False)
    monkeypatch.setattr(riskmath, "weekly_to_match_cumhaz", fake)
    design = build_design(df, TRUTH.feature_names, TRUTH.match_share)
    assert np.allclose(design.offset, math.log(0.123 / riskmath.REGULAR_MINUTES))
    assert len(calls) == df["p7"].nunique()
    assert all(share == TRUTH.match_share for _, share in calls)


def test_build_design_keeps_censoring_rows_as_exposure() -> None:
    df = generate_person_minutes(50, seed=4, truth=TRUTH, informative_censoring=False)
    design = build_design(df, TRUTH.feature_names, TRUTH.match_share)
    assert design.n_rows == len(df)
    contact = df["contact"].to_numpy(dtype=np.bool_)
    assert contact.any()
    assert int(design.y[contact].sum()) == 0
    assert design.player_id.shape == (len(df),)


def test_build_design_rejects_rows_after_censoring() -> None:
    df = generate_person_minutes(50, seed=4, truth=TRUTH, informative_censoring=False)
    extra = df[df["contact"] | df["substituted_after"]].iloc[:1].copy()
    extra["t_min"] += 1.0
    extra["y"] = 0
    extra["contact"] = False
    extra["substituted_after"] = False
    bad = pd.concat([df, extra], ignore_index=True)
    with pytest.raises(ValueError, match="censored"):
        build_design(bad, TRUTH.feature_names, TRUTH.match_share)
