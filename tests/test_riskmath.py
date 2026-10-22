"""Reference values from docs/02_MATH.md §12. These tests must never be weakened."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from livrisk import riskmath as rm


def test_weekly_to_match_cumhaz_reference() -> None:
    assert rm.weekly_to_match_cumhaz(0.0245, 0.6) == pytest.approx(0.01488307135, rel=1e-9)


def test_baseline_hazard_reference() -> None:
    lam = rm.baseline_hazard_per_min(rm.weekly_to_match_cumhaz(0.0245, 0.6), 1.0)
    assert lam == pytest.approx(1.6536746e-4, rel=1e-7)


def test_prob_from_cumhaz_reference() -> None:
    assert rm.prob_from_cumhaz(0.0148830714) == pytest.approx(0.0147728659, rel=1e-8)


def test_window_probability_reference() -> None:
    lam = 1.6536746e-4
    p = rm.window_probability([lam] * 15)
    assert p == pytest.approx(-math.expm1(-15 * lam), rel=1e-12)
    assert p == pytest.approx(0.0024774, abs=5e-7)


def test_alert_ppv_and_alerts_per_match() -> None:
    assert rm.alert_ppv(0.3, 0.002625, 0.02) == pytest.approx(0.039375)
    assert rm.alerts_per_match(0.02) == pytest.approx(1.32)


def test_schoenfeld_reference() -> None:
    assert rm.schoenfeld_events(1.22) == 199
    assert rm.schoenfeld_events(1.5) == 48


def test_prior_sd_reference() -> None:
    assert rm.prior_sd_for_multiplier(2.0, 0.95) == pytest.approx(0.3536530, rel=1e-6)


def test_action_threshold_reference() -> None:
    assert rm.action_threshold(1.0, 99.0) == pytest.approx(0.01)


def test_shrink_mean_reference() -> None:
    assert rm.shrink_mean(3, 2.0, 5.0, 1.0) == pytest.approx(1.375)
    assert rm.shrink_mean(0, 2.0, 5.0, 1.0) == pytest.approx(1.0)


def test_beta_calibration_identity() -> None:
    p = np.array([0.001, 0.02, 0.3, 0.9])
    assert np.allclose(rm.beta_calibrate(p, 1.0, 1.0, 0.0), p)


def test_platt_identity() -> None:
    p = np.array([0.001, 0.02, 0.3, 0.9])
    assert np.allclose(rm.platt_calibrate(p, 0.0, 1.0), p)


def test_fisher_weight_small_mu_limit() -> None:
    mu = np.array([1e-6, 1e-5])
    assert np.allclose(rm.fisher_weight(mu), mu, rtol=1e-4)


def test_band_index_boundaries() -> None:
    assert [rm.band_index(t, 1) for t in (0, 14.99, 15, 29.9, 30, 44.9, 45, 47)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert [rm.band_index(t, 2) for t in (45, 59.9, 60, 74.9, 75, 89.9, 90, 95)] == [
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
    ]
    assert rm.band_index(100, 3) == 7
    with pytest.raises(ValueError):
        rm.band_index(10, 5)


def test_time_profile_normalised_to_mean_one_over_regular_bands() -> None:
    g = rm.normalise_time_profile([0.1, -0.2, 0.3, 0.5, 0.0, 0.2, 0.4, 0.6])
    assert float(np.mean(g[list(rm.REGULAR_BANDS)])) == pytest.approx(1.0)


def test_rest_of_match_equals_full_match_at_kickoff_with_flat_profile() -> None:
    lam90 = rm.weekly_to_match_cumhaz(0.0245, 0.6)
    p = rm.projected_probability(lam90, rm.flat_time_profile(), 1.0, 0.0, 1, None, 0.0, 0.0)
    assert p == pytest.approx(rm.prob_from_cumhaz(lam90), rel=1e-9)


def test_p15_scales_with_relative_risk() -> None:
    lam90 = rm.weekly_to_match_cumhaz(0.0245, 0.6)
    g = rm.flat_time_profile()
    p1 = rm.projected_probability(lam90, g, 1.0, 60.0, 2, 15)
    p2 = rm.projected_probability(lam90, g, 2.0, 60.0, 2, 15)
    assert rm.cumhaz_from_prob(p2) == pytest.approx(2 * rm.cumhaz_from_prob(p1), rel=1e-9)


def test_first_half_call_continues_into_second_half() -> None:
    sched = rm.minute_schedule(44.0, 1, None, stoppage_1h_min=2.0, stoppage_2h_min=3.0)
    assert sched[0] == (44.0, 1)
    assert (45.0, 2) in sched
    assert sched[-1] == (92.0, 2)


def test_relative_risk_bounds_and_shapes() -> None:
    assert rm.relative_risk([0.3, -0.2], [1.0, 1.0]) == pytest.approx(math.exp(0.1))
    assert rm.relative_risk([5.0], [4.0], rr_max=4.0) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        rm.relative_risk([0.3], [1.0, 2.0])


def test_shrink_feature_handles_missing_and_clips() -> None:
    out = rm.shrink_feature([np.nan, 10.0, -1.0], [0.5, 0.5, 1.0])
    assert np.allclose(out, [0.0, 2.0, -1.0])


@given(st.floats(min_value=0.0, max_value=0.99))
def test_prob_cumhaz_roundtrip(p: float) -> None:
    assert rm.prob_from_cumhaz(rm.cumhaz_from_prob(p)) == pytest.approx(p, abs=1e-12)


@given(st.floats(min_value=-12.0, max_value=2.0))
def test_cloglog_roundtrip(eta: float) -> None:
    p = float(rm.inv_cloglog(eta))
    if 1e-300 < p < 1.0:
        assert float(rm.cloglog(p)) == pytest.approx(eta, abs=1e-6)


def test_score_matches_numerical_derivative() -> None:
    eta = np.array([-8.0, -5.0, -2.0, 0.0])
    for y in (0.0, 1.0):
        yv = np.full_like(eta, y)
        h = 1e-6
        num = (rm.cloglog_loglik_terms(eta + h, yv) - rm.cloglog_loglik_terms(eta - h, yv)) / (
            2 * h
        )
        assert np.allclose(rm.cloglog_score_terms(eta, yv), num, rtol=1e-5, atol=1e-9)


# --- norms, interval audit, evaluation and research helpers (docs/02_MATH.md §3, §5, §9, §14) ---


def test_recency_weights_and_kish_n() -> None:
    w = rm.recency_weights([0, 10, 20], half_life_matches=10.0)
    np.testing.assert_allclose(w, [1.0, 0.5, 0.25])
    assert rm.kish_effective_n([1.0, 1.0, 1.0]) == pytest.approx(3.0)
    assert rm.kish_effective_n(w) == pytest.approx(1.75**2 / 1.3125)
    assert rm.kish_effective_n([0.0, 0.0]) == 0.0


def test_weighted_mean_sd_equal_weights_is_sample_sd() -> None:
    x = [1.0, 2.0, 4.0, 7.0]
    m, s = rm.weighted_mean_sd(x, [1.0, 1.0, 1.0, 1.0])
    assert m == pytest.approx(3.5)
    assert s == pytest.approx(float(np.std(x, ddof=1)))
    _, s1 = rm.weighted_mean_sd([3.0, 5.0], [1.0, 0.0])
    assert math.isnan(s1)


def test_shrink_accepts_effective_n() -> None:
    assert rm.shrink_mean(1.75, 2.0, 5.0, 1.0) == pytest.approx((1.75 * 2.0 + 5.0) / 6.75)


def test_log_cumhaz_sd_from_interval() -> None:
    sd = rm.log_cumhaz_sd_from_interval(0.01, 0.04, level=0.90)
    expected = (math.log(-math.log1p(-0.04)) - math.log(-math.log1p(-0.01))) / (2 * 1.6448536)
    assert sd == pytest.approx(expected, rel=1e-6)
    assert sd == pytest.approx(0.42606, abs=1e-5)
    assert rm.log_cumhaz_sd_from_interval(0.02, 0.02) == 0.0


def test_importance_ess_and_weighted_quantile() -> None:
    assert rm.importance_ess(np.zeros(100)) == pytest.approx(100.0)
    assert rm.importance_ess(np.array([0.0, -1e6, -1e6])) == pytest.approx(1.0)
    x = np.arange(1.0, 6.0)
    assert rm.weighted_quantile(x, np.ones(5), 0.5) == pytest.approx(3.0)
    assert rm.weighted_quantile(x, np.array([0, 0, 0, 0, 1.0]), 0.1) == pytest.approx(5.0)


def test_holm_adjust() -> None:
    adj = rm.holm_adjust([0.01, 0.04, 0.03, 0.20])
    np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.20])


def test_decayed_event_load_respects_blanking_gap() -> None:
    # events at 100 s and 590 s; as_of 660 s, delta 60 s -> cut 600 s
    v = rm.decayed_event_load([100.0, 590.0, 650.0], as_of_s=660.0, delta_s=60.0, tau_s=300.0)
    assert v == pytest.approx(math.exp(-500 / 300) + math.exp(-10 / 300))
