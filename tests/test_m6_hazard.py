from __future__ import annotations

import numpy as np
import pytest

from livrisk.model.live_hazard import Priors, build_design, fit_map
from livrisk.synth.generator import SynthTruth, generate_person_minutes
from tests.conftest import DEV_SEEDS

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m6]

TRUTH = SynthTruth(p7_mean=0.10)
N_PLAYER_MATCHES = 7000  # ~400 events at p7_mean = 0.10


def _fit(seed: int, truth: SynthTruth):  # type: ignore[no-untyped-def]
    df = generate_person_minutes(
        N_PLAYER_MATCHES, seed=seed, truth=truth, informative_censoring=False
    )
    design = build_design(df, truth.feature_names, truth.match_share)
    return fit_map(design, Priors(), config_hash="test", version="test")


def test_parameter_recovery_coverage() -> None:
    inside = np.zeros(len(TRUTH.beta))
    seeds = DEV_SEEDS
    for s in seeds:
        f = _fit(s, TRUTH)
        k0 = len(f.param_names) - len(f.beta)
        sd = np.sqrt(np.diag(np.asarray(f.cov))[k0:])
        lo, hi = np.asarray(f.beta) - 1.645 * sd, np.asarray(f.beta) + 1.645 * sd
        inside += (lo <= np.asarray(TRUTH.beta)) & (np.asarray(TRUTH.beta) <= hi)
    cover = inside / len(seeds)
    assert np.all((cover >= 0.85) & (cover <= 0.95)), cover


def test_null_signal_shrinks_to_zero() -> None:
    null = SynthTruth(p7_mean=0.10, beta=(0.0,) * 6)
    betas = np.array([_fit(s, null).beta for s in DEV_SEEDS[:20]])
    assert np.all(np.abs(betas.mean(axis=0)) < 0.1)


def test_time_profile_recovery() -> None:
    from livrisk import riskmath as rm

    true_g = rm.normalise_time_profile(TRUTH.theta)
    gs = np.array([_fit(s, TRUTH).g_levels() for s in DEV_SEEDS[:10]])
    med = np.median(gs, axis=0)
    regular = list(rm.REGULAR_BANDS)
    assert np.all(np.abs(med[regular] / true_g[regular] - 1) < 0.2)


def test_interval_audit_passes_when_correct_and_flags_shrunk_covariance() -> None:
    # TODO: interval_audit on a correctly specified fit passes (khat < 0.7, quantiles within 10 %);
    # the same audit with the Laplace covariance scaled by 0.25 fails.
    pytest.fail("implement: importance-sampling audit")
