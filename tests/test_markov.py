from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from livrisk import markov as mk


def two_state(a: float, b: float) -> np.ndarray:
    return np.array([[1.0 - a, a], [b, 1.0 - b]])


def test_validation_rejects_bad_matrices() -> None:
    with pytest.raises(ValueError):
        mk.as_transition_matrix([[0.5, 0.5]])
    with pytest.raises(ValueError):
        mk.as_transition_matrix([[1.2, -0.2], [0.5, 0.5]])
    with pytest.raises(ValueError):
        mk.as_transition_matrix([[0.5, 0.4], [0.5, 0.5]])


def test_two_state_closed_forms() -> None:
    p = two_state(0.1, 0.3)
    assert mk.is_ergodic(p)
    np.testing.assert_allclose(mk.stationary_distribution(p), [0.75, 0.25], atol=1e-12)
    assert mk.slem(p) == pytest.approx(0.6, abs=1e-12)
    # worst-start TV distance = max(a, b) / (a + b) * |1 - a - b|^t = 0.75 * 0.6^t
    assert mk.mixing_time(p, eps=0.25) == math.ceil(math.log(1 / 3) / math.log(0.6))
    assert mk.mixing_time(p, eps=0.25) == 3


def test_periodic_and_reducible_chains() -> None:
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert mk.is_irreducible(flip)
    assert mk.period(flip) == 2
    assert not mk.is_ergodic(flip)
    assert mk.slem(flip) == pytest.approx(1.0)
    np.testing.assert_allclose(mk.stationary_distribution(flip), [0.5, 0.5])
    with pytest.raises(ValueError):
        mk.mixing_time(flip)
    absorbing = np.array([[0.9, 0.1], [0.0, 1.0]])
    assert not mk.is_irreducible(absorbing)
    with pytest.raises(ValueError):
        mk.stationary_distribution(absorbing)
    with pytest.raises(ValueError):
        mk.jump_chain(absorbing)


def test_three_cycle_with_one_shortcut_is_aperiodic() -> None:
    cycle = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert mk.period(cycle) == 3
    shortcut = np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert mk.period(shortcut) == 1


@settings(max_examples=50, deadline=None)
@given(st.integers(min_value=2, max_value=6), st.integers(min_value=0, max_value=10_000))
def test_stationary_is_fixed_point_and_markov_equals_semi_markov(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(n), size=n)
    pi = mk.stationary_distribution(p)
    np.testing.assert_allclose(pi @ p, pi, atol=1e-10)
    # A Markov chain is a semi-Markov process with geometric dwell: both views give the same
    # long-run occupancy.
    jumps, dwell = mk.jump_chain(p)
    np.testing.assert_allclose(mk.semi_markov_occupancy(jumps, dwell), pi, atol=1e-9)


def test_semi_markov_alternating_states() -> None:
    occ = mk.semi_markov_occupancy([[0.0, 1.0], [1.0, 0.0]], [3.0, 1.0])
    np.testing.assert_allclose(occ, [0.75, 0.25])


def test_propagate_homogeneous_and_inhomogeneous() -> None:
    p = two_state(0.1, 0.3)
    path = mk.propagate([1.0, 0.0], p, steps=200)
    assert path.shape == (201, 2)
    np.testing.assert_allclose(path[-1], [0.75, 0.25], atol=1e-12)
    q = two_state(0.5, 0.5)
    seq = mk.propagate([1.0, 0.0], [p, q])
    np.testing.assert_allclose(seq[2], np.array([1.0, 0.0]) @ p @ q)
    with pytest.raises(ValueError):
        mk.propagate([1.0, 0.0], [p, q], steps=2)


def test_empirical_transition_matrix_with_prior() -> None:
    est = mk.empirical_transition_matrix([0, 0, 1, 0, 1, 1], 2)
    np.testing.assert_allclose(est, [[1 / 3, 2 / 3], [1 / 2, 1 / 2]])
    prior = 4.0 * two_state(0.5, 0.5)
    shrunk = mk.empirical_transition_matrix([0, 0, 0], 3 - 1, prior_counts=prior)
    np.testing.assert_allclose(shrunk[0], [(2 + 2) / 6, 2 / 6])
    np.testing.assert_allclose(shrunk[1], [0.5, 0.5])
    with pytest.raises(ValueError):
        mk.empirical_transition_matrix([0, 0, 0], 2)


def test_ar1_effective_n_matches_covariance_formula() -> None:
    assert mk.ar1_effective_n(15, 0.0) == pytest.approx(15.0)
    n, phi = 15, 0.8
    idx = np.arange(n)
    cov = phi ** np.abs(idx[:, None] - idx[None, :])
    assert mk.ar1_effective_n(n, phi) == pytest.approx(n * n / cov.sum(), rel=1e-12)
    assert mk.ar1_effective_n(n, phi) == pytest.approx(2.3338, abs=1e-4)
    assert mk.ar1_effective_n(10_000, phi) == pytest.approx(10_000 * 0.2 / 1.8, rel=1e-3)


def test_tv_distance() -> None:
    assert mk.tv_distance([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.5)
