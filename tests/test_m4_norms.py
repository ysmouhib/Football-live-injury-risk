from __future__ import annotations

import pytest

from livrisk.features.norms import team_relative

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m4]


def test_team_relative_removes_shared_shift() -> None:
    z = {"a": 1.0, "b": 1.0, "c": 1.0, "d": None}
    out = team_relative(z)
    assert out["a"] == pytest.approx(0.0)
    assert out["d"] is None


def test_norms_ignore_future_matches() -> None:
    # TODO: build history with an extreme value on/after as_of_date; assert norms unchanged.
    pytest.fail("implement: future-match leak trap")


def test_new_player_gets_position_norm_and_veteran_gets_own() -> None:
    pytest.fail("implement: shrinkage behaviour with n = 0 and n large")


def test_recency_weighting_and_effective_matches() -> None:
    # TODO: 20 past matches where the last 5 carry a level shift; the shrunk mean moves toward the
    # recent level more than an unweighted mean would, and NormEntry.n_effective equals
    # riskmath.kish_effective_n of the recency weights (not the number of windows).
    pytest.fail("implement: recency weights and Kish n")


def test_spread_uses_non_overlapping_windows() -> None:
    # TODO: feeding every 10-s snapshot instead of 3 windows per band gives the same n_effective.
    pytest.fail("implement: no pseudo-replication")


def test_norm_reset_after_absence() -> None:
    # TODO: days_since_return >= 14 -> reset for the next 3 matches: own norm capped at one
    # effective match, NormEntry.reset is True, quality reason 'norm_reset'.
    pytest.fail("implement: reset rule")
