from __future__ import annotations

import pytest

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m7]


def test_beta_zero_flat_profile_equals_riskmath() -> None:
    # TODO: FittedHazard with beta = 0 and theta giving g = 1; RiskEngine.update at 60:00 must give
    # r_rest == riskmath.projected_probability(Lambda, flat, 1.0, 60.0, 2, None) exactly.
    pytest.fail("implement")


def test_quality_rules() -> None:
    # low coverage -> z* = 0 and DEGRADED; no valid feature -> PREMATCH_ONLY; age > 180 s -> STALE
    pytest.fail("implement")


def test_intervals_contain_point_and_widen_with_cov() -> None:
    pytest.fail("implement")


def test_monitor_only_mode() -> None:
    # rr == 1 and r_rest == r_usual for every player
    pytest.fail("implement")


def test_update_speed_30_players() -> None:
    # 30 players, 1,000 draws each: <= 500 ms (generous tolerance on CI machines)
    pytest.fail("implement")


def test_prematch_interval_widens_displayed_interval() -> None:
    # TODO: same player with and without p7_lo/p7_hi: point estimates identical, the interval with
    # the p7 interval strictly wider; sd from riskmath.log_cumhaz_sd_from_interval.
    pytest.fail("implement: p7 uncertainty propagation")
