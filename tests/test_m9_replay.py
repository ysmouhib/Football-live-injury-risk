from __future__ import annotations

import pytest

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m9]


def test_replay_equals_offline() -> None:
    pytest.fail("implement: replay output == offline computation for same inputs")


def test_leaky_feature_raises_lookahead_error() -> None:
    # Inject a feature that peeks 30 s ahead; replay must raise replay.simulator.LookAheadError
    pytest.fail("implement")


def test_stale_rules_hold_under_latency_and_dropout() -> None:
    # ReplaySettings(latency_s=30, dropout_rate=0.05, burst_s=30): no displayed value violates
    # docs/06_UI_SPEC.md §4
    pytest.fail("implement")


def test_replay_speed() -> None:
    # A 90-minute synthetic match replays at >= 50x real time
    pytest.fail("implement")
