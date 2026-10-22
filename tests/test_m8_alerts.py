from __future__ import annotations

import pytest

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m8]

# TODO: write table-driven cases. Each case = sequence of
# (rr, r_rest, quality, on_pitch) per update for one or more players,
# and the expected alerts after each update. Cover at least:
CASES = [
    "check opens after 2 consecutive updates with rr >= 1.5 and r_rest >= floor",
    "check does not open with only 1 update above threshold",
    "check does not open when r_rest < floor",
    "check does not open when quality is prematch_only or stale",
    "hysteresis: stays open at rr = 1.4, clears after 2 updates below 1.3",
    "upgrade check -> discuss after 3 updates with rr >= 2.0 without consuming budget",
    "budget of 2 exhausted: third qualifying player gets no alert",
    "two players qualify in the same update: higher r_rest wins the last budget slot",
    "substitution clears the player's open alert",
    "monitor_only mode: no check/discuss alerts",
    "determinism: same inputs -> identical alerts",
]


@pytest.mark.parametrize("case", CASES)
def test_alert_rules(case: str) -> None:
    pytest.fail(f"implement: {case}")


def test_operating_point_matches_simulation() -> None:
    # TODO: draw RR level paths from a known level matrix, run AlertLogic on them, and compare
    # expected Check openings and flaps per team-match with alerts.analysis.operating_point
    # (relative difference <= alerts.analysis.replay_agreement_rel).
    pytest.fail("implement: analysis vs simulation")
