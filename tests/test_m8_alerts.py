"""Acceptance tests for M8: alert rules (docs/06_UI_SPEC.md §3) and operating point (§7.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from livrisk.alerts.analysis import (
    estimate_rr_level_matrix,
    operating_point,
    replay_agreement_rel,
    rr_level,
)
from livrisk.alerts.logic import AlertLogic
from livrisk.config import Config, load_config
from livrisk.contracts import (
    Action,
    ActorRole,
    Alert,
    AlertLevel,
    DecisionRecord,
    Mode,
    PrematchBand,
    Quality,
    RiskSnapshot,
)

pytestmark = [pytest.mark.milestone_m8]

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "default.yaml")
MATCH_ID = "m-alerts-test"
ABOVE = 0.010  # r_rest above the 0.004 floor (alerts.check.r_rest_floor)
BELOW = 0.001  # r_rest under the floor
N_UPDATES = CFG.alerts.analysis.updates_per_match
N_LEVELS = len(CFG.alerts.analysis.rr_level_edges) + 1

#: One representative RR per RR level of alerts.analysis.rr_level_edges (docs/02 §7.1).
LEVEL_RR = (1.0, 1.4, 1.7, 2.5)

#: Per-update RR level transition matrix used for analysis-vs-simulation agreement.
LEVEL_MATRIX = np.asarray(
    [
        [0.980, 0.012, 0.006, 0.002],
        [0.750, 0.120, 0.100, 0.030],
        [0.450, 0.150, 0.300, 0.100],
        [0.300, 0.100, 0.200, 0.400],
    ]
)
INIT = np.asarray([0.97, 0.015, 0.010, 0.005])
R_REST_OK = 0.8


def _snap(
    pid: str,
    rr: float,
    t_s: float,
    r_rest: float = ABOVE,
    quality: Quality = Quality.GOOD,
    mode: Mode = Mode.RISK,
) -> RiskSnapshot:
    """A risk row with consistent intervals; only the alert-relevant fields vary."""
    return RiskSnapshot(
        match_id=MATCH_ID,
        player_id=pid,
        period=1 if t_s < 2700.0 else 2,
        as_of_t_s=t_s,
        data_age_s=5.0,
        quality=quality,
        mode=mode,
        p15=r_rest / 2.0,
        p15_lo=r_rest / 2.0,
        p15_hi=r_rest / 2.0,
        r_rest=r_rest,
        r_rest_lo=r_rest,
        r_rest_hi=r_rest,
        r_usual=0.010,
        rr=rr,
        prematch_band=PrematchBand.LOW,
        top_reason_code="HSR_HIGH",
        model_version="test-v0",
    )


@dataclass(frozen=True)
class Row:
    """One player's inputs at one update of a rule scenario."""

    pid: str
    rr: float
    r_rest: float = ABOVE
    quality: Quality = Quality.GOOD


@dataclass(frozen=True)
class Step:
    """One risk update of a scenario: rows, players leaving the pitch, expected alerts.

    ``expect`` holds (kind, player_id, level) with kind "opened" | "upgraded" | "cleared" in
    the return order of ``AlertLogic.update`` (opened by priority, then upgraded, then cleared).
    """

    rows: tuple[Row, ...]
    expect: tuple[tuple[str, str, AlertLevel], ...] = ()
    off_pitch: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    steps: tuple[Step, ...]
    budget_used: int
    open_levels: tuple[tuple[str, AlertLevel], ...] = ()
    muted: tuple[str, ...] = ()
    monitor_only: bool = False


def _kind(alert: Alert, t_s: float) -> str:
    """Event kind of a returned alert: clearing, upgrade (opened earlier) or opening."""
    if alert.cleared_t_s is not None:
        return "cleared"
    if alert.level == AlertLevel.DISCUSS and alert.opened_t_s < t_s:
        return "upgraded"
    return "opened"


def _run(scenario: Scenario) -> tuple[list[list[Alert]], AlertLogic]:
    """Play a scenario through a fresh AlertLogic; returns per-update alerts and the logic."""
    cfg = (
        CFG.model_copy(update={"engine": CFG.engine.model_copy(update={"mode": "monitor_only"})})
        if scenario.monitor_only
        else CFG
    )
    logic = AlertLogic(cfg, MATCH_ID)
    on_pitch: set[str] = set()
    out: list[list[Alert]] = []
    for i, step in enumerate(scenario.steps):
        t_s = 60.0 * (i + 1)
        on_pitch |= {r.pid for r in step.rows}
        on_pitch -= set(step.off_pitch)
        mode = Mode.MONITOR_ONLY if scenario.monitor_only else Mode.RISK
        snaps = [
            _snap(r.pid, r.rr, t_s, r.r_rest, r.quality, mode)
            for r in step.rows
            if r.pid in on_pitch
        ]
        out.append(logic.update(snaps, on_pitch))
    return out, logic


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

SCENARIOS: dict[str, Scenario] = {
    CASES[0]: Scenario(
        steps=(
            Step((Row("P1", 1.6),)),
            Step((Row("P1", 1.6),), (("opened", "P1", AlertLevel.CHECK),)),
            Step((Row("P1", 1.6),)),
            Step((Row("P1", 1.0),)),
            Step((Row("P1", 1.0),), (("cleared", "P1", AlertLevel.CHECK),)),
        ),
        budget_used=1,
    ),
    CASES[1]: Scenario(
        steps=(
            Step((Row("P1", 1.6),)),
            Step((Row("P1", 1.4),)),
            Step((Row("P1", 1.6),)),
            Step((Row("P1", 1.0),)),
        ),
        budget_used=0,
    ),
    CASES[2]: Scenario(
        steps=(
            Step((Row("P1", 1.6, BELOW),)),
            Step((Row("P1", 1.6, BELOW),)),
            # The floor gates the opening update, not the streak: once r_rest clears the
            # floor with the streak intact, the alert opens.
            Step((Row("P1", 1.6, ABOVE),), (("opened", "P1", AlertLevel.CHECK),)),
        ),
        budget_used=1,
        open_levels=(("P1", AlertLevel.CHECK),),
    ),
    CASES[3]: Scenario(
        steps=(
            Step(
                (
                    Row("P1", 1.6, ABOVE, Quality.PREMATCH_ONLY),
                    Row("P2", 1.7, ABOVE, Quality.DEGRADED),
                )
            ),
            Step(
                (
                    Row("P1", 1.6, ABOVE, Quality.PREMATCH_ONLY),
                    Row("P2", 1.7, ABOVE, Quality.DEGRADED),
                ),
                (("opened", "P2", AlertLevel.CHECK),),
            ),
            Step((Row("P1", 1.6, ABOVE, Quality.STALE), Row("P2", 1.7, ABOVE, Quality.DEGRADED))),
            # One good update after prematch_only/stale updates is not persistence.
            Step((Row("P1", 1.6), Row("P2", 1.0))),
            Step(
                (Row("P1", 1.6), Row("P2", 1.0)),
                (("opened", "P1", AlertLevel.CHECK), ("cleared", "P2", AlertLevel.CHECK)),
            ),
        ),
        budget_used=2,
        open_levels=(("P1", AlertLevel.CHECK),),
    ),
    CASES[4]: Scenario(
        steps=(
            Step((Row("P1", 1.6),)),
            Step((Row("P1", 1.6),), (("opened", "P1", AlertLevel.CHECK),)),
            Step((Row("P1", 1.4),)),
            Step((Row("P1", 1.4),)),
            Step((Row("P1", 1.2),)),
            Step((Row("P1", 1.2),), (("cleared", "P1", AlertLevel.CHECK),)),
            Step((Row("P1", 1.4),)),
            Step((Row("P1", 1.6),)),
            Step((Row("P1", 1.6),), (("opened", "P1", AlertLevel.CHECK),)),
        ),
        budget_used=2,
        open_levels=(("P1", AlertLevel.CHECK),),
    ),
    CASES[5]: Scenario(
        steps=(
            Step((Row("P1", 2.2),)),
            Step((Row("P1", 2.2),), (("opened", "P1", AlertLevel.CHECK),)),
            Step((Row("P1", 2.2),), (("upgraded", "P1", AlertLevel.DISCUSS),)),
            Step((Row("P1", 2.2),)),
        ),
        budget_used=1,
        open_levels=(("P1", AlertLevel.DISCUSS),),
    ),
    CASES[6]: Scenario(
        steps=(
            Step((Row("P1", 1.6, 0.010), Row("P2", 1.7, 0.020), Row("P3", 1.4, 0.015))),
            Step(
                (Row("P1", 1.6, 0.010), Row("P2", 1.7, 0.020), Row("P3", 1.6, 0.015)),
                (("opened", "P2", AlertLevel.CHECK), ("opened", "P1", AlertLevel.CHECK)),
            ),
            Step((Row("P1", 1.6, 0.010), Row("P2", 1.7, 0.020), Row("P3", 1.6, 0.015))),
            Step((Row("P1", 1.6, 0.010), Row("P2", 1.7, 0.020), Row("P3", 1.6, 0.015))),
        ),
        budget_used=2,
        open_levels=(("P2", AlertLevel.CHECK), ("P1", AlertLevel.CHECK)),
        muted=("P3",),
    ),
    CASES[7]: Scenario(
        steps=(
            Step((Row("P0", 1.6, 0.012),)),
            Step(
                (Row("P0", 1.6, 0.012), Row("P1", 1.6, 0.010), Row("P2", 1.6, 0.020)),
                (("opened", "P0", AlertLevel.CHECK),),
            ),
            Step(
                (Row("P0", 1.6, 0.012), Row("P1", 1.6, 0.010), Row("P2", 1.6, 0.020)),
                (("opened", "P2", AlertLevel.CHECK),),
            ),
            Step((Row("P0", 1.6, 0.012), Row("P1", 1.6, 0.010), Row("P2", 1.6, 0.020))),
        ),
        budget_used=2,
        open_levels=(("P0", AlertLevel.CHECK), ("P2", AlertLevel.CHECK)),
        muted=("P1",),
    ),
    CASES[8]: Scenario(
        steps=(
            Step((Row("P1", 1.6), Row("P2", 1.0))),
            Step((Row("P1", 1.6), Row("P2", 1.0)), (("opened", "P1", AlertLevel.CHECK),)),
            Step((Row("P2", 1.0),), (("cleared", "P1", AlertLevel.CHECK),), ("P1",)),
            Step((Row("P2", 1.0),)),
        ),
        budget_used=1,
    ),
    CASES[9]: Scenario(
        steps=(
            # monitor_only rows always report rr == 1 (the RiskSnapshot contract), and the
            # state machine additionally refuses to open or upgrade in this mode.
            Step((Row("P1", 1.0),)),
            Step((Row("P1", 1.0),)),
            Step((Row("P1", 1.0),)),
            Step((Row("P1", 1.0),)),
        ),
        budget_used=0,
        monitor_only=True,
    ),
    CASES[10]: Scenario(
        steps=(
            Step((Row("P1", 2.2, 0.020), Row("P2", 1.6, 0.010), Row("P3", 1.6, 0.005))),
            Step(
                (Row("P1", 2.2, 0.020), Row("P2", 1.6, 0.010), Row("P3", 1.6, 0.005)),
                (("opened", "P1", AlertLevel.CHECK), ("opened", "P2", AlertLevel.CHECK)),
            ),
            Step(
                (Row("P1", 2.2, 0.020), Row("P2", 1.0, 0.010), Row("P3", 1.6, 0.005)),
                (("upgraded", "P1", AlertLevel.DISCUSS),),
            ),
            Step(
                (Row("P1", 2.2, 0.020), Row("P2", 1.0, 0.010), Row("P3", 1.6, 0.005)),
                (("cleared", "P2", AlertLevel.CHECK),),
            ),
            Step((Row("P1", 2.2, 0.020), Row("P2", 1.0, 0.010), Row("P3", 1.6, 0.005))),
        ),
        budget_used=2,
        open_levels=(("P1", AlertLevel.DISCUSS),),
        muted=("P3",),
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_alert_rules(case: str) -> None:
    scenario = SCENARIOS[case]
    first, logic = _run(scenario)
    for i, (alerts, step) in enumerate(zip(first, scenario.steps, strict=True)):
        t_s = 60.0 * (i + 1)
        got = tuple((_kind(a, t_s), a.player_id, a.level) for a in alerts)
        assert got == step.expect, f"{case}, update {i + 1}"
        for alert in alerts:
            if alert.cleared_t_s is not None:
                assert alert.cleared_t_s == t_s
            elif _kind(alert, t_s) == "opened":
                assert alert.opened_t_s == t_s
                assert alert.budget_used_after <= CFG.alerts.budget_per_match
    # Alert identity is stable across upgrade and clearing; a re-opening gets a new id.
    open_ids: dict[str, str] = {}
    for i, alerts in enumerate(first):
        for alert in alerts:
            if _kind(alert, 60.0 * (i + 1)) == "opened":
                open_ids[alert.player_id] = alert.alert_id
            else:
                assert open_ids.get(alert.player_id) == alert.alert_id
    assert logic.budget_used == scenario.budget_used
    assert tuple((a.player_id, a.level) for a in logic.open_alerts()) == scenario.open_levels
    assert logic.muted_qualifiers() == scenario.muted
    second, _ = _run(scenario)
    assert first == second, f"{case}: replaying the same inputs gave different alerts"


def test_tie_on_r_rest_broken_by_player_id() -> None:
    """Two qualifiers with equal r_rest and one slot left: the lower player_id wins."""
    for order in (("P7", "P8"), ("P8", "P7")):
        rows = [Row(pid, 1.6, 0.010) for pid in order]
        scenario = Scenario(
            steps=(
                Step((Row("P9", 1.6),)),
                Step((Row("P9", 1.6), *rows), (("opened", "P9", AlertLevel.CHECK),)),
                Step((Row("P9", 1.6), *rows), (("opened", "P7", AlertLevel.CHECK),)),
            ),
            budget_used=2,
            muted=("P8",),
        )
        first, logic = _run(scenario)
        assert [a.player_id for a in first[-1]] == ["P7"]
        assert logic.muted_qualifiers() == ("P8",)


def test_team_wide_reason_never_opens_a_card() -> None:
    """D-030: updates led by a team-wide event (DISRUPT) open no card and spend no budget, but
    the streak keeps counting, so the first player-specific update can open at once."""
    logic = AlertLogic(CFG, MATCH_ID)
    disrupt = [
        _snap("P1", 1.8, t).model_copy(update={"top_reason_code": "DISRUPT"})
        for t in (60.0, 120.0, 180.0)
    ]
    for row in disrupt:
        assert logic.update([row], on_pitch={"P1"}) == []
    assert logic.budget_used == 0
    opened = logic.update([_snap("P1", 1.8, 240.0)], on_pitch={"P1"})
    assert [(a.player_id, a.level, a.reason_code) for a in opened] == [
        ("P1", AlertLevel.CHECK, "HSR_HIGH")
    ]
    assert logic.budget_used == 1


def test_direct_discuss_opening_when_floor_blocked_check() -> None:
    """RR >= 2.0 for 3 updates with the floor met only at the third: a Discuss opens directly."""
    scenario = Scenario(
        steps=(
            Step((Row("P1", 2.2, BELOW),)),
            Step((Row("P1", 2.2, BELOW),)),
            Step((Row("P1", 2.2, ABOVE),), (("opened", "P1", AlertLevel.DISCUSS),)),
        ),
        budget_used=1,
        open_levels=(("P1", AlertLevel.DISCUSS),),
    )
    first, logic = _run(scenario)
    assert first[-1][0].budget_used_after == 1
    assert logic.budget_used == 1


def test_staff_concern_flag_upgrades_open_check() -> None:
    logic = AlertLogic(CFG, MATCH_ID)
    on_pitch = {"P1"}
    logic.update([_snap("P1", 1.6, 60.0)], on_pitch)
    opened = logic.update([_snap("P1", 1.6, 120.0)], on_pitch)
    assert len(opened) == 1 and opened[0].level == AlertLevel.CHECK
    record = DecisionRecord(
        alert_id=opened[0].alert_id,
        t_s=130.0,
        actor_role=ActorRole.PHYSIO,
        action=Action.OTHER,
        note="concern after physio look",
    )
    logic.record_decision(record)
    assert logic.budget_used == 1
    emitted = logic.update([_snap("P1", 1.4, 180.0)], on_pitch)
    assert len(emitted) == 1
    assert emitted[0].level == AlertLevel.DISCUSS
    assert emitted[0].alert_id == opened[0].alert_id
    assert emitted[0].cleared_t_s is None
    assert logic.budget_used == 1
    assert logic.decisions == (record,)


def test_decision_without_concern_or_matching_alert_changes_nothing() -> None:
    logic = AlertLogic(CFG, MATCH_ID)
    on_pitch = {"P1"}
    logic.update([_snap("P1", 1.6, 60.0)], on_pitch)
    opened = logic.update([_snap("P1", 1.6, 120.0)], on_pitch)
    logic.record_decision(
        DecisionRecord(
            alert_id=opened[0].alert_id,
            t_s=130.0,
            actor_role=ActorRole.ANALYST,
            action=Action.PHYSIO_CHECK,
        )
    )
    logic.record_decision(
        DecisionRecord(
            alert_id="A99",
            t_s=131.0,
            actor_role=ActorRole.PHYSIO,
            action=Action.OTHER,
            note="concern",
        )
    )
    assert logic.update([_snap("P1", 1.6, 180.0)], on_pitch) == []
    assert logic.open_alerts()[0].level == AlertLevel.CHECK
    assert len(logic.decisions) == 2


def test_update_rejects_rows_of_another_match() -> None:
    logic = AlertLogic(CFG, MATCH_ID)
    row = _snap("P1", 1.6, 60.0).model_copy(update={"match_id": "other-match"})
    with pytest.raises(ValueError, match="other-match"):
        logic.update([row], {"P1"})


def test_rr_level_bins() -> None:
    edges = (1.3, 1.5, 2.0)
    assert [rr_level(x, edges) for x in (0.5, 1.29, 1.3, 1.49, 1.5, 1.99, 2.0, 3.9)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    with pytest.raises(ValueError, match="positive"):
        rr_level(0.0, edges)
    with pytest.raises(ValueError, match="ascending"):
        rr_level(1.0, (1.5, 1.3))


def test_estimate_rr_level_matrix_counts_within_paths_only() -> None:
    edges = (1.3, 1.5, 2.0)
    paths = [(2.5, 2.5, 1.7, 1.7), (1.4, 1.4, 1.0, 1.0)]
    mat = estimate_rr_level_matrix(paths, edges)
    # No stitched transition between the end of one path and the start of the next.
    assert mat[2, 1] == 0.0 and mat[2, 2] == 1.0
    assert mat[3, 3] == 0.5 and mat[3, 2] == 0.5
    assert mat[1, 1] == 0.5 and mat[1, 0] == 0.5
    assert mat[0, 0] == 1.0
    with pytest.raises(ValueError, match="no transitions"):
        estimate_rr_level_matrix([(1.0, 1.0, 1.4, 1.4)], edges)
    with pytest.raises(ValueError, match="positive"):
        estimate_rr_level_matrix([(-1.0, 1.0)], edges)


def test_estimate_rr_level_matrix_recovers_truth() -> None:
    rng = np.random.default_rng(CFG.seeds.default)
    cum = np.cumsum(LEVEL_MATRIX, axis=1)
    n_paths, length = 400, N_UPDATES
    paths = np.empty((n_paths, length), dtype=np.int64)
    paths[:, 0] = (rng.random(n_paths)[:, None] > np.cumsum(INIT)).sum(axis=1)
    for u in range(1, length):
        paths[:, u] = (rng.random(n_paths)[:, None] > cum[paths[:, u - 1]]).sum(axis=1)
    level_rr = np.asarray(LEVEL_RR)
    estimated = estimate_rr_level_matrix(
        level_rr[paths].tolist(), CFG.alerts.analysis.rr_level_edges
    )
    assert np.abs(estimated - LEVEL_MATRIX).max() < 0.10
    estimated_again = estimate_rr_level_matrix(
        level_rr[paths].tolist(), CFG.alerts.analysis.rr_level_edges
    )
    np.testing.assert_array_equal(estimated, estimated_again)


def _simulate_alert_runs(
    level_matrix: NDArray[np.float64],
    init: NDArray[np.float64],
    r_rest_ok: float,
    n_matches: int,
    cfg: Config,
    seed: int,
) -> tuple[float, float, float, NDArray[np.float64]]:
    """Mean openings, upgrades and flaps per team-match plus budget-exhaustion rates.

    Plays ``n_matches`` team-matches of ``players_per_team`` independent RR level paths through
    a budget-unconstrained ``AlertLogic``; the R_rest floor passes independently with
    probability ``r_rest_ok`` per update, matching the analysis assumptions.
    """
    players = cfg.alerts.analysis.players_per_team
    updates = cfg.alerts.analysis.updates_per_match
    big_budget = cfg.model_copy(
        update={"alerts": cfg.alerts.model_copy(update={"budget_per_match": 10**6})}
    )
    cum = np.cumsum(level_matrix, axis=1)
    cum_init = np.cumsum(init)
    rng = np.random.default_rng(seed)
    pool = {
        (j, level, gate): _snap(f"P{j:03d}", LEVEL_RR[level], 60.0, ABOVE if gate else BELOW)
        for j in range(players)
        for level in range(N_LEVELS)
        for gate in (True, False)
    }
    opens = np.zeros(n_matches)
    upgrades = np.zeros(n_matches)
    flaps = np.zeros(n_matches)
    exhausted = np.zeros((n_matches, updates + 1))
    for m in range(n_matches):
        paths = np.empty((players, updates), dtype=np.int64)
        paths[:, 0] = (rng.random(players)[:, None] > cum_init).sum(axis=1)
        for u in range(1, updates):
            paths[:, u] = (rng.random(players)[:, None] > cum[paths[:, u - 1]]).sum(axis=1)
        gates = rng.random((players, updates)) < r_rest_ok
        logic = AlertLogic(big_budget, MATCH_ID)
        on_pitch = {f"P{j:03d}" for j in range(players)}
        open_ids: set[str] = set()
        n_open = n_upgrade = n_flap = cum_open = 0
        for u in range(updates):
            rows = [pool[(j, int(paths[j, u]), bool(gates[j, u]))] for j in range(players)]
            for alert in logic.update(rows, on_pitch):
                if alert.cleared_t_s is not None:
                    open_ids.discard(alert.alert_id)
                    n_flap += 1
                elif alert.alert_id in open_ids:
                    n_upgrade += 1
                else:
                    open_ids.add(alert.alert_id)
                    n_open += 1
                    n_flap += 1
                    cum_open += 1
            exhausted[m, u + 1] = cum_open >= cfg.alerts.budget_per_match
        opens[m] = n_open
        upgrades[m] = n_upgrade
        flaps[m] = n_flap
    return float(opens.mean()), float(upgrades.mean()), float(flaps.mean()), exhausted.mean(axis=0)


def test_operating_point_matches_simulation() -> None:
    op = operating_point(LEVEL_MATRIX, INIT, R_REST_OK, CFG)
    mean_open, mean_upgrade, mean_flap, exhausted = _simulate_alert_runs(
        LEVEL_MATRIX, INIT, R_REST_OK, 2000, CFG, CFG.seeds.default
    )
    tol = replay_agreement_rel(CFG)
    for name, analytic, simulated in (
        ("expected_checks", op.expected_checks, mean_open),
        ("expected_discuss_upgrades", op.expected_discuss_upgrades, mean_upgrade),
        ("expected_flaps", op.expected_flaps, mean_flap),
    ):
        assert analytic > 0.2, f"{name}: degenerate operating point"
        assert abs(simulated - analytic) / analytic <= tol, (
            f"{name}: analysis {analytic:.4f} vs simulation {simulated:.4f}"
        )
    for u in (60, 90):
        assert abs(exhausted[u] - op.p_budget_exhausted[u]) <= 0.05, (
            f"p_budget_exhausted[{u}]: analysis {op.p_budget_exhausted[u]:.4f} "
            f"vs simulation {exhausted[u]:.4f}"
        )


def test_operating_point_quiet_matrix_stays_quiet() -> None:
    quiet = np.zeros((N_LEVELS, N_LEVELS))
    quiet[:, 0] = 1.0
    init = np.zeros(N_LEVELS)
    init[0] = 1.0
    op = operating_point(quiet, init, 1.0, CFG)
    assert op.expected_checks == 0.0
    assert op.expected_discuss_upgrades == 0.0
    assert op.expected_flaps == 0.0
    assert all(p == 0.0 for p in op.p_budget_exhausted)
    assert len(op.p_budget_exhausted) == N_UPDATES + 1


def test_operating_point_requires_thresholds_on_the_level_grid() -> None:
    off_grid = CFG.model_copy(
        update={
            "alerts": CFG.alerts.model_copy(
                update={"check": CFG.alerts.check.model_copy(update={"rr_min": 1.4})}
            )
        }
    )
    with pytest.raises(ValueError, match="rr_level_edges"):
        operating_point(LEVEL_MATRIX, INIT, R_REST_OK, off_grid)


def test_replay_agreement_rel_comes_from_config() -> None:
    assert replay_agreement_rel(CFG) == CFG.alerts.analysis.replay_agreement_rel == 0.10
