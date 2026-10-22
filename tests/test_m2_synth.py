from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from livrisk.synth.generator import SynthTruth, generate_match, generate_person_minutes

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m2]


def test_person_minutes_deterministic() -> None:
    a = generate_person_minutes(200, seed=1)
    b = generate_person_minutes(200, seed=1)
    assert a.equals(b)


def test_person_minutes_columns() -> None:
    df = generate_person_minutes(50, seed=2)
    truth = SynthTruth()
    for col in (
        "match_id",
        "player_id",
        "club_id",
        "season",
        "period",
        "t_min",
        "band",
        "p7",
        "y",
        "contact",
    ):
        assert col in df.columns
    for f in truth.feature_names:
        assert f in df.columns
    # at most one non-contact event per player-match, and nothing after it
    ev = df.groupby(["match_id", "player_id"])["y"].sum()
    assert ev.max() <= 1


def test_realised_rate_matches_truth() -> None:
    truth = SynthTruth(p7_mean=0.10)
    df = generate_person_minutes(4000, seed=3, truth=truth, informative_censoring=False)
    expected = truth.match_share * -np.log1p(-truth.p7_mean)  # approx mean Lambda per 90 min
    minutes = len(df)
    realised_per_90 = df["y"].sum() / minutes * 90
    assert realised_per_90 == pytest.approx(expected, rel=0.25)


def test_full_match_sanity() -> None:
    m = generate_match(seed=4)
    assert len(m.lineups) == 2
    s = m.samples
    assert {"match_id", "player_id", "team_id", "period", "t_s", "x_m", "y_m", "source"} <= set(
        s.columns
    )
    # median total distance of 90-minute outfield players within sanity bounds
    # (TODO: compute per player with the M3 feature code once available,
    # or a simple path length here)


# --- speed process (docs/05_VALIDATION.md §1, config synth) ---


def test_speed_process_matrices_are_ergodic(config_path: Path) -> None:
    from livrisk import markov as mk
    from livrisk.config import load_config
    from livrisk.synth.generator import jump_matrix_for, speed_process_from_config

    proc = speed_process_from_config(load_config(config_path))
    for band in range(8):
        for pos in ("CB", "FB", "MF", "W", "FW"):
            assert mk.is_ergodic(jump_matrix_for(proc, band, pos))


def test_simulated_occupancy_matches_semi_markov_law(config_path: Path) -> None:
    # TODO: simulate 200 midfielder-bands with simulate_speed; for each band the pooled share of
    # ticks per target zone must be within total-variation distance 0.03 of expected_occupancy.
    pytest.fail("implement: occupancy vs closed form, per band")


def test_dwell_times_follow_configured_distribution(config_path: Path) -> None:
    # TODO: bout lengths per zone pass a KS test (p > 0.01) against dwell_min + Gamma(shape,
    # (mean - min) / shape); sprint bouts are never shorter than 1 s.
    pytest.fail("implement: dwell shape")


def test_acceleration_limits_and_late_decline(config_path: Path) -> None:
    # TODO: per-tick speed change within accel/decel limits (plus jitter tolerance); median HSR in
    # band 6 is lower than in band 0 by a share inside synth.late_hsr_drop_bounds.
    pytest.fail("implement: physics and within-match decline")


def test_burn_in_removes_kickoff_bias(config_path: Path) -> None:
    # TODO: zone occupancy in minutes 0-2 matches band-0 expected occupancy (TV < 0.05 over 500
    # simulated players); without burn-in it would start in stand.
    pytest.fail("implement: stationary start")
