from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.typing import NDArray
from scipy import stats

from livrisk import markov, riskmath
from livrisk.config import load_config
from livrisk.contracts import PositionGroup, TrackingSample
from livrisk.synth.generator import (
    SynthTruth,
    expected_occupancy,
    generate_match,
    generate_person_minutes,
    jump_matrix_for,
    simulate_speed,
    speed_process_from_config,
    write_match,
)

pytestmark = [pytest.mark.milestone_m2]

FloatArray = NDArray[np.float64]


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
    ev = df.groupby(["match_id", "player_id"], observed=True)["y"].sum()
    assert ev.max() <= 1
    last = df[df["y"] == 1].groupby(["match_id", "player_id"], observed=True)["t_min"].max()
    rows = df.groupby(["match_id", "player_id"], observed=True)["t_min"].max()
    assert (last <= rows.loc[last.index]).all()


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
    # Median total distance (path length) of near-full-match outfield players within bounds.
    total_ticks = int(s.groupby("player_id")["t_s"].count().max())
    position = {p.player_id: p.position_group for lu in m.lineups for p in lu.players if p.starting}
    distances: list[float] = []
    for pid, grp in s.groupby("player_id"):
        if position.get(str(pid)) in (None, PositionGroup.GK) or len(grp) < 0.95 * total_ticks:
            continue
        xy = grp.sort_values(["period", "t_s"])[["x_m", "y_m"]].to_numpy()
        distances.append(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()))
    assert len(distances) >= 10
    assert 9000.0 <= float(np.median(distances)) <= 13000.0


# --- speed process (docs/05_VALIDATION.md §1, config synth) ---

_HZ = 10.0


def _fixed_band(band: int) -> Callable[[float], int]:
    """Band lookup pinned to one band."""

    def lookup(_t_s: float) -> int:
        return band

    return lookup


def _match_band_of_t(p1_minutes: int = 47) -> Callable[[float], int]:
    """Band lookup for a standard match: period 1 of ``p1_minutes``, then period 2."""

    def band_of_t(t_s: float) -> int:
        if t_s < p1_minutes * 60.0:
            return riskmath.band_index(t_s / 60.0, 1)
        return riskmath.band_index((2700.0 + t_s - p1_minutes * 60.0) / 60.0, 2)

    return band_of_t


def test_speed_process_matrices_are_ergodic(config_path: Path) -> None:
    proc = speed_process_from_config(load_config(config_path))
    for band in range(8):
        for pos in ("CB", "FB", "MF", "W", "FW"):
            assert markov.is_ergodic(jump_matrix_for(proc, band, pos))


def test_speed_process_rejects_non_ergodic(config_path: Path, tmp_path: Path) -> None:
    data = yaml.safe_load(config_path.read_text())
    data["synth"]["jump_matrix"] = [
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="ergodic"):
        speed_process_from_config(load_config(p))


def test_simulated_occupancy_matches_semi_markov_law(config_path: Path) -> None:
    proc = speed_process_from_config(load_config(config_path))
    rng = np.random.default_rng(11)
    for band in range(8):
        counts = np.zeros(6)
        for _ in range(25):
            _, zones = simulate_speed(proc, "MF", 8.9, 900.0, _fixed_band(band), rng, hz=_HZ)
            counts += np.bincount(zones, minlength=6)
        empirical = counts / counts.sum()
        assert markov.tv_distance(empirical, expected_occupancy(proc, band, "MF")) < 0.03


def _bout_lengths_s(zones: NDArray[np.int64], hz: float) -> dict[int, list[float]]:
    """Bout lengths per zone in seconds, excluding the censored first and last bouts."""
    edges = np.flatnonzero(np.diff(zones)) + 1
    runs = np.split(zones, edges)
    out: dict[int, list[float]] = {}
    for run in runs[1:-1]:
        out.setdefault(int(run[0]), []).append(len(run) / hz)
    return out


def test_dwell_times_follow_configured_distribution(config_path: Path) -> None:
    cfg = load_config(config_path)
    proc = speed_process_from_config(cfg)
    rng = np.random.default_rng(12)
    bouts: dict[int, list[float]] = {}
    for _ in range(60):
        _, zones = simulate_speed(proc, "MF", 8.9, 900.0, _fixed_band(0), rng, hz=_HZ)
        for zone, lengths in _bout_lengths_s(zones, _HZ).items():
            bouts.setdefault(zone, []).extend(lengths)
    for zone in range(6):
        sample = np.asarray(bouts[zone])
        assert len(sample) > 50, f"zone {zone}: too few bouts"
        floor = cfg.synth.dwell_min_s[zone]
        shape = cfg.synth.dwell_gamma_shape[zone]
        scale = (cfg.synth.dwell_mean_s[zone] - floor) / shape
        dist = stats.gamma(a=shape, loc=floor, scale=scale)
        assert stats.kstest(sample, dist.cdf).pvalue > 0.01, f"zone {zone} dwell shape"
    assert min(bouts[5]) >= 1.0, "sprint bouts are never shorter than 1 s"


def test_acceleration_limits_and_late_decline(config_path: Path) -> None:
    cfg = load_config(config_path)
    proc = speed_process_from_config(cfg)
    rng = np.random.default_rng(13)
    limit = cfg.synth.decel_max_mps2 / _HZ
    hsr: dict[int, float] = {}
    for band in (0, 6):
        total = 0.0
        for _ in range(250):
            speeds, _ = simulate_speed(proc, "MF", 8.9, 900.0, _fixed_band(band), rng, hz=_HZ)
            assert np.all(np.abs(np.diff(speeds)) <= limit + 1e-9)
            total += float(speeds[speeds >= cfg.features.hsr_threshold_mps].sum()) / _HZ
        hsr[band] = total
    drop = 1.0 - hsr[6] / hsr[0]
    lo, hi = cfg.synth.late_hsr_drop_bounds
    assert lo <= drop <= hi


def test_burn_in_removes_kickoff_bias(config_path: Path) -> None:
    proc = speed_process_from_config(load_config(config_path))
    rng = np.random.default_rng(14)
    counts = np.zeros(6)
    for _ in range(500):
        _, zones = simulate_speed(proc, "MF", 8.9, 120.0, _fixed_band(0), rng, hz=_HZ)
        counts += np.bincount(zones, minlength=6)
    empirical = counts / counts.sum()
    assert markov.tv_distance(empirical, expected_occupancy(proc, 0, "MF")) < 0.05


# --- full path: determinism, sanity, contracts ---


def test_full_match_byte_identical_parquet(tmp_path: Path) -> None:
    a = write_match(generate_match(seed=5), tmp_path / "a")
    b = write_match(generate_match(seed=5), tmp_path / "b")
    for name in a:
        assert a[name].read_bytes() == b[name].read_bytes(), name


def test_full_path_sanity_200_players(config_path: Path) -> None:
    cfg = load_config(config_path)
    proc = speed_process_from_config(cfg)
    rng = np.random.default_rng(15)
    band_of_t = _match_band_of_t()
    duration_s = 95.0 * 60.0
    dist: list[float] = []
    hsr: list[float] = []
    spr: list[float] = []
    peak: list[float] = []
    positions = ["CB", "FB", "MF", "W", "FW"]
    vmax_lo, vmax_hi = cfg.synth.vmax_bounds_mps
    for i in range(200):
        vmax = float(
            np.clip(rng.normal(cfg.synth.vmax_mean_mps, cfg.synth.vmax_sd_mps), vmax_lo, vmax_hi)
        )
        speeds, _ = simulate_speed(proc, positions[i % 5], vmax, duration_s, band_of_t, rng)
        dist.append(float(speeds.sum()) / _HZ)
        hsr.append(float(speeds[speeds >= cfg.features.hsr_threshold_mps].sum()) / _HZ)
        spr.append(float(speeds[speeds >= cfg.features.sprint_threshold_mps].sum()) / _HZ)
        window = round(_HZ)
        peak.append(float(np.max(np.convolve(speeds, np.ones(window) / window, mode="same"))))
    assert 9000.0 <= float(np.median(dist)) <= 13000.0
    assert 400.0 <= float(np.median(hsr)) <= 1200.0
    assert 100.0 <= float(np.median(spr)) <= 450.0
    assert 8.0 <= float(np.median(peak)) <= 10.0


def test_outputs_validate_contracts() -> None:
    m = generate_match(seed=6)
    rows = m.samples.iloc[::997]
    for record in rows.to_dict(orient="records"):
        TrackingSample.model_validate({k: (None if v is pd.NA else v) for k, v in record.items()})
    ev = m.person_minutes.groupby("player_id")["y"].sum()
    assert ev.max() <= 1
    # Labels and lineups are contract instances by construction; check coherence.
    assert all(lbl.match_id == m.match_id for lbl in m.injuries)
    assert all(lu.match_id == m.match_id for lu in m.lineups)
    on_pitch = set(m.samples["player_id"].unique())
    assert {p.player_id for lu in m.lineups for p in lu.players if p.starting} <= (
        on_pitch | {lbl.player_id for lbl in m.injuries}
    )


def test_realised_rate_and_profile_over_2000_matches() -> None:
    truth = SynthTruth()
    n_matches = 2000
    df = generate_person_minutes(n_matches * 18, seed=7)
    hours = len(df) / 60.0
    realised = df["y"].sum() / hours * 1000.0
    configured = truth.match_share * -np.log1p(-truth.p7_mean) / 1.5 * 1000.0
    assert realised == pytest.approx(configured, rel=0.15)
    # Time profile: observed events per band vs the exact hazard of the generated rows.
    g = truth.g_levels()
    beta = np.asarray(truth.beta)
    z = df[list(truth.feature_names)].to_numpy()
    lam = (
        truth.match_share
        * (-np.log1p(df["p7"].to_numpy()))
        / riskmath.REGULAR_MINUTES
        * g[df["band"].to_numpy()]
        * np.exp(z @ beta)
    )
    bands = sorted(df["band"].unique())
    observed = df.groupby("band")["y"].sum().to_numpy(dtype=np.float64)
    expected = np.array([lam[(df["band"].to_numpy() == b)].sum() for b in bands], dtype=np.float64)
    expected = expected * float(observed.sum() / expected.sum())
    assert stats.chisquare(observed, expected).pvalue > 0.01
