"""M13 acceptance tests: video identity assignment (docs/VIDEO_INPUT.md).

Synthetic external tracks are built from an M2 match: one track per player spell, per-frame
jersey reads with 2 % misreads (and 3 % unreadable frames), and injected tracker identity
swaps between the physically closest same-team pair. The assigner must recover the physical
player for >= 95 % of player-seconds and flag every injected swap within 5 s; a jersey-read
outage must lower identity_confidence enough to trip the downstream identity_low gate.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from livrisk.config import load_config
from livrisk.contracts import PERIOD_START_S, EventType, MatchEvent, Quality, TrackingSample
from livrisk.features import stream
from livrisk.synth.generator import SynthMatch, generate_match, match_tables
from livrisk.video.identity import assign_identities

pytestmark = [pytest.mark.milestone_m13]

FloatArray = NDArray[np.float64]

SEED = 7
SAMPLE_HZ = 5.0
MISREAD_P = 0.02
UNREADABLE_P = 0.03
READ_SEED = 20261023


def _period_starts(events: tuple[MatchEvent, ...]) -> dict[int, float]:
    """Play-clock start of each period from the period_end events."""
    lens = {1: 2700.0, 2: 2700.0, 3: 900.0, 4: 900.0}
    for e in events:
        if e.type == EventType.PERIOD_END:
            lens[e.period] = e.t_s - PERIOD_START_S[e.period]
    return {1: 0.0, 2: lens[1], 3: lens[1] + lens[2], 4: lens[1] + lens[2] + lens[3]}


def _u_vec(starts: dict[int, float], period: NDArray[np.int64], t_s: FloatArray) -> FloatArray:
    """Play seconds for columns of (period, t_s)."""
    start_of = np.asarray([starts[p] for p in (1, 2, 3, 4)])
    origin = np.asarray([PERIOD_START_S[p] for p in (1, 2, 3, 4)])
    return np.asarray(start_of[period - 1] + (t_s - origin[period - 1]), dtype=np.float64)


def _u(starts: dict[int, float], period: int, t_s: float) -> float:
    return starts[period] + (t_s - PERIOD_START_S[period])


@dataclass(frozen=True)
class TrackCase:
    """External-pipeline tracks with ground truth: ``truth[i]`` is the physical player of
    row i, whatever its track label says."""

    match: SynthMatch
    tracks: pd.DataFrame
    truth: NDArray[np.object_]
    starts: dict[int, float]
    swaps: tuple[tuple[float, str, str], ...]


def _base_tracks(match: SynthMatch, seed: int) -> tuple[pd.DataFrame, NDArray[np.object_]]:
    """One track per player spell with per-frame jersey reads, before any swap injection.

    A read is the physical player's shirt with probability 1 - MISREAD_P - UNREADABLE_P;
    a misread is another squad shirt, never the true one; an unreadable frame has no read.
    """
    rng = np.random.default_rng(seed)
    s = match.samples
    truth = np.asarray(s["player_id"].to_numpy(), dtype=object)
    track_of = {pid: f"trk{i:03d}" for i, pid in enumerate(sorted(set(truth.tolist())))}
    shirt_of = {p.player_id: p.shirt_number for lu in match.lineups for p in lu.players}
    shirts = np.asarray([shirt_of[str(p)] for p in truth], dtype=np.int64)
    n = int(truth.size)
    read = shirts.astype(np.float64)
    roll = rng.random(n)
    misread = roll < MISREAD_P
    unreadable = (roll >= MISREAD_P) & (roll < MISREAD_P + UNREADABLE_P)
    offset = rng.integers(1, 20, size=n)
    wrong = ((shirts - 1 + offset) % 20 + 1).astype(np.float64)
    read = np.where(misread, wrong, read)
    read = np.where(unreadable, np.nan, read)
    conf = np.where(unreadable, np.nan, rng.uniform(0.6, 0.95, size=n))
    tracks = pd.DataFrame(
        {
            "frame_t_s": s["t_s"].to_numpy(dtype=np.float64),
            "period": s["period"].to_numpy(dtype=np.int64),
            "track_id": np.asarray([track_of[str(p)] for p in truth], dtype=object),
            "team_guess": s["team_id"].to_numpy(dtype=object),
            "jersey_read": read,
            "jersey_conf": conf,
            "x_m": s["x_m"].to_numpy(dtype=np.float64),
            "y_m": s["y_m"].to_numpy(dtype=np.float64),
            "detected": s["detected"].to_numpy(dtype=np.bool_),
        }
    )
    return tracks, truth


def _inject_swaps(
    tracks: pd.DataFrame,
    truth: NDArray[np.object_],
    match: SynthMatch,
    starts: dict[int, float],
    n_swaps: int = 3,
) -> tuple[tuple[float, str, str], ...]:
    """Exchange two track ids from a chosen moment on: a tracker identity swap.

    The pair is the physically closest same-team pair on the pitch at that moment; the jersey
    reads keep following the physical player, which is what lets the assigner catch the swap.
    Swap moments stay 60 s away from substitutions and red cards and 30 s from the end.
    """
    period = tracks["period"].to_numpy(dtype=np.int64)
    t_s = tracks["frame_t_s"].to_numpy(dtype=np.float64)
    u = _u_vec(starts, period, t_s)
    busy = sorted(
        _u(starts, e.period, e.t_s)
        for e in match.events
        if e.type in (EventType.SUBSTITUTION, EventType.RED_CARD)
    )
    max_u = float(np.max(u))
    candidates = [
        c
        for c in (900.0, 1500.0, 2100.0, 3600.0, 4200.0, 4800.0, 5300.0)
        if c < max_u - 30.0 and all(abs(c - b) >= 60.0 for b in busy)
    ]
    tid = tracks["track_id"].to_numpy(dtype=object).copy()
    teams = tracks["team_guess"].to_numpy(dtype=object)
    x = tracks["x_m"].to_numpy(dtype=np.float64)
    y = tracks["y_m"].to_numpy(dtype=np.float64)
    swaps: list[tuple[float, str, str]] = []
    for k, u_tau in enumerate(candidates[:n_swaps]):
        team = ("T1", "T2")[k % 2]
        idx = np.flatnonzero((np.abs(u - u_tau) < 0.5 / SAMPLE_HZ) & (teams == team))
        pos = np.column_stack([x[idx], y[idx]])
        d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        a_i, b_i = np.unravel_index(int(np.argmin(d2)), d2.shape)
        ta, tb = str(tid[idx[a_i]]), str(tid[idx[b_i]])
        pa, pb = str(truth[idx[a_i]]), str(truth[idx[b_i]])
        mask = u >= u_tau
        mask_a = mask & (tid == ta)
        mask_b = mask & (tid == tb)
        tid[mask_a] = tb
        tid[mask_b] = ta
        swaps.append((u_tau, pa, pb))
    tracks["track_id"] = tid
    return tuple(swaps)


def _truth_by_key(case: TrackCase) -> dict[tuple[str, int, float], str]:
    """(track_id, period, frame_t_s) -> physical player, from the generator's truth."""
    keys = zip(
        case.tracks["track_id"].astype(str),
        case.tracks["period"].astype(int),
        case.tracks["frame_t_s"].astype(float),
        strict=True,
    )
    return {k: str(p) for k, p in zip(keys, case.truth.tolist(), strict=True)}


def _valid_sample_row(record: dict[Hashable, Any]) -> TrackingSample:
    """One output row (NaN/NA allowed for the optional fields) as a TrackingSample."""
    data = dict(record)
    data.pop("track_id")
    for key in ("x_m", "y_m", "speed_mps", "wall_time_ns", "ingest_time_ns"):
        value = data[key]
        if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
            data[key] = None
    return TrackingSample.model_validate(data)


@pytest.fixture(scope="module")
def match() -> SynthMatch:
    return generate_match(SEED, sample_hz=SAMPLE_HZ)


def test_identity_accuracy_with_misreads_and_swaps(match: SynthMatch) -> None:
    tracks, truth = _base_tracks(match, READ_SEED)
    starts = _period_starts(match.events)
    swaps = _inject_swaps(tracks, truth, match, starts)
    assert len(swaps) >= 3
    case = TrackCase(match=match, tracks=tracks, truth=truth, starts=starts, swaps=swaps)

    result = assign_identities(case.tracks, match.lineups, match.events)
    out = result.samples
    truth_of = _truth_by_key(case)
    keys = zip(
        out["track_id"].astype(str),
        out["period"].astype(int),
        out["t_s"].astype(float),
        strict=True,
    )
    physical = np.asarray([truth_of[k] for k in keys], dtype=object)
    correct = physical == out["player_id"].to_numpy(dtype=object)
    accuracy = float(correct.mean())
    # Player-seconds are sample shares at a fixed rate. Dropped rows are a coverage gap, so
    # they count against the match-level share but not against the emitted accuracy.
    emitted_share = out.shape[0] / tracks.shape[0]
    assert accuracy >= 0.95
    assert emitted_share * accuracy >= 0.95
    assert result.dropped_rows == tracks.shape[0] - out.shape[0]

    # Every injected swap is flagged on both tracks, within 5 s of the injection.
    assert result.swaps.shape[0] == 2 * len(swaps)
    for u_tau, pa, pb in swaps:
        flags = result.swaps.loc[
            ((result.swaps["from_player_id"] == pa) & (result.swaps["to_player_id"] == pb))
            | ((result.swaps["from_player_id"] == pb) & (result.swaps["to_player_id"] == pa))
        ]
        directions = set(zip(flags["from_player_id"], flags["to_player_id"], strict=True))
        assert directions == {(pa, pb), (pb, pa)}
        for f_period, f_t in zip(flags["period"], flags["t_s"], strict=True):
            latency = _u(starts, int(f_period), float(f_t)) - u_tau
            assert 0.0 <= latency <= 5.0

    # Substituted-on players are identified through the event-driven on-pitch set.
    subbed_in = {str(e.player_in_id) for e in match.events if e.type == EventType.SUBSTITUTION}
    seen = set(out["player_id"].astype(str).unique())
    assert subbed_in <= seen

    # The output satisfies the TrackingSample contract row by row (spot check).
    for rec in out.sample(min(300, out.shape[0]), random_state=0).to_dict("records"):
        _valid_sample_row(rec)

    # Deterministic: a prefix assigned twice gives identical output.
    u_in = _u_vec(
        starts,
        tracks["period"].to_numpy(dtype=np.int64),
        tracks["frame_t_s"].to_numpy(dtype=np.float64),
    )
    prefix = tracks.loc[u_in <= 900.0]
    first = assign_identities(prefix, match.lineups, match.events)
    second = assign_identities(prefix, match.lineups, match.events)
    pd.testing.assert_frame_equal(first.samples, second.samples)
    pd.testing.assert_frame_equal(first.swaps, second.swaps)


def test_low_confidence_lowers_identity_confidence(match: SynthMatch, config_path: Path) -> None:
    cfg = load_config(config_path)
    tracks, truth = _base_tracks(match, READ_SEED + 1)
    starts = _period_starts(match.events)
    u_in = _u_vec(
        starts,
        tracks["period"].to_numpy(dtype=np.int64),
        tracks["frame_t_s"].to_numpy(dtype=np.float64),
    )
    # A starter on the pitch all match carries the outage.
    u_max = float(np.max(u_in))
    span = pd.Series(u_in).groupby(pd.Series(truth)).agg(["min", "max"])
    durable = span.loc[(span["min"] <= 60.0) & (span["max"] >= u_max - 60.0)]
    target = str(sorted(durable.index)[0])
    tau0, tau1 = 1800.0, 1920.0  # period 1, away from the break
    outage = (truth == target) & (u_in >= tau0) & (u_in < tau1)
    tracks.loc[outage, "jersey_read"] = np.nan
    tracks.loc[outage, "jersey_conf"] = np.nan

    result = assign_identities(tracks, match.lineups, match.events)
    assert result.swaps.shape[0] == 0  # a read outage must not fake a swap
    out = result.samples
    u_out = _u_vec(
        starts,
        out["period"].to_numpy(dtype=np.int64),
        out["t_s"].to_numpy(dtype=np.float64),
    )
    mine = out["player_id"].to_numpy(dtype=object) == target
    conf = out["identity_confidence"].to_numpy(dtype=np.float64)
    early = mine & (u_out >= tau0) & (u_out < tau0 + 20.0)
    drained = mine & (u_out >= tau0 + 40.0) & (u_out < tau1)
    assert int(early.sum()) > 0 and int(drained.sum()) > 0
    # The trailing read memory keeps confidence up briefly, then honesty takes over.
    assert bool((conf[early] >= cfg.quality.min_identity).all())
    assert bool((conf[drained] < cfg.quality.min_identity).all())
    # Positional continuity still carries the identity through the outage: of the target's
    # own track rows in the drained segment, nearly all come out as the target (drops and
    # wrong identities would both count against this share).
    assigned = dict(
        zip(
            zip(
                out["track_id"].astype(str),
                out["period"].astype(int),
                out["t_s"].astype(float),
                strict=True,
            ),
            out["player_id"].astype(str),
            strict=True,
        )
    )
    seg_in = (truth == target) & (u_in >= tau0 + 40.0) & (u_in < tau1)
    seg_rows = tracks.loc[seg_in]
    seg_keys = zip(
        seg_rows["track_id"].astype(str),
        seg_rows["period"].astype(int),
        seg_rows["frame_t_s"].astype(float),
        strict=True,
    )
    kept = float(np.mean([assigned.get(k) == target for k in seg_keys]))
    assert kept >= 0.9

    # Downstream: the 15-minute window minimum trips the identity_low quality gate while
    # coverage stays fine (positions are real; only the reads are missing).
    events = match_tables(match)["events"]
    feed = out.drop(columns=["track_id"])
    snap = stream.compute_features_offline(
        feed, events, target, tau1 + cfg.engine.blanking_delta_s, match.vmax_mps.get(target), cfg
    )
    assert snap.identity_confidence < cfg.quality.min_identity
    assert "identity_low" in snap.quality_reasons
    assert snap.quality == Quality.DEGRADED
    assert snap.coverage["hsr_5m"] >= cfg.quality.min_coverage
    # Control: before the outage the same player is fully identified.
    before = stream.compute_features_offline(
        feed, events, target, tau0 - 70.0, match.vmax_mps.get(target), cfg
    )
    assert before.identity_confidence >= 0.9
    assert before.quality == Quality.GOOD


def test_input_validation(match: SynthMatch) -> None:
    tracks, _ = _base_tracks(match, READ_SEED)
    small = tracks.iloc[:1000].copy()
    for col in ("frame_t_s", "track_id", "team_guess"):
        bad = small.drop(columns=[col])
        with pytest.raises(ValueError, match="missing required columns"):
            assign_identities(bad, match.lineups, match.events)
    bad_team = small.copy()
    bad_team["team_guess"] = "referees"
    with pytest.raises(ValueError, match="team_guess"):
        assign_identities(bad_team, match.lineups, match.events)
    bad_conf = small.copy()
    bad_conf.loc[bad_conf.index[0], "jersey_conf"] = 1.5
    with pytest.raises(ValueError, match="jersey_conf"):
        assign_identities(bad_conf, match.lineups, match.events)
    with pytest.raises(ValueError, match="at least one lineup"):
        assign_identities(small, [], match.events)
