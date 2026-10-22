"""M3 acceptance tests: stream feature engine (docs/02_MATH.md §10).

The online engine must reproduce the batch reference exactly, never read past the blanking
cutoff, and flag low-quality data. Times in the unit fixtures are play-clock seconds
(period 1 only), matching the convention of ``livrisk.features.stream``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from livrisk.config import load_config
from livrisk.contracts import (
    PERIOD_START_S,
    EventType,
    FeatureSnapshot,
    MatchEvent,
    Quality,
    Source,
    TrackingSample,
)
from livrisk.features import stream
from livrisk.features.stream import detect_efforts, smooth_speed
from livrisk.synth.generator import SynthMatch, generate_match, match_tables

pytestmark = [pytest.mark.milestone_m3]

FloatArray = NDArray[np.float64]

_SAMPLE_COLUMNS = list(TrackingSample.model_fields)
_EVENT_COLUMNS = list(MatchEvent.model_fields)


def _trace(segments: list[tuple[float, float]], hz: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant speed trace from (duration_s, speed_mps) segments."""
    t: list[float] = []
    v: list[float] = []
    now = 0.0
    for dur, speed in segments:
        n = round(dur * hz)
        t.extend(now + i / hz for i in range(n))
        v.extend([speed] * n)
        now += dur
    return np.asarray(t), np.asarray(v)


def _unit_samples(
    t_s: FloatArray,
    speed_mps: FloatArray,
    *,
    player_id: str = "P1",
    detected: NDArray[np.bool_] | None = None,
    identity: FloatArray | None = None,
) -> pd.DataFrame:
    """One player's tracking frame (contract columns), period 1, speed-only."""
    n = int(t_s.size)
    nan = np.full(n, np.nan)
    missing = pd.array([pd.NA] * n, dtype="Int64")
    return pd.DataFrame(
        {
            "match_id": "m1",
            "player_id": player_id,
            "team_id": "T1",
            "period": np.ones(n, dtype=np.int64),
            "t_s": t_s,
            "x_m": nan,
            "y_m": nan,
            "speed_mps": speed_mps,
            "source": "synthetic",
            "detected": np.ones(n, dtype=np.bool_) if detected is None else detected,
            "identity_confidence": np.ones(n) if identity is None else identity,
            "wall_time_ns": missing,
            "ingest_time_ns": missing.copy(),
        }
    )[_SAMPLE_COLUMNS]


def _unit_events(rows: list[tuple[float, str]]) -> pd.DataFrame:
    """Events frame (contract columns) from (t_s, type) pairs in period 1."""
    return pd.DataFrame(
        {
            "match_id": ["m1"] * len(rows),
            "period": [1] * len(rows),
            "t_s": [t for t, _ in rows],
            "type": [ty for _, ty in rows],
            "team_id": [None] * len(rows),
            "player_id": [None] * len(rows),
            "player_in_id": [None] * len(rows),
            "stoppage_min": [None] * len(rows),
            "source": ["official_feed"] * len(rows),
        }
    )[_EVENT_COLUMNS]


def _opt_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _opt_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _to_event(rec: dict[Hashable, Any]) -> MatchEvent:
    """One events-frame row (NaN for missing) as a MatchEvent."""
    return MatchEvent(
        match_id=str(rec["match_id"]),
        period=int(rec["period"]),
        t_s=float(rec["t_s"]),
        type=EventType(str(rec["type"])),
        team_id=_opt_str(rec["team_id"]),
        player_id=_opt_str(rec["player_id"]),
        player_in_id=_opt_str(rec["player_in_id"]),
        stoppage_min=_opt_float(rec["stoppage_min"]),
    )


def _true_period_starts(events: pd.DataFrame) -> dict[int, float]:
    """Play-clock start of each period from the period_end events (real-time order)."""
    lens = {1: 2700.0, 2: 2700.0, 3: 900.0, 4: 900.0}
    ends = events.loc[events["type"] == str(EventType.PERIOD_END)]
    for rec in ends.to_dict("records"):
        period = int(rec["period"])
        lens[period] = float(rec["t_s"]) - PERIOD_START_S[period]
    starts = {1: 0.0}
    for p in (2, 3, 4):
        starts[p] = starts[p - 1] + lens[p - 1]
    return starts


def _play_seconds(starts: dict[int, float], period: Any, t_s: Any) -> float:
    """Play seconds of a match-clock stamp; the association order matches _PeriodClock.to_u."""
    p = int(period)
    return starts[p] + (float(t_s) - PERIOD_START_S[p])


def _feed_match(
    match: SynthMatch,
    events: pd.DataFrame,
    samples: pd.DataFrame,
    engine: stream.StreamFeatureEngine,
    tick_s: float,
) -> list[tuple[float, list[FeatureSnapshot]]]:
    """Play a match into the engine in real-time order, ticking every ``tick_s`` play seconds.

    Events go before samples at the same play second (the period end arrives before the next
    period's samples). Also checks the on-pitch set against an independent mirror of the
    substitution and red-card rules.
    """
    starts = _true_period_starts(events)
    ev_recs = events.to_dict("records")
    ev_u = [_play_seconds(starts, r["period"], r["t_s"]) for r in ev_recs]
    ev_order = np.argsort(np.asarray(ev_u), kind="stable")
    cols = {c: samples[c].to_numpy() for c in samples.columns}
    sam_u = np.asarray(
        [_play_seconds(starts, p, t) for p, t in zip(cols["period"], cols["t_s"], strict=True)]
    )
    sam_order = np.argsort(sam_u, kind="stable")

    on_pitch: set[str] = set()
    out: list[tuple[float, list[FeatureSnapshot]]] = []
    ei = 0
    si = 0
    n_ticks = int(np.max(sam_u) / tick_s)
    for k in range(1, n_ticks + 1):
        u = k * tick_s
        while ei < ev_order.size and ev_u[int(ev_order[ei])] <= u:
            event = _to_event(ev_recs[int(ev_order[ei])])
            engine.push_event(event)
            if (
                event.type == EventType.SUBSTITUTION
                and event.player_id is not None
                and event.player_in_id is not None
            ):
                on_pitch.discard(event.player_id)
                on_pitch.add(event.player_in_id)
            elif event.type == EventType.RED_CARD and event.player_id is not None:
                on_pitch.discard(event.player_id)
            ei += 1
        while si < sam_order.size and float(sam_u[sam_order[si]]) <= u:
            i = int(sam_order[si])
            engine.push_samples(
                [
                    TrackingSample(
                        match_id=str(cols["match_id"][i]),
                        player_id=str(cols["player_id"][i]),
                        team_id=str(cols["team_id"][i]),
                        period=int(cols["period"][i]),
                        t_s=float(cols["t_s"][i]),
                        x_m=float(cols["x_m"][i]),
                        y_m=float(cols["y_m"][i]),
                        speed_mps=float(cols["speed_mps"][i]),
                        source=Source(str(cols["source"][i])),
                        detected=bool(cols["detected"][i]),
                        identity_confidence=float(cols["identity_confidence"][i]),
                    )
                ]
            )
            on_pitch.add(str(cols["player_id"][i]))
            si += 1
        snapshots = engine.tick(u)
        assert sorted(s.player_id for s in snapshots) == sorted(on_pitch)
        out.append((u, snapshots))
    return out


def _assert_snapshot_close(a: FeatureSnapshot, b: FeatureSnapshot) -> None:
    """Field-wise equality to 1e-9 absolute (None counts as equal to None)."""
    assert a.period == b.period
    assert a.as_of_t_s == b.as_of_t_s
    assert a.cutoff_t_s == b.cutoff_t_s
    assert a.quality == b.quality
    assert a.quality_reasons == b.quality_reasons
    assert a.z == b.z
    assert abs(a.identity_confidence - b.identity_confidence) <= 1e-9
    assert a.features.keys() == b.features.keys()
    for name, va in a.features.items():
        vb = b.features[name]
        assert (va is None) == (vb is None), name
        if va is not None and vb is not None:
            assert abs(va - vb) <= 1e-9, name
    assert a.coverage.keys() == b.coverage.keys()
    for name, ca in a.coverage.items():
        assert abs(ca - b.coverage[name]) <= 1e-9, name


def test_effort_detection_merges_short_gaps_and_ignores_blips() -> None:
    t, v = _trace([(5, 3.0), (2.0, 7.5), (0.3, 6.0), (1.5, 7.5), (5, 3.0), (0.6, 7.5), (5, 3.0)])
    efforts = detect_efforts(t, v, threshold_mps=7.0, min_duration_s=1.0, merge_gap_s=0.5)
    assert len(efforts) == 1  # 2.0 s + 0.3 s gap + 1.5 s merged; the 0.6 s blip ignored
    assert efforts[0].start_s == pytest.approx(5.0, abs=0.11)
    assert efforts[0].end_s == pytest.approx(8.8, abs=0.11)


def test_smooth_speed_constant_velocity() -> None:
    hz = 25.0
    t = np.arange(0, 20, 1 / hz)
    x = 6.0 * t
    y = np.zeros_like(t)
    s = smooth_speed(t, x, y, window_s=1.0, polyorder=2)
    assert np.allclose(s[int(hz) : -int(hz)], 6.0, atol=1e-6)


def test_online_equals_offline_on_synthetic_match(config_path: Path) -> None:
    cfg = load_config(config_path)
    match = generate_match(seed=7, cfg=cfg)
    events = match_tables(match)["events"]
    team = str(match.samples["team_id"].iloc[0])
    samples = match.samples.loc[match.samples["team_id"] == team]
    by_player = {str(pid): g for pid, g in samples.groupby("player_id")}
    empty_frame = samples.iloc[0:0]

    engine = stream.StreamFeatureEngine(cfg, match.match_id, match.vmax_mps)
    ticks = _feed_match(match, events, samples, engine, float(cfg.engine.feature_tick_s))
    assert len(ticks) > 500
    n_checked = 0
    for u, snapshots in ticks:
        for snap in snapshots:
            ref = stream.compute_features_offline(
                by_player.get(snap.player_id, empty_frame),
                events,
                snap.player_id,
                u,
                match.vmax_mps.get(snap.player_id),
                cfg,
            )
            _assert_snapshot_close(snap, ref)
            n_checked += 1
    assert n_checked > 5000


def test_no_read_beyond_cutoff(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(config_path)
    match = generate_match(seed=9, cfg=cfg)
    events = match_tables(match)["events"]
    team = str(match.samples["team_id"].iloc[0])
    samples = match.samples.loc[match.samples["team_id"] == team]

    original_prefix = stream._SampleBuffer.prefix
    n_calls = 0

    def spying_prefix(self: stream._SampleBuffer, cutoff_u: float) -> stream._SeriesData:
        nonlocal n_calls
        series = original_prefix(self, cutoff_u)
        n_calls += 1
        if series.grid.size:
            assert float(series.grid[-1]) <= cutoff_u + 1e-9
        return series

    monkeypatch.setattr(stream._SampleBuffer, "prefix", spying_prefix)
    engine = stream.StreamFeatureEngine(cfg, match.match_id, match.vmax_mps)
    ticks = _feed_match(match, events, samples, engine, 60.0)
    assert ticks
    assert n_calls > 0


def test_buffer_prefix_never_includes_later_samples(config_path: Path) -> None:
    cfg = load_config(config_path)
    buf = stream._SampleBuffer(cfg)
    for k in range(1000):
        buf.add(k / 10.0, math.nan, math.nan, 3.0, 1.0, 1.0)
    first = buf.prefix(40.0)
    for k in range(1001, 2000):
        buf.add(k / 10.0, math.nan, math.nan, 3.0, 1.0, 1.0)
    buf.prefix(50.0)
    second = buf.prefix(40.0)
    assert second.n_samples == 401
    assert np.array_equal(first.grid, second.grid)
    assert np.array_equal(first.speed, second.speed)
    assert np.array_equal(first.valid, second.valid)


def test_snapshot_guard_rejects_data_beyond_cutoff(config_path: Path) -> None:
    cfg = load_config(config_path)
    t = np.arange(0.0, 100.0, 0.1)
    nan = np.full(t.size, np.nan)
    series = stream._build_series(
        t, nan, nan, np.full(t.size, 3.0), np.ones(t.size, dtype=np.bool_), np.ones(t.size), cfg
    )
    with pytest.raises(RuntimeError, match="beyond the cutoff"):
        stream._snapshot_from_series(
            match_id="m1",
            player_id="P1",
            period=1,
            as_of_u=110.0,
            cutoff_u=50.0,
            series=series,
            disrupt_u=np.zeros(0),
            vmax_mps=None,
            cfg=cfg,
        )


def test_coverage_counts_only_detected_samples(config_path: Path) -> None:
    cfg = load_config(config_path)
    t = np.arange(0.0, 900.0 + 0.05, 0.1)
    samples = _unit_samples(t, np.full(t.size, 6.0), detected=t <= 750.0)
    snap = stream.compute_features_offline(samples, _unit_events([]), "P1", 960.0, 9.0, cfg)
    assert snap.quality == Quality.DEGRADED
    assert "low_coverage" in snap.quality_reasons
    # 5-min window (600, 900]: only the detected half (600, 750] carries valid samples.
    assert snap.coverage["hsr_5m"] == pytest.approx(0.5, abs=0.01)
    assert snap.features["hsr_5m"] == pytest.approx(900.0, abs=5.0)
    # 10-min window (300, 900]: three quarters detected.
    assert snap.coverage["recovery_10m"] == pytest.approx(0.75, abs=0.01)


def test_identity_confidence_is_window_minimum(config_path: Path) -> None:
    cfg = load_config(config_path)
    t = np.arange(0.0, 1560.0 + 0.05, 0.1)
    identity = np.ones(t.size)
    identity[1000] = 0.5  # t = 100 s, outside the 15-min window at cutoff 1560
    identity[9000] = 0.85  # t = 900 s, inside
    samples = _unit_samples(t, np.full(t.size, 3.0), identity=identity)
    snap = stream.compute_features_offline(samples, _unit_events([]), "P1", 1620.0, None, cfg)
    assert snap.identity_confidence == pytest.approx(0.85)
    assert snap.quality == Quality.GOOD

    identity[9000] = 0.75
    samples = _unit_samples(t, np.full(t.size, 3.0), identity=identity)
    snap = stream.compute_features_offline(samples, _unit_events([]), "P1", 1620.0, None, cfg)
    assert snap.identity_confidence == pytest.approx(0.75)
    assert snap.quality == Quality.DEGRADED
    assert "identity_low" in snap.quality_reasons


def test_implausible_speeds_are_missing_and_lower_coverage(config_path: Path) -> None:
    cfg = load_config(config_path)
    t = np.arange(0.0, 360.0 + 0.05, 0.1)
    empty = _unit_events([])

    fast = _unit_samples(t, np.full(t.size, 12.5))
    snap = stream.compute_features_offline(fast, empty, "P1", 420.0, None, cfg)
    assert snap.quality == Quality.PREMATCH_ONLY
    assert snap.features["hsr_5m"] is None
    assert snap.coverage["hsr_5m"] == 0.0

    ok = _unit_samples(t, np.full(t.size, 11.5))
    snap = stream.compute_features_offline(ok, empty, "P1", 420.0, None, cfg)
    assert snap.quality == Quality.GOOD
    assert snap.coverage["hsr_5m"] == pytest.approx(1.0, abs=1e-9)
    assert snap.features["hsr_5m"] == pytest.approx(3450.0, abs=1e-6)
    assert snap.features["spr_5m"] == pytest.approx(3450.0, abs=1e-6)


def test_disrupt_5m_counts_window_events_only(config_path: Path) -> None:
    cfg = load_config(config_path)
    t = np.arange(0.0, 720.0 + 0.05, 0.1)
    samples = _unit_samples(t, np.zeros(t.size))
    events = _unit_events(
        [
            (0.0, "kickoff"),
            (359.9, "yellow_card"),
            (361.0, "goal"),
            (400.0, "substitution"),
            (500.0, "goal"),
            (660.0, "injury_stoppage"),
            (660.1, "red_card"),
            (800.0, "goal"),
        ]
    )
    # Cutoff 660 s, 5-min window (360, 660]: goals at 361 and 500 and the stoppage at 660 count.
    snap = stream.compute_features_offline(samples, events, "P1", 720.0, None, cfg)
    assert snap.features["disrupt_5m"] == 3.0
    assert snap.coverage["disrupt_5m"] == 1.0


def test_hir_drop_zero_reference_is_missing(config_path: Path) -> None:
    cfg = load_config(config_path)
    t = np.arange(0.0, 2000.0 + 0.05, 0.1)
    empty = _unit_events([])

    slow = _unit_samples(t, np.where(t < 900.0, 3.0, 4.5))
    snap = stream.compute_features_offline(slow, empty, "P1", 2000.0, None, cfg)
    assert snap.features["hir_drop_15m"] is None

    constant = _unit_samples(t, np.full(t.size, 6.0))
    snap = stream.compute_features_offline(constant, empty, "P1", 2000.0, None, cfg)
    assert snap.features["hir_drop_15m"] == pytest.approx(1.0, abs=1e-3)


def test_recovery_10m_median_gap_between_efforts(config_path: Path) -> None:
    cfg = load_config(config_path)
    t = np.arange(0.0, 200.0 + 0.05, 0.1)
    v = np.full(t.size, 3.0)
    v[(t >= 100.0) & (t < 103.0)] = 6.5
    v[(t >= 131.0) & (t < 134.0)] = 6.5
    snap = stream.compute_features_offline(
        _unit_samples(t, v), _unit_events([]), "P1", 200.0, None, cfg
    )
    # One pair of HIEs 28 s apart (smoothing shifts both edges slightly).
    assert snap.features["recovery_10m"] == pytest.approx(28.0, abs=1.0)


def test_nearmax_cum_counts_efforts(config_path: Path) -> None:
    cfg = load_config(config_path)
    t = np.arange(0.0, 760.0 + 0.05, 0.1)
    v = np.full(t.size, 3.0)
    v[(t >= 500.0) & (t < 503.0)] = 8.2
    v[(t >= 600.0) & (t < 604.0)] = 8.2
    samples = _unit_samples(t, v)
    empty = _unit_events([])

    snap = stream.compute_features_offline(samples, empty, "P1", 760.0, 9.0, cfg)
    assert snap.features["nearmax_cum"] == 2.0  # threshold 0.85 x 9.0 = 7.65 m/s
    snap = stream.compute_features_offline(samples, empty, "P1", 610.0, 9.0, cfg)
    assert snap.features["nearmax_cum"] == 1.0  # cutoff 550 s: only the first effort is in
    snap = stream.compute_features_offline(samples, empty, "P1", 760.0, None, cfg)
    assert snap.features["nearmax_cum"] is None  # no vmax from past matches


def test_tick_benchmark_30_players_25hz(config_path: Path) -> None:
    cfg = load_config(config_path)
    match = generate_match(seed=11, sample_hz=25.0, cfg=cfg)
    base = match.samples.loc[(match.samples["period"] == 1) & (match.samples["t_s"] <= 2100.0)]
    ghost_ids = [str(pid) for pid in dict.fromkeys(base["player_id"])][:8]
    ghosts = []
    for pid in ghost_ids:
        g = base.loc[base["player_id"] == pid].copy()
        g["player_id"] = f"{pid}-B"
        ghosts.append(g)
    frame = pd.concat([base, *ghosts]).sort_values(["t_s", "player_id"], kind="stable")
    vmax = dict(match.vmax_mps)
    for pid in ghost_ids:
        vmax[f"{pid}-B"] = match.vmax_mps[pid]

    cols = {c: frame[c].to_numpy() for c in frame.columns}
    construct = TrackingSample.model_construct  # test setup only; skips re-validation
    batch = [
        construct(
            match_id=match.match_id,
            player_id=str(cols["player_id"][i]),
            team_id=str(cols["team_id"][i]),
            period=int(cols["period"][i]),
            t_s=float(cols["t_s"][i]),
            x_m=float(cols["x_m"][i]),
            y_m=float(cols["y_m"][i]),
            speed_mps=float(cols["speed_mps"][i]),
            source=Source.SYNTHETIC,
            detected=bool(cols["detected"][i]),
            identity_confidence=float(cols["identity_confidence"][i]),
            wall_time_ns=None,
            ingest_time_ns=None,
        )
        for i in range(frame.shape[0])
    ]
    engine = stream.StreamFeatureEngine(cfg, match.match_id, vmax)
    engine.push_samples(batch)
    for warm in (300.0, 1000.0, 1800.0):
        engine.tick(warm)
    elapsed: list[float] = []
    for k in range(15):
        start = time.perf_counter()
        snaps = engine.tick(1810.0 + k * 10.0)
        elapsed.append(time.perf_counter() - start)
        assert len(snaps) == 30
    # docs/04_ARCHITECTURE.md §5 budgets one feature tick at 200 ms for 30 players at 25 Hz;
    # the bound leaves a 5x margin for slow CI machines.
    assert float(np.median(elapsed)) <= 1.0
