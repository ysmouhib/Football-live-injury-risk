from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from livrisk.alerts.logic import AlertLogic
from livrisk.config import Config, config_hash, load_config
from livrisk.contracts import (
    PERIOD_START_S,
    EventType,
    FeatureSnapshot,
    PositionGroup,
    Quality,
    RiskSnapshot,
)
from livrisk.engine.risk_engine import MatchState, RiskEngine
from livrisk.features.norms import NormTable, build_norms
from livrisk.features.stream import StreamFeatureEngine, compute_features_offline
from livrisk.model.live_hazard import FittedHazard, Priors, build_design, fit_map
from livrisk.replay.simulator import (
    GREY,
    HIDDEN,
    LookAheadError,
    LookAheadGuard,
    ReplayResult,
    ReplaySettings,
    display_state,
    replay_match,
)
from livrisk.storage.logstore import LogStore, records_frame
from livrisk.synth.generator import (
    SynthMatch,
    SynthTruth,
    generate_match,
    generate_person_minutes,
    match_tables,
)
from tests.conftest import DEFAULT_CONFIG

pytestmark = [pytest.mark.milestone_m9]

HISTORY_SEED = 201
MATCH_SEED = 203
#: The acceptance scenario of docs/05_VALIDATION.md §3.
SETTINGS = ReplaySettings(latency_s=30.0, dropout_rate=0.05, burst_s=30.0, seed=1)
#: Half-time break the simulator puts on the wall clock (docs/04: kick-off of period 2).
HALF_TIME_S = 900.0
NS = 1_000_000_000
#: (band, period, band start minute) of the six regular bands.
REGULAR_BANDS = ((0, 1, 0.0), (1, 1, 15.0), (2, 1, 30.0), (4, 2, 45.0), (5, 2, 60.0), (6, 2, 75.0))
NUMBERS = ("rr", "p15", "r_rest", "r_rest_lo", "r_rest_hi", "r_usual", "reason_code")

IntArray = NDArray[np.int64]


def _norm_history(match: SynthMatch, cfg: Config) -> pd.DataFrame:
    """Raw feature values of one past match at the ends of three 5-minute windows per band."""
    events = match_tables(match)["events"]
    p1_end = float(
        events.loc[(events["type"] == "period_end") & (events["period"] == 1), "t_s"].iloc[0]
    )
    starts = {1: 0.0, 2: p1_end}
    day = match.lineups[0].kickoff_utc.date()
    rows: list[tuple[str, object, int, str, float]] = []
    for pid, frame in match.samples.groupby("player_id"):
        for band, period, b0 in REGULAR_BANDS:
            for w in (5.0, 10.0, 15.0):
                end_u = starts[period] + ((b0 + w) * 60.0 - PERIOD_START_S[period])
                snap = compute_features_offline(
                    frame,
                    events,
                    str(pid),
                    end_u + cfg.engine.blanking_delta_s,
                    match.vmax_mps.get(str(pid)),
                    cfg,
                )
                for feat, value in snap.features.items():
                    if value is not None and snap.coverage[feat] >= cfg.quality.min_coverage:
                        rows.append((str(pid), day, band, feat, value))
    return pd.DataFrame(rows, columns=["player_id", "match_date", "band", "feature", "value"])


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def fitted(cfg: Config) -> FittedHazard:
    truth = SynthTruth(p7_mean=0.10)
    df = generate_person_minutes(3000, seed=3, truth=truth, informative_censoring=False)
    return fit_map(
        build_design(df, truth.feature_names, truth.match_share),
        Priors.from_config(cfg),
        config_hash=config_hash(cfg),
        version="replay-test",
    )


@pytest.fixture(scope="module")
def match(cfg: Config) -> SynthMatch:
    return generate_match(MATCH_SEED, cfg=cfg)


@pytest.fixture(scope="module")
def norms(cfg: Config, match: SynthMatch) -> NormTable:
    history = _norm_history(generate_match(HISTORY_SEED, cfg=cfg), cfg)
    positions = {p.player_id: p.position_group for lu in match.lineups for p in lu.players}
    return build_norms(history, positions, match.lineups[0].kickoff_utc.date(), cfg)


@dataclass(frozen=True)
class FullRun:
    result: ReplayResult
    elapsed_s: float
    store: LogStore


@pytest.fixture(scope="module")
def full_run(
    cfg: Config,
    fitted: FittedHazard,
    norms: NormTable,
    match: SynthMatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> FullRun:
    store = LogStore(
        tmp_path_factory.mktemp("logs"),
        match.match_id,
        config_hash=config_hash(cfg),
        match_date=match.lineups[0].kickoff_utc.date(),
    )
    t0 = time.perf_counter()
    result = replay_match(match, fitted, cfg, SETTINGS, norms=norms, store=store)
    return FullRun(result, time.perf_counter() - t0, store)


class Mirror:
    """Independent view of what had been delivered by a given wall time."""

    def __init__(self, match: SynthMatch, result: ReplayResult) -> None:
        self.kickoff_ns = int(pd.Timestamp(match.lineups[0].kickoff_utc).value)
        ends = {e.period: e.t_s for e in match.events if e.type == EventType.PERIOD_END}
        self.p1_end_u = ends[1] - PERIOD_START_S[1]
        self.end_u = self.p1_end_u + ends[2] - PERIOD_START_S[2]
        self.starters = {p.player_id for lu in match.lineups for p in lu.players if p.starting}
        d = result.delivered
        self.samples = {str(pid): g for pid, g in d.groupby("player_id")}
        self.by_player: dict[str, tuple[IntArray, IntArray]] = {
            pid: (
                g["ingest_time_ns"].to_numpy(dtype=np.int64),
                g["wall_time_ns"].to_numpy(dtype=np.int64),
            )
            for pid, g in self.samples.items()
        }
        self.events = result.delivered_events
        subs = self.events.loc[self.events["type"] == "substitution"]
        #: player coming on -> (ingest, production) wall time of his substitution event
        self.sub_in: dict[str, tuple[int, int]] = {
            str(r["player_in_id"]): (int(r["ingest_time_ns"]), int(r["wall_time_ns"]))
            for r in subs.to_dict("records")
        }

    def tick_wall_ns(self, u: float) -> int:
        brk = HALF_TIME_S if u > self.p1_end_u + 1e-9 else 0.0
        return self.kickoff_ns + round((u + brk) * NS)

    def events_by(self, wall_ns: int) -> pd.DataFrame:
        return self.events.loc[self.events["ingest_time_ns"] <= wall_ns]

    def samples_by(self, pid: str, wall_ns: int) -> pd.DataFrame:
        g = self.samples[pid]
        return g.loc[g["ingest_time_ns"].to_numpy(dtype=np.int64) <= wall_ns]

    def age_s(self, pid: str, wall_ns: int) -> float:
        newest = -1
        if pid in self.by_player:
            ingest, wall = self.by_player[pid]
            k = int(np.searchsorted(ingest, wall_ns, side="right"))
            if k:
                newest = int(np.max(wall[:k]))
        entry = self.kickoff_ns if pid in self.starters else -1
        if pid in self.sub_in and self.sub_in[pid][0] <= wall_ns:
            entry = self.sub_in[pid][1]
        return max(0.0, (wall_ns - max(newest, entry)) / NS)

    def state(self, match_id: str, u: float, wall_ns: int) -> MatchState:
        ev = self.events_by(wall_ns)
        second_half = bool(((ev["type"] == "kickoff") & (ev["period"] == 2)).any())
        stoppage = {
            int(r["period"]): float(r["stoppage_min"])
            for r in ev.loc[ev["type"] == "stoppage_announced"].to_dict("records")
        }
        if second_half:
            p1_len = float(
                ev.loc[(ev["type"] == "period_end") & (ev["period"] == 1), "t_s"].iloc[0]
            )
            period, t_s = 2, 2700.0 + (u - p1_len)
        else:
            period, t_s = 1, u
        return MatchState(match_id, period, t_s, stoppage.get(1), stoppage.get(2))


def _assert_rows_equal(
    replayed: pd.Series[Any], offline: pd.Series[Any], columns: list[str]
) -> None:
    """Field-wise equality to 1e-9; None and NaN both mean missing (frames differ in dtype)."""
    for col in columns:
        a, b = replayed[col], offline[col]
        a_missing = a is None or (isinstance(a, float) and math.isnan(a))
        b_missing = b is None or (isinstance(b, float) and math.isnan(b))
        assert a_missing == b_missing, col
        if a_missing:
            continue
        if isinstance(a, float | int | np.floating) and not isinstance(a, bool):
            assert abs(float(a) - float(b)) <= 1e-9, col
        else:
            assert a == b, col


def test_replay_equals_offline(
    cfg: Config, fitted: FittedHazard, norms: NormTable, match: SynthMatch, full_run: FullRun
) -> None:
    res = full_run.result
    mirror = Mirror(match, res)
    positions: dict[str, PositionGroup] = {
        p.player_id: p.position_group for lu in match.lineups for p in lu.players
    }
    offline_engine = RiskEngine(
        cfg, fitted, norms, {r.player_id: r for r in match.prematch}, positions
    )
    compare = [
        c
        for c in res.features.columns
        if c.startswith(("features.", "coverage."))
        or c in ("period", "cutoff_t_s", "identity_confidence", "quality", "quality_reasons")
    ]
    risk_cols = [
        "period",
        "as_of_t_s",
        "data_age_s",
        "quality",
        "p15",
        "p15_lo",
        "p15_hi",
        "r_rest",
        "r_rest_lo",
        "r_rest_hi",
        "r_usual",
        "rr",
        "prematch_band",
        "top_reason_code",
    ]
    n_features = 0
    n_risks = 0
    for u in np.arange(360.0, mirror.end_u, 360.0):
        wall = mirror.tick_wall_ns(float(u))
        events = mirror.events_by(wall)
        rows = res.features.loc[res.features["as_of_t_s"] == u]
        assert rows.shape[0] >= 20
        snaps: list[FeatureSnapshot] = []
        for _, row in rows.iterrows():
            pid = str(row["player_id"])
            frame = mirror.samples_by(pid, wall) if pid in mirror.samples else match.samples[:0]
            snap = compute_features_offline(
                frame, events, pid, float(u), match.vmax_mps.get(pid), cfg
            )
            _assert_rows_equal(row, records_frame([snap]).iloc[0], compare)
            snaps.append(snap)
            n_features += 1
        state = mirror.state(match.match_id, float(u), wall)
        ages = {s.player_id: mirror.age_s(s.player_id, wall) for s in snaps}
        offline = records_frame(list(offline_engine.update(snaps, state, ages).risks))
        replayed = res.risks.loc[
            (res.risks["period"] == state.period) & (res.risks["as_of_t_s"] == state.t_s)
        ]
        assert replayed["player_id"].tolist() == offline["player_id"].tolist()
        for (_, a), (_, b) in zip(replayed.iterrows(), offline.iterrows(), strict=True):
            _assert_rows_equal(a, b, risk_cols)
            n_risks += 1
    assert n_features > 300
    assert n_risks > 250


def _risk_rows(frame: pd.DataFrame) -> list[RiskSnapshot]:
    out: list[RiskSnapshot] = []
    for rec in frame.to_dict("records"):
        data = {name: rec[name] for name in RiskSnapshot.model_fields}
        if pd.isna(data["top_reason_code"]):
            data["top_reason_code"] = None
        out.append(RiskSnapshot.model_validate(data))
    return out


def test_alerts_follow_the_replayed_risk_rows(
    cfg: Config, match: SynthMatch, full_run: FullRun
) -> None:
    res = full_run.result
    feats = res.features
    team_of = {p.player_id: lu.team_id for lu in match.lineups for p in lu.players}
    teams = sorted({lu.team_id for lu in match.lineups})
    logic = {t: AlertLogic(cfg, match.match_id, id_prefix=f"{t}-") for t in teams}
    expected: list[tuple[str, str, str, float, float | None]] = []
    seen: set[str] = set()
    for (period, t_s), group in res.risks.groupby(["period", "as_of_t_s"], sort=True):
        tick = feats.loc[(feats["match_period"] == period) & (feats["match_t_s"] == t_s)]
        on_pitch = set(tick["player_id"].astype(str))
        rows = _risk_rows(group)
        for team in teams:
            team_rows = [r for r in rows if team_of[r.player_id] == team]
            team_on = {p for p in on_pitch if team_of[p] == team}
            for alert in logic[team].update(team_rows, team_on):
                if alert.cleared_t_s is not None:
                    change = "cleared"
                elif alert.alert_id in seen:
                    change = "upgraded"
                else:
                    change = "opened"
                    seen.add(alert.alert_id)
                expected.append(
                    (alert.alert_id, alert.level.value, change, alert.opened_t_s, alert.cleared_t_s)
                )
    got = [
        (
            str(r["alert_id"]),
            str(r["level"]),
            str(r["change"]),
            float(r["opened_t_s"]),
            None if pd.isna(r["cleared_t_s"]) else float(r["cleared_t_s"]),
        )
        for r in res.alerts.to_dict("records")
    ]
    assert got == expected
    assert any(change == "opened" for _, _, change, _, _ in got)
    assert {a.split("-")[0] for a, *_ in got} <= set(teams)


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def test_stale_rules_hold_under_latency_and_dropout(
    cfg: Config, match: SynthMatch, full_run: FullRun
) -> None:
    res = full_run.result
    mirror = Mirror(match, res)
    shown = res.displayed
    assert shown.shape[0] > 10_000
    grey, hide = float(cfg.quality.stale_grey_s), float(cfg.quality.stale_hide_s)
    delay_ns = round(SETTINGS.push_delay_s * NS)
    for rec in shown.to_dict("records"):
        wall = mirror.tick_wall_ns(float(rec["play_s"]))
        assert int(rec["shown_wall_ns"]) == wall + delay_ns
        age = mirror.age_s(str(rec["player_id"]), wall) + SETTINGS.push_delay_s
        assert abs(float(rec["data_age_s"]) - age) <= 1e-6
        assert age >= SETTINGS.latency_s + SETTINGS.push_delay_s - 1e-9
        assert rec["state"] == display_state(age, grey, hide)
        if rec["state"] == HIDDEN:
            assert all(_missing(rec[c]) for c in NUMBERS)
        else:
            assert not _missing(rec["r_rest"])
        if rec["quality"] in (Quality.PREMATCH_ONLY.value, Quality.STALE.value):
            assert _missing(rec["rr"])
    stale = res.risks.loc[res.risks["data_age_s"] > hide]
    assert (stale["quality"] == Quality.STALE.value).all()


def test_replay_speed(match: SynthMatch, full_run: FullRun) -> None:
    mirror = Mirror(match, full_run.result)
    assert mirror.end_u >= 5400.0
    assert full_run.elapsed_s <= mirror.end_u / 50.0, full_run.elapsed_s


def test_log_store_holds_everything_shown(cfg: Config, full_run: FullRun) -> None:
    res, store = full_run.result, full_run.store
    assert store.read("tracking").shape[0] == res.delivered.shape[0]
    assert store.read("events").shape[0] == res.delivered_events.shape[0]
    assert store.read("risk").shape[0] == res.risks.shape[0]
    assert store.read("alerts").shape[0] == res.alerts.shape[0]
    assert store.read("display").shape[0] == res.displayed.shape[0]
    stages = store.read("features")["stage"].value_counts()
    assert stages["feature_tick"] == res.features.shape[0]
    assert stages["risk_tick"] == res.risks.shape[0]  # one z-enriched snapshot per risk row
    for kind in ("tracking", "risk", "display"):
        assert (store.read(kind)["config_hash"] == config_hash(cfg)).all()
    n = store.sql("select count(*) as n from risk where quality = 'good'")["n"].iloc[0]
    assert n == int((res.risks["quality"] == "good").sum())
    assert len(list((store.directory / "tracking").glob("part-*.parquet"))) > 1


class PeekingEngine(StreamFeatureEngine):
    """A deliberately leaky feature stage: it computes every snapshot 30 s ahead."""

    def tick(self, as_of_t_s: float) -> list[FeatureSnapshot]:
        return super().tick(as_of_t_s + 30.0)


def test_leaky_feature_raises_lookahead_error(
    cfg: Config, fitted: FittedHazard, norms: NormTable, match: SynthMatch
) -> None:
    settings = ReplaySettings(until_s=300.0)
    with pytest.raises(LookAheadError):
        replay_match(match, fitted, cfg, settings, norms=norms, feature_engine=PeekingEngine)


def test_guard_allows_reads_up_to_the_cutoff_only() -> None:
    guard = LookAheadGuard(delta_s=60.0)
    with pytest.raises(LookAheadError):
        guard.check_read("P1", -1e6)
    guard.advance(600.0)
    guard.check_read("P1", 540.0)
    with pytest.raises(LookAheadError):
        guard.check_read("P1", 540.5)


def test_feed_outage_greys_then_hides_values(
    cfg: Config, fitted: FittedHazard, norms: NormTable, match: SynthMatch
) -> None:
    settings = ReplaySettings(latency_s=5.0, burst_s=600.0, until_s=1500.0, seed=2)
    res = replay_match(match, fitted, cfg, settings, norms=norms)
    states = set(res.displayed["state"])
    assert {GREY, HIDDEN} <= states
    hidden = res.displayed.loc[res.displayed["state"] == HIDDEN]
    assert hidden[list(NUMBERS)].isna().all().all()
    stale = res.risks.loc[res.risks["quality"] == Quality.STALE.value]
    assert stale.shape[0] > 0
    assert (stale["data_age_s"] > cfg.quality.stale_hide_s).all()
    assert np.allclose(stale["rr"], 1.0)
    assert (stale["top_reason_code"] == "DATA_LOST").all()


def test_short_replays_are_deterministic_and_inject_drift_and_overflow(
    cfg: Config, fitted: FittedHazard, norms: NormTable, match: SynthMatch
) -> None:
    settings = ReplaySettings(
        latency_s=8.0,
        dropout_rate=0.05,
        clock_drift_s=4.0,
        queue_capacity=1500,
        until_s=420.0,
        seed=5,
    )
    a = replay_match(match, fitted, cfg, settings, norms=norms)
    b = replay_match(match, fitted, cfg, settings, norms=norms)
    for name in ("features", "risks", "alerts", "displayed", "delivered", "delivered_events"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))
    assert a.n_queue_dropped > 0
    assert a.n_lost > 0
    mirror = Mirror(match, a)
    d = a.delivered.loc[a.delivered["period"] == 1]
    true_u = (d["wall_time_ns"].to_numpy(dtype=np.int64) - mirror.kickoff_ns) / NS
    drift = d["t_s"].to_numpy(dtype=np.float64) - true_u
    assert np.allclose(drift, settings.clock_drift_s * true_u / mirror.end_u, atol=1e-6)


def test_paced_replay_takes_its_time(
    cfg: Config, fitted: FittedHazard, norms: NormTable, match: SynthMatch
) -> None:
    settings = ReplaySettings(speed=300.0, until_s=90.0)
    t0 = time.perf_counter()
    replay_match(match, fitted, cfg, settings, norms=norms)
    assert time.perf_counter() - t0 >= 90.0 / 300.0


def test_log_store_is_append_only(tmp_path: Path) -> None:
    from livrisk.contracts import DecisionRecord

    rows = [DecisionRecord(alert_id="T1-A1", t_s=600.0, actor_role="physio", action="physio_check")]
    store = LogStore(
        tmp_path, "m1", config_hash="abc", match_date=pd.Timestamp("2026-05-01").date()
    )
    store.append("decisions", rows)
    store.flush()
    first = store.directory / "decisions" / "part-00000.parquet"
    before = first.read_bytes()
    store.append("decisions", rows, note_len=[3])
    store.flush()
    store.flush()
    assert first.read_bytes() == before
    assert len(list((store.directory / "decisions").glob("part-*.parquet"))) == 2
    again = LogStore(
        tmp_path, "m1", config_hash="abc", match_date=pd.Timestamp("2026-05-01").date()
    )
    again.append("decisions", rows)
    again.flush()
    table = again.read("decisions")
    assert table.shape[0] == 3
    assert (table["config_hash"] == "abc").all()
    with pytest.raises(ValueError):
        store.append("notes", rows)
    with pytest.raises(ValueError):
        store.append("decisions", rows, extra=[1, 2])
