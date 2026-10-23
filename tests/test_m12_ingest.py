"""M12: ingest adapters (docs/03 §2-§4, §8; docs/04 §4)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from livrisk.config import config_hash, load_config
from livrisk.contracts import (
    EventSource,
    EventType,
    Lineup,
    MatchEvent,
    PositionGroup,
    PreMatchRisk,
    Source,
    TrackingSample,
)
from livrisk.features.norms import build_norms
from livrisk.ingest.clock import DRIFT_TOLERANCE_S, ClockSync
from livrisk.ingest.sources import (
    load_skillcorner_match,
    read_events_table,
    read_tracking_table,
)
from livrisk.model.live_hazard import Priors, build_design, fit_map
from livrisk.replay.simulator import ReplaySettings, replay_match
from livrisk.synth.generator import (
    SynthTruth,
    generate_match,
    generate_person_minutes,
    match_tables,
)
from tests.conftest import DEV_SEEDS, OPEN_DATA, skip_without
from tests.replay_setup import HISTORY_SEED, norm_history

pytestmark = [pytest.mark.milestone_m12]

#: Vendor-style column names for the roundtrip export; the map is contract field -> vendor name.
VENDOR_TRACKING_MAP: dict[str, str] = {
    "player_id": "Athlete ID",
    "team_id": "Squad",
    "period": "Half",
    "t_s": "Match Time (s)",
    "x_m": "X (m)",
    "y_m": "Y (m)",
    "speed_mps": "Speed (m/s)",
    "detected": "Tracked",
    "identity_confidence": "ID Confidence",
    "wall_time_ns": "Device Timestamp (ns)",
    "ingest_time_ns": "Received Timestamp (ns)",
}

VENDOR_EVENT_MAP: dict[str, str] = {
    "period": "Period",
    "t_s": "Clock (s)",
    "type": "Event",
    "team_id": "Team",
    "player_id": "Player Out",
    "player_in_id": "Player In",
    "stoppage_min": "Added (min)",
}

#: Minimal tracking map used by the validation cases (positions plus speed).
_BASE_MAP: dict[str, str] = {
    "player_id": "Athlete",
    "team_id": "Squad",
    "period": "Half",
    "t_s": "Clock",
    "x_m": "X",
    "y_m": "Y",
    "speed_mps": "Speed",
}
_BASE_ROW: dict[str, object] = {
    "Athlete": "P1",
    "Squad": "T1",
    "Half": 1,
    "Clock": 10.0,
    "X": 1.0,
    "Y": 2.0,
    "Speed": 3.0,
}


def _write_csv(tmp_path: Path, rows: list[dict[str, object]], name: str) -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_tabular_roundtrip(tmp_path: Path) -> None:
    """A synthetic match exported to CSV and Parquet with vendor column names reads back
    exactly through the column map (match_id and source come from the call, not the file)."""
    match = generate_match(DEV_SEEDS[0], sample_hz=2.0)
    samples = match.samples
    export = samples.drop(columns=["match_id", "source"]).rename(columns=VENDOR_TRACKING_MAP)
    csv_path = tmp_path / "tracking.csv"
    parquet_path = tmp_path / "tracking.parquet"
    export.to_csv(csv_path, index=False)
    export.to_parquet(parquet_path, index=False)
    cols = [c for c in samples.columns if c not in ("match_id", "source")]
    for path, exact in ((csv_path, False), (parquet_path, True)):
        got = read_tracking_table(path, VENDOR_TRACKING_MAP, Source.WEARABLE, match.match_id)
        assert (got["match_id"] == match.match_id).all()
        assert (got["source"] == Source.WEARABLE.value).all()
        # CSV text loses the last 1-2 significant digits of a float64; Parquet is exact.
        pd.testing.assert_frame_equal(
            got[cols].reset_index(drop=True),
            samples[cols].reset_index(drop=True),
            check_exact=exact,
            rtol=1e-9,
            atol=1e-9,
        )


def test_tracking_table_validation(tmp_path: Path) -> None:
    """Out-of-range or incomplete rows and bad maps raise instead of passing bad data on."""

    def read(rows: list[dict[str, object]], mapping: dict[str, str], name: str) -> pd.DataFrame:
        return read_tracking_table(
            _write_csv(tmp_path, rows, name), mapping, Source.WEARABLE, "m-val"
        )

    with pytest.raises(ValueError, match="speed_mps must be >= 0"):
        read([{**_BASE_ROW, "Speed": -0.5}], _BASE_MAP, "neg_speed.csv")
    with pytest.raises(ValueError, match="period values must be in 1-4"):
        read([{**_BASE_ROW, "Half": 0}], _BASE_MAP, "bad_period.csv")
    with pytest.raises(ValueError, match="before the start of its period"):
        read([{**_BASE_ROW, "Half": 2, "Clock": 100.0}], _BASE_MAP, "early_t.csv")
    with pytest.raises(ValueError, match="together or neither"):
        read([_BASE_ROW], {k: v for k, v in _BASE_MAP.items() if k != "y_m"}, "only_x.csv")
    speed_only = {k: v for k, v in _BASE_MAP.items() if k not in ("x_m", "y_m")}
    with pytest.raises(ValueError, match="position or a speed"):
        read(
            [{"Athlete": "P1", "Squad": "T1", "Half": 1, "Clock": 10.0, "Speed": None}],
            speed_only,
            "no_data.csv",
        )
    with pytest.raises(ValueError, match="x_m/y_m or speed_mps"):
        read([_BASE_ROW], {k: v for k, v in speed_only.items() if k != "speed_mps"}, "no_pos.csv")
    with pytest.raises(ValueError, match="missing required fields"):
        read([_BASE_ROW], {k: v for k, v in _BASE_MAP.items() if k != "t_s"}, "no_t.csv")
    with pytest.raises(ValueError, match="must be contract fields"):
        read([_BASE_ROW], {**_BASE_MAP, "heart_rate": "HR"}, "bad_key.csv")
    with pytest.raises(ValueError, match="mapped columns not present"):
        read([_BASE_ROW], {**_BASE_MAP, "wall_time_ns": "No Such Column"}, "no_col.csv")
    with pytest.raises(ValueError, match="identity_confidence must be in"):
        read(
            [{**_BASE_ROW, "Conf": 1.5}],
            {**_BASE_MAP, "identity_confidence": "Conf"},
            "bad_ident.csv",
        )
    with pytest.raises(ValueError, match="detected must be true/false"):
        read(
            [{**_BASE_ROW, "Tracked": "maybe"}],
            {**_BASE_MAP, "detected": "Tracked"},
            "bad_det.csv",
        )


def test_events_table_roundtrip(tmp_path: Path) -> None:
    """A synthetic match's events exported with feed column names read back as MatchEvents."""
    match = generate_match(DEV_SEEDS[1], tracking=False)
    frame = match_tables(match)["events"]
    export = frame.drop(columns=["match_id", "source"]).rename(columns=VENDOR_EVENT_MAP)
    path = tmp_path / "events.csv"
    export.to_csv(path, index=False)
    got = read_events_table(path, VENDOR_EVENT_MAP, match.match_id)
    expected = sorted(match.events, key=lambda e: (e.period, e.t_s))
    assert len(got) == len(expected)
    for g, w in zip(got, expected, strict=True):
        assert g.source == EventSource.OFFICIAL_FEED
        assert g.model_dump(exclude={"source"}) == w.model_dump(exclude={"source"})


def test_events_table_rejects_unknown_types(tmp_path: Path) -> None:
    rows = [{"Period": 1, "Clock": 0.0, "Event": "foul"}]
    mapping = {"period": "Period", "t_s": "Clock", "type": "Event"}
    with pytest.raises(ValueError, match="unknown event type"):
        read_events_table(_write_csv(tmp_path, rows, "events_bad.csv"), mapping, "m1")


# --------------------------------------------------------------------------------------------
# SkillCorner open data
# --------------------------------------------------------------------------------------------


def _sc_player_json(
    pid: int,
    team: int,
    number: int,
    acronym: str,
    group: str,
    start_time: str | None,
    end_time: str | None,
    start_frame: int | None,
    end_frame: int | None,
    red_card: int = 0,
) -> dict[str, Any]:
    """One players[] entry of a SkillCorner <id>_match.json file (real field names)."""
    total = None if start_frame is None else {"start_frame": start_frame, "end_frame": end_frame}
    return {
        "player_role": {"id": 0, "position_group": group, "name": acronym, "acronym": acronym},
        "start_time": start_time,
        "end_time": end_time,
        "number": number,
        "yellow_card": 0,
        "red_card": red_card,
        "injured": False,
        "goal": 0,
        "own_goal": 0,
        "playing_time": {"total": total, "by_period": []},
        "team_id": team,
        "id": pid,
        "first_name": "A",
        "last_name": "B",
        "short_name": "A. B",
        "birthday": None,
        "trackable_object": pid,
        "gender": "male",
    }


def _write_skillcorner_dir(root: Path) -> Path:
    """A minimal SkillCorner-format match directory: two periods of 50 frames, one
    substitution (1002 off, 1003 on, frames 84/85), one red card (2002, frame 109)."""
    match_dir = root / "99001"
    match_dir.mkdir(parents=True)
    meta = {
        "id": 99001,
        "date_time": "2025-03-01T10:00:00Z",
        "home_team": {"id": 100, "name": "Home FC", "short_name": "HOM", "acronym": "HOM"},
        "away_team": {"id": 200, "name": "Away FC", "short_name": "AWY", "acronym": "AWY"},
        "match_periods": [
            {
                "period": 1,
                "name": "period_1",
                "start_frame": 5,
                "end_frame": 55,
                "duration_frames": 50,
                "duration_minutes": 0.083,
            },
            {
                "period": 2,
                "name": "period_2",
                "start_frame": 60,
                "end_frame": 110,
                "duration_frames": 50,
                "duration_minutes": 0.083,
            },
        ],
        "players": [
            _sc_player_json(1001, 100, 1, "GK", "Other", "00:00:00", None, 5, 110),
            _sc_player_json(1002, 100, 9, "CF", "Center Forward", "00:00:00", "00:45:02", 5, 84),
            _sc_player_json(1003, 100, 7, "SUB", "Other", "00:45:02", None, 85, 110),
            _sc_player_json(2001, 200, 1, "GK", "Other", "00:00:00", None, 5, 110),
            _sc_player_json(
                2002, 200, 11, "RW", "Wide Attacker", "00:00:00", "00:45:07", 5, 109, red_card=1
            ),
            _sc_player_json(2003, 200, 4, "SUB", "Other", None, None, None, None),
        ],
        "pitch_length": 105,
        "pitch_width": 68,
    }
    (match_dir / "99001_match.json").write_text(json.dumps(meta), encoding="utf-8")
    frames: dict[int, list[int]] = {}
    for pid, first, last in (
        (1001, 5, 110),
        (1002, 5, 84),
        (1003, 85, 110),
        (2001, 5, 110),
        (2002, 5, 109),
    ):
        for frame in range(first, last + 1):
            frames.setdefault(frame, []).append(pid)
    lines = []
    for frame in range(112):
        period = 1 if 5 <= frame <= 55 else (2 if 60 <= frame <= 110 else None)
        players: list[dict[str, Any]] = [
            {
                "x": (pid % 40) - 20 + 0.01 * frame,
                "y": float((pid % 17) - 8),
                "player_id": pid,
                "is_detected": not (pid == 1002 and frame == 6),
            }
            for pid in frames.get(frame, [])
        ]
        if frame == 5:
            # A position-less entry and an unknown id: both must be dropped by the loader.
            players.append({"x": None, "y": None, "player_id": 1001, "is_detected": None})
            players.append({"x": 0.0, "y": 0.0, "player_id": 9999, "is_detected": True})
        lines.append(
            json.dumps(
                {
                    "frame": frame,
                    "timestamp": None,
                    "period": period,
                    "ball_data": None,
                    "possession": None,
                    "player_data": players,
                }
            )
        )
    tracking_path = match_dir / "99001_tracking_extrapolated.jsonl"
    tracking_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return match_dir


def _validate_tracking_row(record: dict[Any, Any]) -> None:
    data = dict(record)
    for key in ("x_m", "y_m", "speed_mps", "wall_time_ns", "ingest_time_ns"):
        value = data[key]
        if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
            data[key] = None
    data["period"] = int(data["period"])
    data["t_s"] = float(data["t_s"])
    data["detected"] = bool(data["detected"])
    data["identity_confidence"] = float(data["identity_confidence"])
    TrackingSample.model_validate(data)


def test_skillcorner_minifake(tmp_path: Path) -> None:
    """A hand-written SkillCorner-format match parses into tracking, events and lineups."""
    tracking, events, lineups = load_skillcorner_match(_write_skillcorner_dir(tmp_path))

    assert set(tracking["player_id"].astype(str)) == {"1001", "1002", "1003", "2001", "2002"}
    # Frames 56-59 carry no period (the break) and are dropped: 106-4, 80-4, 26, 106-4, 105-4.
    counts = tracking.groupby("player_id").size()
    assert counts.to_dict() == {"1001": 102, "1002": 76, "1003": 26, "2001": 102, "2002": 101}
    assert (tracking["source"] == Source.BROADCAST.value).all()
    assert bool(tracking["speed_mps"].isna().all())
    assert (tracking["identity_confidence"] == 1.0).all()
    assert bool(tracking["wall_time_ns"].isna().all())
    assert tracking["t_s"].min() == 0.0
    second = tracking.loc[tracking["period"] == 2]
    assert second["t_s"].min() == 2700.0
    assert second["t_s"].max() == 2705.0
    row = tracking.loc[(tracking["player_id"] == "1002") & (tracking["t_s"] == 0.1)]
    assert not bool(row["detected"].iloc[0])
    start = tracking.loc[(tracking["player_id"] == "1001") & (tracking["t_s"] == 0.0)]
    assert float(start["x_m"].iloc[0]) == pytest.approx(-18.95)
    assert float(start["y_m"].iloc[0]) == pytest.approx(7.0)
    for record in tracking.to_dict("records"):
        _validate_tracking_row(record)

    clock = {(e.type, e.period): e.t_s for e in events}
    assert clock[(EventType.KICKOFF, 1)] == 0.0
    assert clock[(EventType.PERIOD_END, 1)] == 5.0
    assert clock[(EventType.KICKOFF, 2)] == 2700.0
    assert clock[(EventType.PERIOD_END, 2)] == 2705.0
    subs = [e for e in events if e.type == EventType.SUBSTITUTION]
    assert len(subs) == 1
    assert (subs[0].period, subs[0].player_id, subs[0].player_in_id) == (2, "1002", "1003")
    assert subs[0].t_s == pytest.approx(2702.5)
    reds = [e for e in events if e.type == EventType.RED_CARD]
    assert len(reds) == 1
    assert (reds[0].period, reds[0].player_id) == (2, "2002")
    assert reds[0].t_s == pytest.approx(2704.9)
    assert all(e.source == EventSource.OFFICIAL_FEED for e in events)

    assert [lu.team_id for lu in lineups] == ["100", "200"]
    assert [sum(p.starting for p in lu.players) for lu in lineups] == [2, 2]
    by_id = {p.player_id: p for lu in lineups for p in lu.players}
    assert by_id["1001"].position_group == PositionGroup.GK
    assert by_id["1002"].position_group == PositionGroup.FW
    assert by_id["1003"].position_group == PositionGroup.MF  # unlisted role falls back to MF
    assert by_id["2002"].position_group == PositionGroup.W
    assert by_id["2003"].starting is False
    assert lineups[0].kickoff_utc == datetime(2025, 3, 1, 10, 0, tzinfo=UTC)


@dataclass(frozen=True)
class _OpenMatch:
    """MatchRecord view of a loaded SkillCorner match for the replay simulator."""

    match_id: str
    lineups: tuple[Lineup, ...]
    events: tuple[MatchEvent, ...]
    samples: pd.DataFrame
    prematch: tuple[PreMatchRisk, ...]
    vmax_mps: dict[str, float]


@pytest.mark.needs_open_data
@skip_without(OPEN_DATA / "skillcorner")
def test_skillcorner_match_end_to_end(config_path: Path) -> None:
    """One real open-data match loads, validates and replays through the whole live path.

    The replay is a plumbing check on real broadcast tracking, not a risk claim: pre-match
    risks are a fixed 0.025 for every player and v_max a fixed default, because no weekly
    model output or player history exists for these matches.
    """
    matches_root = OPEN_DATA / "skillcorner" / "data" / "matches"
    match_dirs = (
        sorted(p for p in matches_root.iterdir() if p.is_dir()) if matches_root.is_dir() else []
    )
    if not match_dirs:
        pytest.skip(f"no SkillCorner matches under {matches_root}")
    tracking, events, lineups = load_skillcorner_match(match_dirs[0])
    match_id = str(tracking["match_id"].iloc[0])

    assert tracking.shape[0] > 500_000
    for rec in tracking.sample(500, random_state=0).to_dict("records"):
        _validate_tracking_row(rec)
    pid = str(tracking["player_id"].value_counts().idxmax())
    one = tracking.loc[tracking["player_id"] == pid]
    dts = np.diff(np.sort(one["t_s"].to_numpy(dtype=np.float64)))
    assert float(np.median(dts)) == pytest.approx(0.1)  # 10 fps
    types = {(e.type, e.period) for e in events}
    assert (EventType.KICKOFF, 1) in types and (EventType.KICKOFF, 2) in types
    assert (EventType.PERIOD_END, 1) in types and (EventType.PERIOD_END, 2) in types
    assert any(e.type == EventType.SUBSTITUTION for e in events)
    assert len(lineups) == 2
    assert all(sum(p.starting for p in lu.players) == 11 for lu in lineups)

    cfg = load_config(config_path)
    truth = SynthTruth(p7_mean=0.10)
    minutes = generate_person_minutes(3000, seed=3, truth=truth, informative_censoring=False)
    fitted = fit_map(
        build_design(minutes, truth.feature_names, truth.match_share),
        Priors.from_config(cfg),
        config_hash=config_hash(cfg),
        version="open-data-replay",
    )
    history_match = generate_match(HISTORY_SEED, cfg=cfg)
    history = norm_history(history_match, cfg)
    kickoff_day = lineups[0].kickoff_utc.date()
    history["match_date"] = history["match_date"].map(lambda d: d - timedelta(days=800))
    norm_positions = {
        p.player_id: p.position_group for lu in history_match.lineups for p in lu.players
    }
    norms = build_norms(history, norm_positions, kickoff_day, cfg)
    players = [p for lu in lineups for p in lu.players]
    record = _OpenMatch(
        match_id=match_id,
        lineups=lineups,
        events=tuple(events),
        samples=tracking,
        prematch=tuple(
            PreMatchRisk(
                player_id=p.player_id,
                as_of_date=kickoff_day - timedelta(days=1),
                p7=0.025,
                model_version="open-data-default",
            )
            for p in players
        ),
        vmax_mps={p.player_id: float(cfg.synth.vmax_mean_mps) for p in players},
    )
    result = replay_match(record, fitted, cfg, ReplaySettings(latency_s=5.0, seed=1), norms=norms)
    assert result.n_lost == 0
    # Samples played in the final latency-plus-tick window arrive after the last tick and are
    # never delivered; everything before that window must reach the feature engine.
    tail_s = float(cfg.engine.feature_tick_s) + 5.0
    tail = int((tracking["t_s"] > tracking["t_s"].max() - tail_s).sum())
    assert result.delivered.shape[0] >= tracking.shape[0] - tail
    assert result.features.shape[0] > 0
    assert result.risks.shape[0] > 1000
    assert result.risks["player_id"].nunique() >= 20
    live = result.risks.loc[result.risks["quality"].isin(["good", "degraded"])]
    assert live.shape[0] > 0


def test_clock_drift_flags_degraded() -> None:
    """docs/04 §4: drift beyond 2 s flags the stream degraded until it is resynced."""
    assert DRIFT_TOLERANCE_S == 2.0
    clock = ClockSync()
    with pytest.raises(ValueError, match="before sync"):
        clock.observe(0.0, 0.0)
    clock.sync(0.0, 1_000.0)  # kick-off: the source reads 1000 s at match t_s 0
    assert clock.offset_s == 1_000.0
    assert clock.observe(600.0, 1_601.5) == pytest.approx(1.5)
    assert not clock.degraded
    assert clock.observe(900.0, 1_902.0) == pytest.approx(2.0)
    assert not clock.degraded  # exactly at the tolerance: not yet degraded
    clock.observe(1_200.0, 2_203.1)
    assert clock.drift_s == pytest.approx(3.1)
    assert clock.degraded
    clock.observe(1_500.0, 2_501.0)  # back within tolerance: still degraded
    assert clock.drift_s == pytest.approx(1.0)
    assert clock.degraded
    clock.sync(2_700.0, 3_701.0)  # resync at the second-half kick-off
    assert not clock.degraded
    assert clock.observe(3_300.0, 4_301.4) == pytest.approx(0.4)
    assert not clock.degraded
