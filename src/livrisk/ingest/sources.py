"""Concrete source adapters (docs/03 §2-§4, §8; docs/04 §4).

Three loaders produce the contract shapes of docs/03 from vendor files:

- ``read_tracking_table``: a CSV or Parquet wearable/optical export with vendor column names,
  described by a column map.
- ``read_events_table``: an official-feed event export with the same mapping idea.
- ``load_skillcorner_match``: the SkillCorner open-data match format (10 fps broadcast tracking,
  ``is_detected`` mapped to ``detected``, source ``broadcast``).

All three validate the contract ranges of docs/03 §2-§3 and raise ``ValueError`` on
out-of-range data rather than passing bad rows downstream. The returned frames and events carry
only their sample and event times: the no-look-ahead rule (features at ``t`` read data with
``t_s <= t - delta``) is applied downstream by the feature engine, not here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from livrisk.contracts import (
    PERIOD_START_S,
    EventSource,
    EventType,
    Lineup,
    LineupPlayer,
    MatchEvent,
    PositionGroup,
    Source,
    TrackingSample,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ObjectArray = NDArray[np.object_]

#: Contract column order of a tracking frame (docs/03 §2).
TRACKING_COLUMNS: tuple[str, ...] = tuple(TrackingSample.model_fields)

#: Sampling rate of the SkillCorner open-data tracking, frames per second (docs/03 §8).
SKILLCORNER_FPS: float = 10.0

#: Tolerance when pairing a substitution's entering frame with the leaver's last frame, frames.
_SUB_PAIR_FRAME_TOL: int = 30

_TRACKING_FIELDS: frozenset[str] = frozenset(TRACKING_COLUMNS) - {"match_id", "source"}
_REQUIRED_TRACKING: tuple[str, ...] = ("player_id", "team_id", "period", "t_s")
_EVENT_FIELDS: frozenset[str] = frozenset(MatchEvent.model_fields) - {"match_id", "source"}
_REQUIRED_EVENTS: tuple[str, ...] = ("period", "t_s", "type")

_TRUE_WORDS = frozenset({"true", "1", "yes"})
_FALSE_WORDS = frozenset({"false", "0", "no"})


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format {suffix!r}: use CSV or Parquet")


def _apply_column_map(
    df: pd.DataFrame, column_map: dict[str, str], allowed: frozenset[str], required: tuple[str, ...]
) -> pd.DataFrame:
    """Rename vendor columns to contract fields (``column_map``: contract -> vendor column)."""
    unknown = sorted(set(column_map) - allowed)
    if unknown:
        raise ValueError(
            f"column_map keys must be contract fields from {sorted(allowed)}; got {unknown} "
            "(match_id and source come from the arguments)"
        )
    missing = [name for name in required if name not in column_map]
    if missing:
        raise ValueError(f"column_map is missing required fields: {missing}")
    absent = sorted({vendor for vendor in column_map.values() if vendor not in df.columns})
    if absent:
        raise ValueError(f"mapped columns not present in the file: {absent}")
    out = df.rename(columns={vendor: field for field, vendor in column_map.items()})
    if out.columns.duplicated().any():
        raise ValueError("column_map collides with an existing column name")
    return out


def _floats(series: pd.Series, name: str) -> FloatArray:
    """Column as float64, NaN for missing; raises on non-numeric content."""
    return np.asarray(
        pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan),
        dtype=np.float64,
    )


def _whole_ints(series: pd.Series, name: str) -> IntArray:
    """Column as int64; raises on missing or non-whole values."""
    values = _floats(series, name)
    if not bool(np.all(np.isfinite(values))) or not bool(np.all(values == np.round(values))):
        raise ValueError(f"{name} must hold whole numbers")
    return values.astype(np.int64)


def _id_strings(series: pd.Series, name: str) -> ObjectArray:
    """ID column as strings; whole float ids (123.0) become "123"; missing values raise."""
    if bool(series.isna().any()):
        raise ValueError(f"{name} must not be missing")
    if pd.api.types.is_float_dtype(series.dtype):
        values = series.to_numpy(dtype=np.float64)
        if bool(np.all(values == np.round(values))):
            return np.asarray([str(int(v)) for v in values], dtype=object)
    return series.astype(str).to_numpy(dtype=object)


def _parse_detected(series: pd.Series) -> BoolArray:
    """Detection flag from a bool, 0/1 or word column; blank cells keep the default True."""
    words = series.astype("string").str.strip().str.lower()
    known = words.isin(_TRUE_WORDS | _FALSE_WORDS) | words.isna()
    if not bool(known.all()):
        bad = sorted({str(w) for w in words[~known]})
        raise ValueError(f"detected must be true/false (or 1/0); got {bad}")
    return (words.isna() | words.isin(_TRUE_WORDS)).to_numpy(dtype=np.bool_)


def _ns_column(df: pd.DataFrame, name: str) -> pd.arrays.IntegerArray:
    """Wall/ingest nanosecond column as nullable Int64; all-missing when unmapped.

    CSV readers hand back float64 for a column with gaps, which cannot hold 19-digit nanosecond
    timestamps exactly; export Parquet when the wall-clock values matter.
    """
    n = df.shape[0]
    if name not in df.columns:
        return pd.arrays.IntegerArray(np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.bool_))
    values = _floats(df[name], name)
    finite = values[np.isfinite(values)]
    if finite.size and not bool(np.all(finite == np.round(finite))):
        raise ValueError(f"{name} must hold whole nanosecond timestamps")
    mask = ~np.isfinite(values)
    return pd.arrays.IntegerArray(np.where(mask, 0, values).astype(np.int64), mask)


def _check_tracking_frame(df: pd.DataFrame) -> None:
    """Contract ranges of docs/03 §2 on a tracking frame; raises ValueError on any violation."""
    period = df["period"].to_numpy(dtype=np.int64)
    if not bool(np.isin(period, [1, 2, 3, 4]).all()):
        raise ValueError("period values must be in 1-4")
    t_s = df["t_s"].to_numpy(dtype=np.float64)
    if not bool(np.all(np.isfinite(t_s))):
        raise ValueError("t_s must not be missing")
    starts = np.asarray([PERIOD_START_S[int(p)] for p in period])
    if bool(np.any(t_s < starts)):
        raise ValueError("t_s before the start of its period")
    speed = df["speed_mps"].to_numpy(dtype=np.float64, na_value=np.nan)
    if bool(np.any(speed < 0.0)):
        raise ValueError("speed_mps must be >= 0 (m/s)")
    ident = df["identity_confidence"].to_numpy(dtype=np.float64, na_value=np.nan)
    finite = ident[np.isfinite(ident)]
    if finite.size and bool(np.any((finite < 0.0) | (finite > 1.0))):
        raise ValueError("identity_confidence must be in [0, 1]")
    has_x = np.isfinite(df["x_m"].to_numpy(dtype=np.float64, na_value=np.nan))
    has_y = np.isfinite(df["y_m"].to_numpy(dtype=np.float64, na_value=np.nan))
    if bool(np.any(has_x != has_y)):
        raise ValueError("x_m and y_m must both be set or both be missing")
    if bool(np.any(~has_x & ~np.isfinite(speed))):
        raise ValueError("every sample needs a position or a speed (docs/03 §2)")


def read_tracking_table(
    path: str | Path,
    column_map: dict[str, str],
    source: Source,
    match_id: str,
) -> pd.DataFrame:
    """CSV or Parquet tracking export to a ``TrackingSample`` frame (docs/03 §2).

    ``column_map`` maps contract field names to the vendor's column names. Required fields are
    ``player_id``, ``team_id``, ``period`` and ``t_s``, plus ``x_m``/``y_m`` together or
    ``speed_mps``; ``match_id`` and ``source`` come from the arguments, not the file.
    ``detected`` defaults to True and ``identity_confidence`` to 1.0 where unmapped (blank cells
    keep the default). Units are the contract's: metres, m/s, seconds on the match clock with
    period starts 0/2700/5400/6300 s; rows failing the ranges of docs/03 §2 raise ValueError.
    Rows are sorted by (period, t_s, player_id). The frame carries only sample times: the as_of
    rule (features at ``t`` use data with ``t_s <= t - delta``) is applied downstream.
    """
    df = _read_table(Path(path))
    df = _apply_column_map(df, column_map, _TRACKING_FIELDS, _REQUIRED_TRACKING)
    mapped = set(column_map)
    if ("x_m" in mapped) != ("y_m" in mapped):
        raise ValueError("column_map must provide x_m and y_m together or neither")
    if "x_m" not in mapped and "speed_mps" not in mapped:
        raise ValueError("column_map must provide x_m/y_m or speed_mps")
    n = df.shape[0]
    period = _whole_ints(df["period"], "period")
    x_m = _floats(df["x_m"], "x_m") if "x_m" in df.columns else np.full(n, np.nan)
    y_m = _floats(df["y_m"], "y_m") if "y_m" in df.columns else np.full(n, np.nan)
    speed = (
        _floats(df["speed_mps"], "speed_mps") if "speed_mps" in df.columns else np.full(n, np.nan)
    )
    if "identity_confidence" in df.columns:
        ident = _floats(df["identity_confidence"], "identity_confidence")
        ident = np.where(np.isfinite(ident), ident, 1.0)
    else:
        ident = np.ones(n, dtype=np.float64)
    detected = (
        _parse_detected(df["detected"]) if "detected" in df.columns else np.ones(n, dtype=np.bool_)
    )
    out = pd.DataFrame(
        {
            "match_id": np.full(n, match_id, dtype=object),
            "player_id": _id_strings(df["player_id"], "player_id"),
            "team_id": _id_strings(df["team_id"], "team_id"),
            "period": period,
            "t_s": _floats(df["t_s"], "t_s"),
            "x_m": x_m,
            "y_m": y_m,
            "speed_mps": speed,
            "source": np.full(n, source.value, dtype=object),
            "detected": detected,
            "identity_confidence": ident,
            "wall_time_ns": _ns_column(df, "wall_time_ns"),
            "ingest_time_ns": _ns_column(df, "ingest_time_ns"),
        }
    )[list(TRACKING_COLUMNS)]
    _check_tracking_frame(out)
    return out.sort_values(["period", "t_s", "player_id"], kind="stable").reset_index(drop=True)


def _event_type(value: object) -> EventType:
    """Normalised event type ("Yellow Card" -> yellow_card); raises on unknown values."""
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return EventType(text)
    except ValueError:
        valid = sorted(e.value for e in EventType)
        raise ValueError(f"unknown event type {value!r}: expected one of {valid}") from None


def _opt_str(value: object) -> str | None:
    """Optional id cell as a string; None when missing; whole floats lose their decimal."""
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _opt_strs(df: pd.DataFrame, name: str) -> list[str | None]:
    if name not in df.columns:
        return [None] * df.shape[0]
    return [_opt_str(v) for v in df[name].tolist()]


def read_events_table(
    path: str | Path, column_map: dict[str, str], match_id: str
) -> list[MatchEvent]:
    """Official-feed event export (CSV or Parquet) to ``MatchEvent`` rows (docs/03 §3).

    ``column_map`` maps contract field names to the feed's column names; ``period``, ``t_s`` and
    ``type`` are required, ``team_id``, ``player_id``, ``player_in_id`` and ``stoppage_min``
    optional. Event type strings are normalised (case, spaces and hyphens) to the EventType
    values; unknown types raise ValueError. Units: seconds on the match clock, minutes for
    ``stoppage_min``. Rows failing the contract ranges raise ValueError. Events are sorted by
    (period, t_s) and stamped ``source=official_feed``; the as_of rule is applied downstream.
    """
    df = _read_table(Path(path))
    df = _apply_column_map(df, column_map, _EVENT_FIELDS, _REQUIRED_EVENTS)
    n = df.shape[0]
    period = _whole_ints(df["period"], "period")
    t_s = _floats(df["t_s"], "t_s")
    if not bool(np.all(np.isfinite(t_s))):
        raise ValueError("t_s must not be missing")
    types = [_event_type(v) for v in df["type"].tolist()]
    team = _opt_strs(df, "team_id")
    player = _opt_strs(df, "player_id")
    player_in = _opt_strs(df, "player_in_id")
    if "stoppage_min" in df.columns:
        stop_values = _floats(df["stoppage_min"], "stoppage_min")
        stoppage = [float(v) if np.isfinite(v) else None for v in stop_values]
    else:
        stoppage = [None] * n
    events = [
        MatchEvent(
            match_id=match_id,
            period=int(period[i]),
            t_s=float(t_s[i]),
            type=types[i],
            team_id=team[i],
            player_id=player[i],
            player_in_id=player_in[i],
            stoppage_min=stoppage[i],
            source=EventSource.OFFICIAL_FEED,
        )
        for i in range(n)
    ]
    events.sort(key=lambda e: (e.period, e.t_s))
    return events


# --------------------------------------------------------------------------------------------
# SkillCorner open data (docs/03 §8)
# --------------------------------------------------------------------------------------------

#: Role acronym -> position group, from the SkillCorner ``player_role.acronym`` values.
_POSITION_BY_ACRONYM: dict[str, PositionGroup] = {
    "GK": PositionGroup.GK,
    "CB": PositionGroup.CB,
    "LCB": PositionGroup.CB,
    "RCB": PositionGroup.CB,
    "RB": PositionGroup.FB,
    "LB": PositionGroup.FB,
    "RWB": PositionGroup.FB,
    "LWB": PositionGroup.FB,
    "FB": PositionGroup.FB,
    "DM": PositionGroup.MF,
    "LDM": PositionGroup.MF,
    "RDM": PositionGroup.MF,
    "CM": PositionGroup.MF,
    "LCM": PositionGroup.MF,
    "RCM": PositionGroup.MF,
    "AM": PositionGroup.MF,
    "CAM": PositionGroup.MF,
    "LM": PositionGroup.MF,
    "RM": PositionGroup.MF,
    "LW": PositionGroup.W,
    "RW": PositionGroup.W,
    "CF": PositionGroup.FW,
    "ST": PositionGroup.FW,
    "SS": PositionGroup.FW,
    "FW": PositionGroup.FW,
}

#: ``player_role.position_group`` fallback when the acronym is unlisted.
_POSITION_BY_GROUP: dict[str, PositionGroup] = {
    "Goalkeeper": PositionGroup.GK,
    "Central Defender": PositionGroup.CB,
    "Full Back": PositionGroup.FB,
    "Midfield": PositionGroup.MF,
    "Wide Attacker": PositionGroup.W,
    "Center Forward": PositionGroup.FW,
}


@dataclass(frozen=True)
class _SCPlayer:
    """One player entry of a SkillCorner ``<id>_match.json`` file."""

    player_id: str
    team_id: str
    shirt_number: int
    position: PositionGroup
    starting: bool
    start_frame: int | None
    end_frame: int | None
    red_cards: int


def _sc_position(acronym: str, group: str) -> PositionGroup:
    """Position group from the role acronym, then the group label; unlisted roles become MF."""
    if acronym.upper() in _POSITION_BY_ACRONYM:
        return _POSITION_BY_ACRONYM[acronym.upper()]
    if group in _POSITION_BY_GROUP:
        return _POSITION_BY_GROUP[group]
    return PositionGroup.MF


def _opt_frame(value: Any) -> int | None:
    return None if value is None else int(value)


def _sc_players(meta: dict[str, Any]) -> list[_SCPlayer]:
    players: list[_SCPlayer] = []
    for raw in meta["players"]:
        role = raw["player_role"]
        total = (raw.get("playing_time") or {}).get("total") or {}
        players.append(
            _SCPlayer(
                player_id=str(raw["id"]),
                team_id=str(raw["team_id"]),
                shirt_number=int(raw["number"]),
                position=_sc_position(
                    str(role.get("acronym") or ""), str(role.get("position_group") or "")
                ),
                starting=str(raw.get("start_time") or "") == "00:00:00",
                start_frame=_opt_frame(total.get("start_frame")),
                end_frame=_opt_frame(total.get("end_frame")),
                red_cards=int(raw.get("red_card") or 0),
            )
        )
    return players


def _sc_periods(meta: dict[str, Any]) -> dict[int, tuple[int, int]]:
    """Frame window [start_frame, end_frame] per period from ``match_periods``."""
    periods: dict[int, tuple[int, int]] = {}
    for entry in meta["match_periods"]:
        period = int(entry["period"])
        if period not in PERIOD_START_S:
            raise ValueError(f"SkillCorner period must be in 1-4, got {period}")
        periods[period] = (int(entry["start_frame"]), int(entry["end_frame"]))
    if not periods:
        raise ValueError("match metadata has no match_periods")
    return periods


def _sc_frame_time(frame: int, periods: dict[int, tuple[int, int]]) -> tuple[int, float]:
    """Match clock (period, t_s seconds) of a frame at SKILLCORNER_FPS.

    A frame between period windows (half-time) snaps to the next period's start; a frame past
    the last window raises.
    """
    for period in sorted(periods):
        start, end = periods[period]
        if start <= frame <= end:
            return period, PERIOD_START_S[period] + (frame - start) / SKILLCORNER_FPS
    later = [p for p in sorted(periods) if periods[p][0] > frame]
    if later:
        return later[0], PERIOD_START_S[later[0]]
    raise ValueError(f"frame {frame} is past the last match period")


def _sc_events(
    match_id: str, players: list[_SCPlayer], periods: dict[int, tuple[int, int]]
) -> list[MatchEvent]:
    """Clock and disciplinary events reconstructed from the match metadata.

    Kick-off and period_end come from the period frame windows. A substitution pairs a
    non-starter's first tracked frame with the same-team leaver whose last tracked frame
    immediately precedes it (within _SUB_PAIR_FRAME_TOL). A red card is timed at the carded
    player's last tracked frame when no substitution consumed it. Goals and cards without
    timing information cannot be placed on the clock and are omitted.
    """
    events: list[MatchEvent] = []
    for period in sorted(periods):
        start, end = periods[period]
        events.append(
            MatchEvent(
                match_id=match_id,
                period=period,
                t_s=PERIOD_START_S[period],
                type=EventType.KICKOFF,
                source=EventSource.OFFICIAL_FEED,
            )
        )
        events.append(
            MatchEvent(
                match_id=match_id,
                period=period,
                t_s=PERIOD_START_S[period] + (end - start) / SKILLCORNER_FPS,
                type=EventType.PERIOD_END,
                source=EventSource.OFFICIAL_FEED,
            )
        )

    ins = sorted(
        (p for p in players if not p.starting and p.start_frame is not None),
        key=lambda p: (p.start_frame or 0, p.player_id),
    )
    outs = sorted(
        (p for p in players if p.end_frame is not None),
        key=lambda p: (p.end_frame or 0, p.player_id),
    )
    used_out: set[str] = set()
    for in_p in ins:
        in_frame = in_p.start_frame
        if in_frame is None:
            continue
        best: _SCPlayer | None = None
        best_gap = _SUB_PAIR_FRAME_TOL + 1
        for out_p in outs:
            if (
                out_p.team_id != in_p.team_id
                or out_p.player_id in used_out
                or out_p.end_frame is None
            ):
                continue
            gap = abs(out_p.end_frame - (in_frame - 1))
            if gap <= _SUB_PAIR_FRAME_TOL and gap < best_gap:
                best, best_gap = out_p, gap
        if best is None:
            continue
        used_out.add(best.player_id)
        period, t_s = _sc_frame_time(in_frame, periods)
        events.append(
            MatchEvent(
                match_id=match_id,
                period=period,
                t_s=t_s,
                type=EventType.SUBSTITUTION,
                team_id=in_p.team_id,
                player_id=best.player_id,
                player_in_id=in_p.player_id,
                source=EventSource.OFFICIAL_FEED,
            )
        )
    for p in players:
        if p.red_cards > 0 and p.end_frame is not None and p.player_id not in used_out:
            period, t_s = _sc_frame_time(p.end_frame, periods)
            events.append(
                MatchEvent(
                    match_id=match_id,
                    period=period,
                    t_s=t_s,
                    type=EventType.RED_CARD,
                    team_id=p.team_id,
                    player_id=p.player_id,
                    source=EventSource.OFFICIAL_FEED,
                )
            )
    events.sort(key=lambda e: (e.period, e.t_s))
    return events


def _sc_tracking(
    path: Path, match_id: str, team_of: dict[str, str], periods: dict[int, tuple[int, int]]
) -> pd.DataFrame:
    """Tracking JSONL to a ``TrackingSample`` frame; unknown or position-less entries dropped."""
    pid_l: list[str] = []
    team_l: list[str] = []
    per_l: list[int] = []
    t_l: list[float] = []
    x_l: list[float] = []
    y_l: list[float] = []
    det_l: list[bool] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            frame_obj = json.loads(line)
            period = frame_obj.get("period")
            if period not in periods:
                continue
            start, end = periods[period]
            frame = int(frame_obj["frame"])
            if frame < start or frame > end:
                continue
            t_s = PERIOD_START_S[period] + (frame - start) / SKILLCORNER_FPS
            for entry in frame_obj.get("player_data") or []:
                pid, x, y = entry.get("player_id"), entry.get("x"), entry.get("y")
                if pid is None or x is None or y is None:
                    continue
                team = team_of.get(str(pid))
                if team is None:
                    continue
                pid_l.append(str(pid))
                team_l.append(team)
                per_l.append(period)
                t_l.append(t_s)
                x_l.append(float(x))
                y_l.append(float(y))
                det_l.append(bool(entry.get("is_detected")))
    n = len(t_l)
    missing = pd.arrays.IntegerArray(np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.bool_))
    df = pd.DataFrame(
        {
            "match_id": np.full(n, match_id, dtype=object),
            "player_id": np.asarray(pid_l, dtype=object),
            "team_id": np.asarray(team_l, dtype=object),
            "period": np.asarray(per_l, dtype=np.int64),
            "t_s": np.asarray(t_l, dtype=np.float64),
            "x_m": np.asarray(x_l, dtype=np.float64),
            "y_m": np.asarray(y_l, dtype=np.float64),
            "speed_mps": np.full(n, np.nan),
            "source": np.full(n, Source.BROADCAST.value, dtype=object),
            "detected": np.asarray(det_l, dtype=np.bool_),
            "identity_confidence": np.ones(n, dtype=np.float64),
            "wall_time_ns": missing,
            "ingest_time_ns": missing.copy(),
        }
    )[list(TRACKING_COLUMNS)]
    _check_tracking_frame(df)
    return df.sort_values(["period", "t_s", "player_id"], kind="stable").reset_index(drop=True)


def _single_file(root: Path, pattern: str) -> Path:
    found = sorted(root.glob(pattern))
    if len(found) != 1:
        raise ValueError(f"expected exactly one {pattern!r} under {root}, found {len(found)}")
    return found[0]


def load_skillcorner_match(
    match_dir: str | Path,
) -> tuple[pd.DataFrame, list[MatchEvent], tuple[Lineup, ...]]:
    """SkillCorner open-data match (``data/matches/<id>/`` of the opendata repository).

    Reads ``<id>_match.json`` and ``<id>_tracking_extrapolated.jsonl``: 10 fps broadcast
    tracking with ``is_detected`` mapped to ``detected`` (null counts as not detected) and
    source ``broadcast``; the coordinates already follow the contract convention (metres,
    centre-spot origin). The data carries no speed and no per-sample identity confidence, so
    ``speed_mps`` stays missing (the feature engine differentiates the positions) and
    ``identity_confidence`` is 1.0. Match times come from the frame count at 10 fps on the
    contract clock (period starts 0/2700/5400/6300 s). Events are reconstructed from the match
    metadata (see ``_sc_events``); lineups cover both squads with starters flagged by a
    ``00:00:00`` start time. Returns (tracking frame, events, lineups).
    """
    root = Path(match_dir)
    meta = json.loads(_single_file(root, "*_match.json").read_text(encoding="utf-8"))
    tracking_path = _single_file(root, "*_tracking_extrapolated.jsonl")
    match_id = str(meta["id"])
    kickoff = datetime.fromisoformat(str(meta["date_time"]))
    periods = _sc_periods(meta)
    players = _sc_players(meta)
    team_of = {p.player_id: p.team_id for p in players}
    lineups = tuple(
        Lineup(
            match_id=match_id,
            team_id=team_id,
            kickoff_utc=kickoff,
            players=tuple(
                LineupPlayer(
                    player_id=p.player_id,
                    shirt_number=p.shirt_number,
                    position_group=p.position,
                    starting=p.starting,
                )
                for p in players
                if p.team_id == team_id
            ),
        )
        for team_id in (str(meta["home_team"]["id"]), str(meta["away_team"]["id"]))
    )
    tracking = _sc_tracking(tracking_path, match_id, team_of, periods)
    events = _sc_events(match_id, players, periods)
    return tracking, events, lineups
