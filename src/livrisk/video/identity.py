"""Lineup-constrained identity assignment for external video tracks (docs/VIDEO_INPUT.md).

The input is the per-frame output of an external tracking pipeline: track labels, team guesses,
jersey reads and pitch positions. The output is TrackingSample rows (docs/03 §2) with a
``player_id`` and an ``identity_confidence``; windows without a trustworthy assignment are
dropped, so a doubtful stretch shows up as a coverage gap and never as a guessed identity.
The method is per-window Hungarian assignment (``scipy.optimize.linear_sum_assignment``) of the
visible tracks to the on-pitch lineup players of each team, with costs from windowed jersey
votes and positional continuity. The on-pitch sets follow the substitution and red-card
events. A track whose proposed player persistently disagrees with its confirmed player is
re-assigned and flagged as an identity swap. No vision models are trained here; this module
only consumes the output of an external pipeline.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from livrisk.contracts import PERIOD_START_S, EventType, Lineup, MatchEvent, Source, TrackingSample

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: Required columns of the per-frame input frame (docs/VIDEO_INPUT.md).
TRACK_COLUMNS: tuple[str, ...] = (
    "frame_t_s",
    "period",
    "track_id",
    "team_guess",
    "jersey_read",
    "jersey_conf",
    "x_m",
    "y_m",
    "detected",
)

_SAMPLE_COLUMNS: tuple[str, ...] = tuple(TrackingSample.model_fields)
_OUT_COLUMNS: list[str] = [*_SAMPLE_COLUMNS, "track_id"]
_SWAP_COLUMNS: tuple[str, ...] = (
    "track_id",
    "period",
    "t_s",
    "from_player_id",
    "to_player_id",
)

#: Assumed period lengths when an event stream carries no period_end, seconds (D-016).
_STANDARD_PERIOD_LEN_S: dict[int, float] = {1: 2700.0, 2: 2700.0, 3: 900.0, 4: 900.0}
#: Cap on the confidence of a window without any jersey read in memory: continuity-only
#: identity stays below quality.min_identity (0.8), so a read outage surfaces downstream.
_CONTINUITY_ONLY_CAP: float = 0.75
#: Confidence when a window has neither reads nor a previous position for the player.
_NO_EVIDENCE_CONF: float = 0.25
#: A previous position stops anchoring continuity after this long without the player, seconds.
_POS_ANCHOR_S: float = 10.0
#: Continuity-only pairs weaker than this are refused when the track has no reads at all.
_MAX_CONTINUITY_COST: float = 0.8


@dataclass(frozen=True)
class IdentityResult:
    """Output of ``assign_identities``.

    ``samples`` carries the TrackingSample columns plus ``track_id`` (the source track behind
    each identity, kept for traceability), in input row order minus dropped windows. ``swaps``
    has one row per confirmed identity switch of a track. ``dropped_rows`` counts input rows
    that left no sample (unconfirmed tracks, pending swaps, refused assignments).
    """

    samples: pd.DataFrame
    swaps: pd.DataFrame
    dropped_rows: int


class _Reads:
    """Jersey reads of one track over a trailing window, with running vote masses.

    Reads are append-only and pruned by age; the vote mass per shirt and the total are updated
    incrementally so a query is O(1). Units: times in play-clock seconds, confidences in [0, 1].
    """

    def __init__(self) -> None:
        self._u: list[float] = []
        self._shirt: list[int] = []
        self._conf: list[float] = []
        self._lo = 0
        self.mass: dict[int, float] = {}
        self.total = 0.0

    def add(self, u_s: float, shirt: int, conf: float) -> None:
        self._u.append(u_s)
        self._shirt.append(shirt)
        self._conf.append(conf)
        self.mass[shirt] = self.mass.get(shirt, 0.0) + conf
        self.total += conf

    def prune(self, keep_from_u: float) -> None:
        """Drop reads older than ``keep_from_u`` (play seconds)."""
        lo = bisect.bisect_left(self._u, keep_from_u, self._lo)
        for i in range(self._lo, lo):
            shirt = self._shirt[i]
            left = self.mass[shirt] - self._conf[i]
            if left <= 1e-12:
                del self.mass[shirt]
            else:
                self.mass[shirt] = left
            self.total -= self._conf[i]
        self._lo = lo
        if self._lo > 4096:
            del self._u[: self._lo]
            del self._shirt[: self._lo]
            del self._conf[: self._lo]
            self._lo = 0

    def support(self, shirt: int) -> tuple[float, float]:
        """(Vote mass for ``shirt``, total mass) over the trailing window."""
        return self.mass.get(shirt, 0.0), self.total

    def clear(self) -> None:
        """Forget every read; used when a track is confirmed to follow another player."""
        self._u.clear()
        self._shirt.clear()
        self._conf.clear()
        self._lo = 0
        self.mass.clear()
        self.total = 0.0


@dataclass
class _TrackState:
    """Confirmed player of a track plus the pending challenger and its consecutive streak."""

    current: str | None = None
    pending: str | None = None
    streak: int = 0


def _period_starts(events: Sequence[MatchEvent]) -> dict[int, float]:
    """Play-clock start of each period from the period_end events (D-016 convention)."""
    lens = dict(_STANDARD_PERIOD_LEN_S)
    for e in events:
        if e.type == EventType.PERIOD_END:
            lens[e.period] = e.t_s - PERIOD_START_S[e.period]
    return {1: 0.0, 2: lens[1], 3: lens[1] + lens[2], 4: lens[1] + lens[2] + lens[3]}


def _play_seconds(starts: dict[int, float], period: int, t_s: float) -> float:
    """Play seconds of a match-clock stamp; seconds."""
    return starts[period] + (t_s - PERIOD_START_S[period])


def _match_clock(starts: dict[int, float], u_s: float) -> tuple[int, float]:
    """(period, t_s) of a play-clock time; seconds."""
    period = 1
    for p in (2, 3, 4):
        if u_s >= starts[p]:
            period = p
    return period, PERIOD_START_S[period] + (u_s - starts[period])


def _validate_tracks(tracks: pd.DataFrame, team_ids: set[str]) -> None:
    """Fail loud on structural problems in the per-frame input."""
    missing = [c for c in TRACK_COLUMNS if c not in tracks.columns]
    if missing:
        raise ValueError(f"tracks is missing required columns: {missing}")
    period = pd.to_numeric(tracks["period"], errors="coerce")
    if bool(period.isna().any()) or not set(period.unique()) <= {1, 2, 3, 4}:
        raise ValueError("period values must be in 1-4")
    t_s = pd.to_numeric(tracks["frame_t_s"], errors="coerce")
    if bool(t_s.isna().any()):
        raise ValueError("frame_t_s must be a number, seconds on the match clock")
    lo = np.asarray([PERIOD_START_S[int(p)] for p in period])
    if bool((t_s.to_numpy(dtype=np.float64) < lo).any()):
        raise ValueError("frame_t_s before the start of its period")
    bad_teams = set(tracks["team_guess"].astype(str).unique()) - team_ids
    if bad_teams:
        raise ValueError(f"team_guess values not in the lineups: {sorted(bad_teams)}")
    conf = pd.to_numeric(tracks["jersey_conf"], errors="coerce")
    if bool(((conf < 0.0) | (conf > 1.0)).any()):
        raise ValueError("jersey_conf must be in [0, 1] or missing")
    read = pd.to_numeric(tracks["jersey_read"], errors="coerce")
    if bool(((read < 1) | (read > 99)).any()):
        raise ValueError("jersey_read must be in 1-99 or missing")
    if bool((read.dropna() % 1 != 0).any()):
        raise ValueError("jersey_read must be a whole shirt number")
    if not set(tracks["detected"].unique()) <= {False, True}:
        raise ValueError("detected must be true/false")
    for col in ("x_m", "y_m"):
        pos = pd.to_numeric(tracks[col], errors="coerce")
        if bool(pos.isna().any()):
            raise ValueError(f"{col} must be present on every row (metres)")


def assign_identities(
    tracks: pd.DataFrame,
    lineups: Sequence[Lineup],
    events: Sequence[MatchEvent],
    *,
    window_s: float = 1.0,
    confirm_windows: int = 2,
    read_memory_s: float = 30.0,
    continuity_weight: float = 0.5,
    continuity_scale_m: float = 5.0,
    prior_reads: float = 4.0,
    prior_misread: float = 0.02,
) -> IdentityResult:
    """Assign per-frame video tracks to on-pitch lineup players.

    Units: positions in metres (docs/03 §1), times in seconds on the match clock,
    ``window_s`` and ``read_memory_s`` in seconds of play. As_of rule: this is an offline
    batch step over a recorded match, but windows are processed in play-clock order and a
    substitution or red card affects only windows at or after its event time, so no window
    uses future information. Deterministic: there is no randomness in the assignment.

    ``jersey_read`` may be missing on any frame (unreadable); ``jersey_conf`` weights the
    vote. A track is emitted only once the same player has won its window assignment for
    ``confirm_windows`` consecutive windows; the same hysteresis re-assigns a track after a
    tracker identity swap, and the switch is flagged in ``IdentityResult.swaps`` at the end of
    the confirming window. ``identity_confidence`` is the share of the last ``read_memory_s``
    of reads supporting the emitted identity, shrunk by a prior of ``prior_reads`` reads at
    ``prior_misread`` error; without reads it falls back to a capped continuity score.
    """
    if not lineups:
        raise ValueError("at least one lineup is required")
    if window_s <= 0.0 or read_memory_s <= 0.0 or continuity_scale_m <= 0.0:
        raise ValueError("window_s, read_memory_s and continuity_scale_m must be > 0")
    if confirm_windows < 1:
        raise ValueError("confirm_windows must be >= 1")
    if prior_reads <= 0.0 or not 0.0 <= prior_misread < 1.0:
        raise ValueError("prior_reads must be > 0 and prior_misread in [0, 1)")
    match_id = lineups[0].match_id
    if any(lu.match_id != match_id for lu in lineups) or any(
        e.match_id != match_id for e in events
    ):
        raise ValueError("lineups and events must belong to one match")
    team_ids = {lu.team_id for lu in lineups}
    _validate_tracks(tracks, team_ids)

    shirt_of = {p.player_id: p.shirt_number for lu in lineups for p in lu.players}
    team_of = {p.player_id: lu.team_id for lu in lineups for p in lu.players}
    starts = _period_starts(events)
    a_prior = prior_reads * (1.0 - prior_misread)
    b_prior = prior_reads * prior_misread

    n = tracks.shape[0]
    period = np.asarray(tracks["period"].to_numpy(), dtype=np.int64)
    t_s = np.asarray(tracks["frame_t_s"].to_numpy(), dtype=np.float64)
    start_of = np.asarray([starts[p] for p in (1, 2, 3, 4)])
    origin = np.asarray([PERIOD_START_S[p] for p in (1, 2, 3, 4)])
    u_all = start_of[period - 1] + (t_s - origin[period - 1])
    x_all = np.asarray(tracks["x_m"].to_numpy(), dtype=np.float64)
    y_all = np.asarray(tracks["y_m"].to_numpy(), dtype=np.float64)
    det_all = np.asarray(tracks["detected"].to_numpy(), dtype=np.bool_)
    read = pd.to_numeric(tracks["jersey_read"], errors="coerce").to_numpy(dtype=np.float64)
    conf = pd.to_numeric(tracks["jersey_conf"], errors="coerce").to_numpy(dtype=np.float64)
    shirt_all = np.where(np.isnan(read), -1, np.rint(read)).astype(np.int64)
    conf_all = np.where(np.isnan(conf) | (shirt_all < 0), 0.0, conf)
    track_id_all = np.asarray(tracks["track_id"].astype(str).to_numpy(), dtype=object)
    team_all = np.asarray(tracks["team_guess"].astype(str).to_numpy(), dtype=object)

    emit_rows: list[int] = []
    emit_player: list[str] = []
    emit_conf: list[float] = []
    flags: list[tuple[str, float, str, str]] = []

    for lu in lineups:
        team_id = lu.team_id
        sel = np.flatnonzero(team_all == team_id)
        if sel.size == 0:
            continue
        order = sel[np.argsort(u_all[sel], kind="stable")]
        u = u_all[order]
        on_pitch = {p.player_id for p in lu.players if p.starting}
        team_events = sorted(
            (
                _play_seconds(starts, e.period, e.t_s),
                str(e.player_id),
                str(e.player_in_id) if e.type == EventType.SUBSTITUTION else None,
            )
            for e in events
            if e.team_id == team_id
            and e.player_id is not None
            and (
                (e.type == EventType.SUBSTITUTION and e.player_in_id is not None)
                or e.type == EventType.RED_CARD
            )
        )
        ev_i = 0
        states: dict[int, _TrackState] = {}
        memories: dict[int, _Reads] = {}
        prev_pos: dict[str, tuple[float, float, int]] = {}
        codes = np.empty(order.size, dtype=np.int64)
        track_ids: list[str] = []
        code_of: dict[str, int] = {}
        for pos, row in enumerate(order):
            tid = str(track_id_all[row])
            code = code_of.get(tid, -1)
            if code < 0:
                code = len(track_ids)
                code_of[tid] = code
                track_ids.append(tid)
            codes[pos] = code

        k_min = math.floor(u[0] / window_s)
        k_max = math.floor(u[-1] / window_s)
        for k in range(k_min, k_max + 1):
            w_start = k * window_s
            while ev_i < len(team_events) and team_events[ev_i][0] <= w_start + 1e-9:
                _, out_id, in_id = team_events[ev_i]
                on_pitch.discard(out_id)
                if in_id is not None:
                    on_pitch.add(in_id)
                ev_i += 1
            lo = int(np.searchsorted(u, w_start, "left"))
            hi = int(np.searchsorted(u, w_start + window_s, "left"))
            players = sorted(on_pitch)
            if lo == hi or not players:
                continue

            w_codes = codes[lo:hi]
            uniq, inv = np.unique(w_codes, return_inverse=True)
            n_t = uniq.size
            w_x = x_all[order[lo:hi]]
            w_y = y_all[order[lo:hi]]
            w_det = det_all[order[lo:hi]]
            det_w = w_det.astype(np.float64)
            cnt_det = np.bincount(inv, weights=det_w, minlength=n_t)
            cnt_all = np.bincount(inv, minlength=n_t)
            sum_x = np.bincount(inv, weights=np.where(w_det, w_x, 0.0), minlength=n_t)
            sum_y = np.bincount(inv, weights=np.where(w_det, w_y, 0.0), minlength=n_t)
            all_x = np.bincount(inv, weights=w_x, minlength=n_t)
            all_y = np.bincount(inv, weights=w_y, minlength=n_t)
            mean_x = np.where(cnt_det > 0, sum_x / np.maximum(cnt_det, 1), all_x / cnt_all)
            mean_y = np.where(cnt_det > 0, sum_y / np.maximum(cnt_det, 1), all_y / cnt_all)

            votes: list[dict[int, float]] = [{} for _ in range(n_t)]
            vtotal = np.zeros(n_t)
            for r in range(lo, hi):
                shirt = int(shirt_all[order[r]])
                if shirt < 0:
                    continue
                i = int(inv[r - lo])
                c = float(conf_all[order[r]])
                votes[i][shirt] = votes[i].get(shirt, 0.0) + c
                vtotal[i] += c
                g = int(uniq[i])
                mem = memories.get(g)
                if mem is None:
                    mem = memories[g] = _Reads()
                mem.add(float(u[r]), shirt, c)

            n_p = len(players)
            shirts = [shirt_of[p] for p in players]
            cost = np.empty((n_t, n_p))
            for i in range(n_t):
                for j in range(n_p):
                    if vtotal[i] > 0.0:
                        c_jersey = 1.0 - votes[i].get(shirts[j], 0.0) / vtotal[i]
                    else:
                        c_jersey = 0.5
                    anchor = prev_pos.get(players[j])
                    if anchor is not None and (k - anchor[2]) * window_s <= _POS_ANCHOR_S:
                        d = math.hypot(mean_x[i] - anchor[0], mean_y[i] - anchor[1])
                        c_cont = d / (d + continuity_scale_m)
                    else:
                        c_cont = 0.5
                    cost[i, j] = c_jersey + continuity_weight * c_cont
            rows_i, cols_j = linear_sum_assignment(cost)
            proposal: dict[int, str] = {}
            for i_raw, j_raw in zip(rows_i.tolist(), cols_j.tolist(), strict=True):
                i, j = int(i_raw), int(j_raw)
                if vtotal[i] > 0.0 and votes[i].get(shirts[j], 0.0) <= 0.0:
                    continue  # reads exist and none support this player: refuse the pair
                if vtotal[i] == 0.0:
                    anchor = prev_pos.get(players[j])
                    if anchor is None or (k - anchor[2]) * window_s > _POS_ANCHOR_S:
                        continue  # no reads and no fresh position: refuse
                    d = math.hypot(mean_x[i] - anchor[0], mean_y[i] - anchor[1])
                    if d / (d + continuity_scale_m) > _MAX_CONTINUITY_COST:
                        continue
                proposal[i] = players[j]

            emitted: list[tuple[int, str]] = []
            for i in range(n_t):
                g = int(uniq[i])
                st = states.get(g)
                if st is None:
                    st = states[g] = _TrackState()
                prop = proposal.get(i)
                if st.current is None:
                    if prop is not None:
                        if st.pending == prop:
                            st.streak += 1
                        else:
                            st.pending, st.streak = prop, 1
                        if st.streak >= confirm_windows:
                            st.current, st.pending, st.streak = prop, None, 0
                elif prop == st.current:
                    st.pending, st.streak = None, 0
                elif prop is None:
                    pass  # one unassigned window keeps the confirmed state
                else:
                    if st.pending == prop:
                        st.streak += 1
                    else:
                        st.pending, st.streak = prop, 1
                    if st.streak >= confirm_windows:
                        flags.append((track_ids[g], (k + 1) * window_s, st.current, prop))
                        st.current, st.pending, st.streak = prop, None, 0
                        mem = memories[g]
                        mem.clear()
                        for r in range(lo, hi):
                            if int(inv[r - lo]) == i and int(shirt_all[order[r]]) >= 0:
                                mem.add(
                                    float(u[r]),
                                    int(shirt_all[order[r]]),
                                    float(conf_all[order[r]]),
                                )
                if st.current is not None and prop == st.current:
                    emitted.append((i, st.current))
                    mem = memories[g]
                    mem.prune(w_start + window_s - read_memory_s)
                    v_for, v_all = mem.support(shirt_of[st.current])
                    if v_all > 0.0:
                        confidence = (v_for + a_prior) / (v_all + a_prior + b_prior)
                    else:
                        anchor = prev_pos.get(st.current)
                        if anchor is not None and (k - anchor[2]) * window_s <= _POS_ANCHOR_S:
                            d = math.hypot(mean_x[i] - anchor[0], mean_y[i] - anchor[1])
                            confidence = _CONTINUITY_ONLY_CAP * (1.0 - d / (d + continuity_scale_m))
                        else:
                            confidence = _NO_EVIDENCE_CONF
                    rows_of_track = order[lo:hi][inv == i].tolist()
                    emit_rows.extend(rows_of_track)
                    emit_player.extend([st.current] * len(rows_of_track))
                    emit_conf.extend([confidence] * len(rows_of_track))
            for i, pid in emitted:
                prev_pos[pid] = (float(mean_x[i]), float(mean_y[i]), k)

    if emit_rows:
        back = np.argsort(np.asarray(emit_rows, dtype=np.int64), kind="stable")
        idx = np.asarray(emit_rows, dtype=np.int64)[back]
        players_out = [emit_player[i] for i in back.tolist()]
        conf_out = np.asarray(emit_conf, dtype=np.float64)[back]
    else:
        idx = np.zeros(0, dtype=np.int64)
        players_out = []
        conf_out = np.zeros(0)
    n_out = int(idx.size)
    missing = pd.arrays.IntegerArray(np.zeros(n_out, dtype=np.int64), np.ones(n_out, np.bool_))
    samples = pd.DataFrame(
        {
            "match_id": np.full(n_out, match_id, dtype=object),
            "player_id": np.asarray(players_out, dtype=object),
            "team_id": np.asarray([team_of[p] for p in players_out], dtype=object),
            "period": period[idx],
            "t_s": t_s[idx],
            "x_m": x_all[idx],
            "y_m": y_all[idx],
            "speed_mps": np.full(n_out, np.nan),
            "source": np.full(n_out, Source.VIDEO.value, dtype=object),
            "detected": det_all[idx],
            "identity_confidence": conf_out,
            "wall_time_ns": missing,
            "ingest_time_ns": missing.copy(),
            "track_id": track_id_all[idx],
        }
    )[_OUT_COLUMNS]
    swap_rows = [
        {
            "track_id": tid,
            "period": _match_clock(starts, u_flag)[0],
            "t_s": _match_clock(starts, u_flag)[1],
            "from_player_id": out_id,
            "to_player_id": in_id,
        }
        for tid, u_flag, out_id, in_id in flags
    ]
    swaps = pd.DataFrame(swap_rows, columns=list(_SWAP_COLUMNS))
    logger.info(
        "identity assignment: %d rows in, %d emitted, %d swaps flagged",
        n,
        n_out,
        len(swap_rows),
    )
    return IdentityResult(samples=samples, swaps=swaps, dropped_rows=n - n_out)
