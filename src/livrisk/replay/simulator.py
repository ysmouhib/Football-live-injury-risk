"""Replay simulator and look-ahead guard (docs/04_ARCHITECTURE.md §3, §7; docs/05_VALIDATION.md §3).

A recorded or synthetic match runs through the live path on a simulated clock: source -> ingest
queue -> ``StreamFeatureEngine`` (every ``engine.feature_tick_s``) -> ``RiskEngine`` (every
``engine.risk_tick_s``) -> ``AlertLogic`` (one per team) -> screen. Latency, random dropouts,
dropout bursts and source clock drift are injected at the source; everything is written to the
log store when one is given.

Clocks
------
Ticks run on the play clock (seconds of play since kick-off, breaks removed; D-016). The
simulated wall clock starts at ``Lineup.kickoff_utc`` and adds the period breaks: a sample of
period p played at play second u is produced at wall time kickoff + u + break before p and
arrives ``latency_s`` later (``wall_time_ns`` / ``ingest_time_ns``). A tick at play second u sees
only what has arrived by its wall time; a tick on a period boundary belongs to the period that
ends there. The match clock handed to the risk engine comes from the kick-off, period-end and
stoppage events delivered so far, as it would live.

Data age
--------
``data_age_s`` is the wall-clock age of the newest sample received for the player, or the time
since he came on before his first sample (D-021). The blanking gap delta is a fixed design
choice and never counts as age. The screen gets a push at every feature tick, ``push_delay_s``
later, with each on-pitch player's latest risk row and his current data age; values are greyed
above ``quality.stale_grey_s`` and hidden above ``quality.stale_hide_s`` (docs/06 §4).

Look-ahead
----------
``LookAheadGuard`` sees every read of the feature buffers and raises ``LookAheadError`` when a
read reaches beyond ``as_of - delta``; the simulator also rejects snapshots stamped later than
the tick. The guard is always on.
"""

from __future__ import annotations

import bisect
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from livrisk.alerts.logic import AlertLogic
from livrisk.config import Config
from livrisk.contracts import (
    PERIOD_START_S,
    Alert,
    EventType,
    FeatureSnapshot,
    Lineup,
    MatchEvent,
    PreMatchRisk,
    Quality,
    RiskSnapshot,
    Source,
    TrackingSample,
)
from livrisk.engine.risk_engine import MatchState, RiskEngine
from livrisk.features.norms import NormTable
from livrisk.features.stream import StreamFeatureEngine
from livrisk.model.live_hazard import FittedHazard
from livrisk.storage.logstore import LogStore, records_frame

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Wall-clock break before each period, seconds (half-time; extra-time intervals).
_BREAK_BEFORE_S: dict[int, float] = {1: 0.0, 2: 900.0, 3: 300.0, 4: 0.0}
#: Standard period lengths before a period_end is known, seconds.
_STANDARD_LEN_S: dict[int, float] = {1: 2700.0, 2: 2700.0, 3: 900.0, 4: 900.0}
_EPS_S: float = 1e-9
_NS: int = 1_000_000_000

#: Display states of a screen row (docs/06 §4).
LIVE, GREY, HIDDEN = "live", "grey", "hidden"


class LookAheadError(RuntimeError):
    """Raised when any component reads data with t_s > as_of - delta."""


class LookAheadGuard:
    """Read guard of the feature buffers (``stream.ReadGuard``).

    The simulator advances it to each tick's play second; a buffer read whose cutoff lies beyond
    ``as_of - delta`` raises ``LookAheadError``. Before the first ``advance`` every read raises.
    """

    def __init__(self, delta_s: float) -> None:
        self._delta = float(delta_s)
        self._as_of = -math.inf

    @property
    def allowed_cutoff_u(self) -> float:
        """Latest play second a read may reach now."""
        return self._as_of - self._delta

    def advance(self, as_of_u: float) -> None:
        """Set the current tick (play seconds); reads may then reach ``as_of_u - delta``."""
        self._as_of = float(as_of_u)

    def check_read(self, player_id: str, cutoff_u: float) -> None:
        """Raise ``LookAheadError`` if a read up to ``cutoff_u`` (play seconds) is not allowed."""
        if cutoff_u > self.allowed_cutoff_u + _EPS_S:
            raise LookAheadError(
                f"read of {player_id or 'a buffer'} up to play second {cutoff_u:.3f} at as_of "
                f"{self._as_of:.3f} (allowed up to {self.allowed_cutoff_u:.3f})"
            )


@dataclass(frozen=True)
class ReplaySettings:
    """Simulation knobs. Times in seconds; ``until_s`` and ``log_flush_s`` on the play clock."""

    #: Pacing in multiples of real time; ``inf`` runs as fast as possible.
    speed: float = math.inf
    #: Source-to-ingest delay of every sample and event.
    latency_s: float = 5.0
    #: Share of samples lost at random.
    dropout_rate: float = 0.0
    #: One contiguous loss of this length per player and period, at a random start.
    burst_s: float = 0.0
    #: Offset of the source clock reached at the end of the match (grows linearly from 0).
    clock_drift_s: float = 0.0
    #: Delay from a feature tick to the screen.
    push_delay_s: float = 1.0
    #: Ingest queue size; when more samples arrive in one feature tick, the oldest are dropped.
    queue_capacity: int = 100_000
    #: Stop after this play second (None: the whole match).
    until_s: float | None = None
    #: Log store part files are written every this many play seconds and at the end.
    log_flush_s: float = 600.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.speed > 0.0:
            raise ValueError("speed must be > 0 (inf for unpaced)")
        if min(self.latency_s, self.burst_s, self.push_delay_s) < 0.0:
            raise ValueError("latency_s, burst_s and push_delay_s must be >= 0")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if self.queue_capacity < 1 or not self.log_flush_s > 0.0:
            raise ValueError("queue_capacity >= 1 and log_flush_s > 0 required")
        if self.until_s is not None and not self.until_s > 0.0:
            raise ValueError("until_s must be > 0")


@dataclass(frozen=True)
class ReplayResult:
    """Everything the live path produced, as flat frames (``storage.logstore.records_frame``).

    ``features``: every 10 s feature snapshot (play-clock times) with the live match clock
    (``match_period``, ``match_t_s``). ``risks``: every risk row with ``team_id``. ``alerts``:
    every alert change with ``team_id`` and ``change`` (opened, upgraded, cleared).
    ``displayed``: what the screen showed and when (one row per player and push).
    ``delivered`` / ``delivered_events``: the source stream as ingested, with wall and ingest
    times (ns); ``n_lost`` counts samples lost at the source, ``n_queue_dropped`` at the queue.
    """

    features: pd.DataFrame
    risks: pd.DataFrame
    alerts: pd.DataFrame
    displayed: pd.DataFrame
    delivered: pd.DataFrame
    delivered_events: pd.DataFrame
    n_lost: int
    n_queue_dropped: int


class MatchRecord(Protocol):
    """What a replay needs from a match: ``SynthMatch`` qualifies, as will recorded matches.

    ``samples`` has the ``TrackingSample`` columns; ``vmax_mps`` holds each player's peak speed
    from past matches.
    """

    @property
    def match_id(self) -> str: ...

    @property
    def lineups(self) -> tuple[Lineup, ...]: ...

    @property
    def events(self) -> tuple[MatchEvent, ...]: ...

    @property
    def samples(self) -> pd.DataFrame: ...

    @property
    def prematch(self) -> tuple[PreMatchRisk, ...]: ...

    @property
    def vmax_mps(self) -> dict[str, float]: ...


def _play_seconds(period: IntArray, t_s: FloatArray, starts: Mapping[int, float]) -> FloatArray:
    """Play seconds of match-clock stamps (vectorised ``(period, t_s) -> u``)."""
    start = np.asarray([0.0, *(starts[p] for p in (1, 2, 3, 4))])
    origin = np.asarray([0.0, *(PERIOD_START_S[p] for p in (1, 2, 3, 4))])
    return np.asarray(start[period] + (t_s - origin[period]), dtype=np.float64)


def _period_layout(match: MatchRecord) -> tuple[dict[int, float], float]:
    """Play second at which each period starts, and the play second at which the match ends.

    Lengths come from the period_end events (standard lengths otherwise); without any
    period_end the match ends with its last sample.
    """
    lengths = dict(_STANDARD_LEN_S)
    ended = [e.period for e in match.events if e.type == EventType.PERIOD_END]
    for e in match.events:
        if e.type == EventType.PERIOD_END:
            lengths[e.period] = e.t_s - PERIOD_START_S[e.period]
    starts = {1: 0.0}
    for p in (2, 3, 4):
        starts[p] = starts[p - 1] + lengths[p - 1]
    if ended:
        last = max(ended)
        return starts, starts[last] + lengths[last]
    s = match.samples
    if s.shape[0] == 0:
        return starts, 0.0
    u = _play_seconds(
        s["period"].to_numpy(dtype=np.int64), s["t_s"].to_numpy(dtype=np.float64), starts
    )
    return starts, float(np.max(u))


def _break_before(period: int) -> float:
    """Wall-clock seconds of breaks before ``period`` starts."""
    return sum(_BREAK_BEFORE_S[p] for p in range(1, period + 1))


def _epoch_ns(moment: datetime) -> int:
    delta = moment - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * _NS + delta.microseconds * 1000


@dataclass(frozen=True)
class _Feed:
    """The source stream after injection. Samples are sorted by arrival; events too."""

    samples: pd.DataFrame
    ingest_ns: IntArray
    events: tuple[MatchEvent, ...]
    event_wall_ns: tuple[int, ...]
    event_ingest_ns: tuple[int, ...]
    n_lost: int


def _source_feed(
    match: MatchRecord,
    starts: Mapping[int, float],
    end_u: float,
    match_end_u: float,
    kickoff_ns: int,
    settings: ReplaySettings,
) -> _Feed:
    """Inject dropout, bursts, drift and latency; stamp wall and ingest times (nanoseconds).

    Bursts fall inside the replayed span (play seconds up to ``end_u``); the drift grows to
    ``clock_drift_s`` at the end of the match (``match_end_u``) whatever the span.
    """
    rng = np.random.default_rng(settings.seed)
    s = match.samples
    n = s.shape[0]
    period = s["period"].to_numpy(dtype=np.int64)
    t_s = s["t_s"].to_numpy(dtype=np.float64)
    u = _play_seconds(period, t_s, starts)
    brk = np.asarray([0.0, *(_break_before(p) for p in (1, 2, 3, 4))])
    wall_ns = kickoff_ns + np.round((u + brk[period]) * _NS).astype(np.int64)
    keep = rng.random(n) >= settings.dropout_rate
    if settings.burst_s > 0.0 and n:
        groups = s.groupby(["player_id", "period"]).indices
        for key in sorted(groups, key=str):
            idx = groups[key]
            p = int(cast(tuple[str, int], key)[1])
            lo = starts[p]
            hi = min(starts[p + 1] if p < 4 else end_u, end_u) - settings.burst_s
            b0 = float(rng.uniform(lo, max(lo, hi)))
            ui = u[idx]
            keep[idx[(ui >= b0) & (ui < b0 + settings.burst_s)]] = False
    origin = np.asarray([0.0, *(PERIOD_START_S[p] for p in (1, 2, 3, 4))])
    stamped = np.maximum(t_s + settings.clock_drift_s * u / max(match_end_u, 1.0), origin[period])
    latency_ns = round(settings.latency_s * _NS)
    ingest_all = wall_ns + latency_ns
    order = np.flatnonzero(keep)[np.argsort(ingest_all[keep], kind="stable")]
    out = s.iloc[order].copy()
    out["t_s"] = stamped[order]
    out["wall_time_ns"] = pd.array(wall_ns[order], dtype="Int64")
    out["ingest_time_ns"] = pd.array(ingest_all[order], dtype="Int64")

    ev_wall = [
        kickoff_ns
        + round(
            (starts[e.period] + e.t_s - PERIOD_START_S[e.period] + _break_before(e.period)) * _NS
        )
        for e in match.events
    ]
    ev_order = sorted(range(len(ev_wall)), key=lambda i: ev_wall[i])
    return _Feed(
        samples=out.reset_index(drop=True),
        ingest_ns=np.asarray(ingest_all[order], dtype=np.int64),
        events=tuple(match.events[i] for i in ev_order),
        event_wall_ns=tuple(ev_wall[i] for i in ev_order),
        event_ingest_ns=tuple(ev_wall[i] + latency_ns for i in ev_order),
        n_lost=n - int(keep.sum()),
    )


class _LiveClock:
    """Match clock as known from the events delivered so far (no truth from the future)."""

    def __init__(self, match_id: str) -> None:
        self._match_id = match_id
        self._len: dict[int, float] = {}
        self._period = 1
        self._stoppage: dict[int, float] = {}

    def observe(self, event: MatchEvent) -> None:
        if event.type == EventType.PERIOD_END:
            self._len[event.period] = event.t_s - PERIOD_START_S[event.period]
        elif event.type == EventType.KICKOFF:
            self._period = max(self._period, event.period)
        elif event.type == EventType.STOPPAGE_ANNOUNCED and event.stoppage_min is not None:
            self._stoppage[event.period] = event.stoppage_min

    def state(self, u_s: float) -> MatchState:
        """Match state at play second ``u_s`` given what is known."""
        p = self._period
        start = sum(self._len.get(q, _STANDARD_LEN_S[q]) for q in range(1, p))
        return MatchState(
            match_id=self._match_id,
            period=p,
            t_s=PERIOD_START_S[p] + max(u_s - start, 0.0),
            stoppage_1h_min=self._stoppage.get(1),
            stoppage_2h_min=self._stoppage.get(2),
        )


def _opt(value: float) -> float | None:
    return None if math.isnan(value) else value


class _Ingest:
    """Turns feed rows into ``TrackingSample`` contracts; tracks each player's newest sample."""

    def __init__(self, feed: pd.DataFrame) -> None:
        self._match = feed["match_id"].to_numpy()
        self._player = feed["player_id"].to_numpy()
        self._team = feed["team_id"].to_numpy()
        self._period = feed["period"].to_numpy(dtype=np.int64)
        self._t = feed["t_s"].to_numpy(dtype=np.float64)
        self._x = feed["x_m"].to_numpy(dtype=np.float64, na_value=np.nan)
        self._y = feed["y_m"].to_numpy(dtype=np.float64, na_value=np.nan)
        self._v = feed["speed_mps"].to_numpy(dtype=np.float64, na_value=np.nan)
        self._src = feed["source"].to_numpy()
        self._det = feed["detected"].to_numpy(dtype=np.bool_)
        self._idn = feed["identity_confidence"].to_numpy(dtype=np.float64)
        self._wall = feed["wall_time_ns"].to_numpy(dtype=np.int64)
        self._ingest = feed["ingest_time_ns"].to_numpy(dtype=np.int64)
        #: Production wall time (ns) of each player's newest sample received.
        self.newest_wall_ns: dict[str, int] = {}

    def rows(self, lo: int, hi: int) -> list[TrackingSample]:
        """Rows [lo, hi) of the feed as validated contracts (columns converted once per call)."""
        cols = zip(
            self._match[lo:hi].tolist(),
            self._player[lo:hi].tolist(),
            self._team[lo:hi].tolist(),
            self._period[lo:hi].tolist(),
            self._t[lo:hi].tolist(),
            self._x[lo:hi].tolist(),
            self._y[lo:hi].tolist(),
            self._v[lo:hi].tolist(),
            self._src[lo:hi].tolist(),
            self._det[lo:hi].tolist(),
            self._idn[lo:hi].tolist(),
            self._wall[lo:hi].tolist(),
            self._ingest[lo:hi].tolist(),
            strict=True,
        )
        out = [
            TrackingSample(
                match_id=m,
                player_id=p,
                team_id=team,
                period=period,
                t_s=t,
                x_m=_opt(x),
                y_m=_opt(y),
                speed_mps=_opt(v),
                source=Source(src),
                detected=det,
                identity_confidence=idn,
                wall_time_ns=wall,
                ingest_time_ns=ing,
            )
            for m, p, team, period, t, x, y, v, src, det, idn, wall, ing in cols
        ]
        newest = self.newest_wall_ns
        for s in out:
            if s.wall_time_ns is not None and s.wall_time_ns > newest.get(s.player_id, -1):
                newest[s.player_id] = s.wall_time_ns
        return out


def display_state(age_s: float, grey_s: float, hide_s: float) -> str:
    """Screen state of a value of this age (docs/06 §4): live, grey or hidden."""
    if age_s > hide_s:
        return HIDDEN
    if age_s > grey_s:
        return GREY
    return LIVE


class _Replay:
    """State of one replay run; ``run`` drives the ticks."""

    def __init__(
        self,
        match: MatchRecord,
        fitted: FittedHazard,
        cfg: Config,
        settings: ReplaySettings,
        norms: NormTable,
        store: LogStore | None,
        feature_engine: type[StreamFeatureEngine],
    ) -> None:
        eng = cfg.engine
        if eng.risk_tick_s % eng.feature_tick_s:
            raise ValueError("engine.risk_tick_s must be a multiple of engine.feature_tick_s")
        self._cfg = cfg
        self._settings = settings
        self._store = store
        self._delta = float(eng.blanking_delta_s)
        self._tick_s = float(eng.feature_tick_s)
        self._risk_every = eng.risk_tick_s // eng.feature_tick_s
        self._mid = match.match_id
        players = [(lu.team_id, p) for lu in match.lineups for p in lu.players]
        self._team_of = {p.player_id: team for team, p in players}
        self._teams = sorted({lu.team_id for lu in match.lineups})
        self._kickoff_ns = _epoch_ns(match.lineups[0].kickoff_utc)
        self._starts, match_end_u = _period_layout(match)
        self._end_u = match_end_u
        if settings.until_s is not None:
            self._end_u = min(self._end_u, float(settings.until_s))
        self._boundaries = sorted(self._starts[p] for p in (2, 3, 4))
        self._feed = _source_feed(
            match, self._starts, self._end_u, match_end_u, self._kickoff_ns, settings
        )
        self._ingest = _Ingest(self._feed.samples)
        self._delivered = np.zeros(self._feed.samples.shape[0], dtype=np.bool_)

        self._guard = LookAheadGuard(self._delta)
        self._features = feature_engine(
            cfg, self._mid, dict(match.vmax_mps), read_guard=self._guard
        )
        self._risk = RiskEngine(
            cfg,
            fitted,
            norms,
            {r.player_id: r for r in match.prematch},
            {p.player_id: p.position_group for _, p in players},
        )
        self._alerts = {t: AlertLogic(cfg, self._mid, id_prefix=f"{t}-") for t in self._teams}
        self._seen_alerts: set[str] = set()
        self._clock = _LiveClock(self._mid)
        self._entry_ns = {p.player_id: self._kickoff_ns for _, p in players if p.starting}
        self._latest_risk: dict[str, RiskSnapshot] = {}

        self._si = 0
        self._ei = 0
        self._n_queue_dropped = 0
        self.feature_rows: list[tuple[FeatureSnapshot, str, int, float]] = []
        self.risk_rows: list[RiskSnapshot] = []
        self.alert_rows: list[tuple[Alert, str]] = []
        self.display_rows: list[dict[str, object]] = []
        self.event_rows: list[tuple[MatchEvent, int, int]] = []
        self._flushed = {"tracking": 0, "events": 0, "features": 0, "risk": 0, "alerts": 0}
        self._flushed_display = 0

    def _tick_wall_ns(self, u: float) -> int:
        """Wall time of a tick; a tick on a period boundary belongs to the period ending there."""
        period = 1 + bisect.bisect_left(self._boundaries, u - _EPS_S)
        return self._kickoff_ns + round((u + _break_before(period)) * _NS)

    def _push_samples(self, hi: int) -> None:
        if hi > self._si:
            self._features.push_samples(self._ingest.rows(self._si, hi))
            self._delivered[self._si : hi] = True
            self._si = hi

    def _deliver(self, now_ns: int) -> None:
        """Move everything that has arrived by ``now_ns`` into the feature engine, in arrival
        order (an event goes before samples that arrive at the same instant)."""
        feed = self._feed
        due = int(np.searchsorted(feed.ingest_ns, now_ns, side="right"))
        overflow = due - self._si - self._settings.queue_capacity
        if overflow > 0:
            self._n_queue_dropped += overflow
            self._si += overflow
        while self._ei < len(feed.events) and feed.event_ingest_ns[self._ei] <= now_ns:
            arrive = feed.event_ingest_ns[self._ei]
            self._push_samples(int(np.searchsorted(feed.ingest_ns, arrive, side="left")))
            event = feed.events[self._ei]
            self._features.push_event(event)
            self._clock.observe(event)
            if event.type == EventType.SUBSTITUTION and event.player_in_id is not None:
                self._entry_ns[event.player_in_id] = feed.event_wall_ns[self._ei]
            self.event_rows.append((event, feed.event_wall_ns[self._ei], arrive))
            self._ei += 1
        self._push_samples(due)

    def _ages(self, snaps: list[FeatureSnapshot], now_ns: int) -> dict[str, float]:
        """D-021: wall-clock age of each player's newest received sample, else since entry."""
        newest = self._ingest.newest_wall_ns
        out: dict[str, float] = {}
        for snap in snaps:
            pid = snap.player_id
            ref = max(newest.get(pid, -1), self._entry_ns.get(pid, self._kickoff_ns))
            out[pid] = max(0.0, (now_ns - ref) / _NS)
        return out

    def _risk_update(
        self, snaps: list[FeatureSnapshot], state: MatchState, ages: dict[str, float]
    ) -> None:
        update = self._risk.update(snaps, state, ages)
        self.risk_rows.extend(update.risks)
        for snap in update.features:
            self.feature_rows.append((snap, "risk_tick", state.period, state.t_s))
        for row in update.risks:
            self._latest_risk[row.player_id] = row
        on_pitch = {s.player_id for s in snaps}
        for team in self._teams:
            rows = [r for r in update.risks if self._team_of.get(r.player_id) == team]
            team_on = {p for p in on_pitch if self._team_of.get(p) == team}
            for alert in self._alerts[team].update(rows, team_on):
                if alert.cleared_t_s is not None:
                    change = "cleared"
                elif alert.alert_id in self._seen_alerts:
                    change = "upgraded"
                else:
                    change = "opened"
                    self._seen_alerts.add(alert.alert_id)
                self.alert_rows.append((alert, change))

    def _push_screen(
        self,
        u: float,
        snaps: list[FeatureSnapshot],
        state: MatchState,
        ages: dict[str, float],
        now_ns: int,
    ) -> None:
        """One push: the latest risk row of each on-pitch player with his current data age."""
        q = self._cfg.quality
        delay = self._settings.push_delay_s
        open_level = {
            a.player_id: a.level.value for t in self._teams for a in self._alerts[t].open_alerts()
        }
        for snap in snaps:
            row = self._latest_risk.get(snap.player_id)
            if row is None:
                continue
            age = ages[snap.player_id] + delay
            shown = display_state(age, float(q.stale_grey_s), float(q.stale_hide_s))
            numbers = shown != HIDDEN
            rr_ok = numbers and row.quality not in (Quality.PREMATCH_ONLY, Quality.STALE)
            self.display_rows.append(
                {
                    "match_id": self._mid,
                    "team_id": self._team_of.get(row.player_id),
                    "player_id": row.player_id,
                    "play_s": u,
                    "shown_wall_ns": now_ns + round(delay * _NS),
                    "match_period": state.period,
                    "match_t_s": state.t_s,
                    "risk_as_of_t_s": row.as_of_t_s,
                    "data_age_s": age,
                    "state": shown,
                    "quality": row.quality.value,
                    "prematch_band": row.prematch_band.value,
                    "rr": row.rr if rr_ok else None,
                    "p15": row.p15 if numbers else None,
                    "r_rest": row.r_rest if numbers else None,
                    "r_rest_lo": row.r_rest_lo if numbers else None,
                    "r_rest_hi": row.r_rest_hi if numbers else None,
                    "r_usual": row.r_usual if numbers else None,
                    "reason_code": row.top_reason_code if numbers else None,
                    "alert_level": open_level.get(row.player_id),
                }
            )

    def _flush(self) -> None:
        store = self._store
        if store is None:
            return
        done = self._flushed
        lo, hi = done["tracking"], self._si
        rows = self._feed.samples.iloc[lo:hi]
        store.append_frame("tracking", rows.loc[self._delivered[lo:hi]])
        done["tracking"] = hi
        events = self.event_rows[done["events"] :]
        store.append(
            "events",
            [e for e, _, _ in events],
            wall_time_ns=[w for _, w, _ in events],
            ingest_time_ns=[i for _, _, i in events],
        )
        done["events"] = len(self.event_rows)
        feats = self.feature_rows[done["features"] :]
        store.append(
            "features",
            [f[0] for f in feats],
            stage=[f[1] for f in feats],
            match_period=[f[2] for f in feats],
            match_t_s=[f[3] for f in feats],
        )
        done["features"] = len(self.feature_rows)
        risks = self.risk_rows[done["risk"] :]
        store.append("risk", risks, team_id=[self._team_of.get(r.player_id) for r in risks])
        done["risk"] = len(self.risk_rows)
        alerts = self.alert_rows[done["alerts"] :]
        store.append(
            "alerts",
            [a for a, _ in alerts],
            team_id=[self._team_of.get(a.player_id) for a, _ in alerts],
            change=[c for _, c in alerts],
        )
        done["alerts"] = len(self.alert_rows)
        shown = self.display_rows[self._flushed_display :]
        if shown:
            store.append_frame("display", pd.DataFrame(shown))
        self._flushed_display = len(self.display_rows)
        store.flush()

    def run(self) -> ReplayResult:
        settings = self._settings
        host_start = time.monotonic()
        n_ticks = math.floor(self._end_u / self._tick_s + _EPS_S)
        for k in range(1, n_ticks + 1):
            u = k * self._tick_s
            now_ns = self._tick_wall_ns(u)
            self._deliver(now_ns)
            self._guard.advance(u)
            snaps = self._features.tick(u)
            for snap in snaps:
                if snap.as_of_t_s > u + _EPS_S or snap.cutoff_t_s > u - self._delta + _EPS_S:
                    raise LookAheadError(
                        f"snapshot of {snap.player_id} stamped {snap.as_of_t_s} with cutoff "
                        f"{snap.cutoff_t_s} at tick {u}"
                    )
            state = self._clock.state(u)
            ages = self._ages(snaps, now_ns)
            self.feature_rows.extend((s, "feature_tick", state.period, state.t_s) for s in snaps)
            if k % self._risk_every == 0:
                self._risk_update(snaps, state, ages)
            self._push_screen(u, snaps, state, ages, now_ns)
            if u % settings.log_flush_s < self._tick_s - _EPS_S:
                self._flush()
            if math.isfinite(settings.speed):
                wait = host_start + u / settings.speed - time.monotonic()
                if wait > 0.0:
                    time.sleep(wait)
        self._flush()
        return self._result()

    def _result(self) -> ReplayResult:
        ticks = [f for f in self.feature_rows if f[1] == "feature_tick"]
        delivered = self._feed.samples.iloc[: self._si]
        return ReplayResult(
            features=records_frame(
                [f[0] for f in ticks],
                match_period=[f[2] for f in ticks],
                match_t_s=[f[3] for f in ticks],
            ),
            risks=records_frame(
                self.risk_rows, team_id=[self._team_of.get(r.player_id) for r in self.risk_rows]
            ),
            alerts=records_frame(
                [a for a, _ in self.alert_rows],
                team_id=[self._team_of.get(a.player_id) for a, _ in self.alert_rows],
                change=[c for _, c in self.alert_rows],
            ),
            displayed=pd.DataFrame(self.display_rows),
            delivered=delivered.loc[self._delivered[: self._si]].reset_index(drop=True),
            delivered_events=records_frame(
                [e for e, _, _ in self.event_rows],
                wall_time_ns=[w for _, w, _ in self.event_rows],
                ingest_time_ns=[i for _, _, i in self.event_rows],
            ),
            n_lost=self._feed.n_lost,
            n_queue_dropped=self._n_queue_dropped,
        )


def replay_match(
    match: MatchRecord,
    fitted: FittedHazard,
    cfg: Config,
    settings: ReplaySettings,
    *,
    norms: NormTable,
    store: LogStore | None = None,
    feature_engine: type[StreamFeatureEngine] = StreamFeatureEngine,
) -> ReplayResult:
    """Run the live path on a recorded or synthetic match with a simulated clock.

    A feature snapshot at play second u uses only samples stamped up to u - delta that have
    arrived by the tick's wall time; any read beyond raises ``LookAheadError``. Risk rows use the
    match clock known from the delivered events and the data ages of D-021; alerts run per team.
    With a ``store``, every delivered sample and event, feature snapshot (10 s ticks, plus the
    z-enriched snapshots of each risk update), risk row, alert change and screen row is appended
    to it. ``feature_engine`` lets a test substitute the feature stage. Deterministic given its
    inputs (``settings.seed`` drives the source injection).
    """
    return _Replay(match, fitted, cfg, settings, norms, store, feature_engine).run()
