"""Stream feature engine (docs/02_MATH.md §10).

Rule: every function takes ``as_of_t_s`` and only reads samples with ``t_s <= as_of_t_s - delta``.
The online engine must reproduce ``compute_features_offline`` exactly.

Clock convention
----------------
The match clock ``t_s`` of docs/03_DATA_CONTRACTS.md §1 is ambiguous across a period break:
first-half stoppage runs past 2700 s, where the second half restarts, so a bare ``t_s`` does not
order samples across periods. This module works on the **play clock** ``u``: seconds of play
since the first kick-off, with the breaks between periods removed (the same convention as the
synthetic generator's internal clock). ``u`` is strictly increasing in real time, so the blanking
rule (features at ``as_of`` use only data with ``u <= as_of - delta``) is a plain comparison.
``StreamFeatureEngine.tick`` and ``compute_features_offline`` take ``as_of_t_s`` on the play
clock, and ``FeatureSnapshot.as_of_t_s`` / ``cutoff_t_s`` carry play-clock seconds; ``period``
is the period at ``as_of``. Period lengths come from ``period_end`` events seen so far; a period
without a known end uses its standard length (2700 s, or 900 s for extra time), which only
affects the cosmetic ``period`` label of ticks in stoppage time until the ``period_end`` arrives.

Speeds come from Savitzky-Golay-smoothed positions (docs/02_MATH.md §10); when a source provides
no positions, its speed is smoothed with the same window instead. Samples with ``detected=False``
or with a smoothed speed above ``features.max_plausible_speed_mps`` are invalid: they never
contribute to features and they lower the per-window coverage.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal

from livrisk.config import Config
from livrisk.contracts import (
    PERIOD_START_S,
    EventType,
    FeatureSnapshot,
    MatchEvent,
    Quality,
    TrackingSample,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]

#: Feature windows of docs/02_MATH.md §10, seconds.
_W5_S: float = 300.0
_W10_S: float = 600.0
_W15_S: float = 900.0

#: Raw features computed here, in the order of docs/02_MATH.md §10 (z-scores arrive in M4).
FEATURE_NAMES: tuple[str, ...] = (
    "hsr_5m",
    "spr_5m",
    "recovery_10m",
    "hir_drop_15m",
    "nearmax_cum",
    "disrupt_5m",
)

#: Tracking-derived features; "no live feature valid" (prematch_only) is judged on these, and
#: identity confidence gates them downstream (the risk engine). disrupt_5m comes from the event
#: feed, not from tracking.
TRACKING_FEATURES: tuple[str, ...] = (
    "hsr_5m",
    "spr_5m",
    "recovery_10m",
    "hir_drop_15m",
    "nearmax_cum",
)

#: Event types counting toward disrupt_5m (docs/02_MATH.md §10, F6).
_DISRUPT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.GOAL,
        EventType.YELLOW_CARD,
        EventType.RED_CARD,
        EventType.INJURY_STOPPAGE,
    }
)

#: Assumed period length before its period_end event is seen, seconds.
_STANDARD_PERIOD_LEN_S: dict[int, float] = {1: 2700.0, 2: 2700.0, 3: 900.0, 4: 900.0}

#: Tolerance for clock arithmetic, seconds (the look-ahead guard, effort durations, grids).
_EPS_S: float = 1e-9

#: Speed source of a resampled series: smoothed positions, smoothed provided speed, or neither
#: complete (fail safe: no usable speed series).
_PATH_POS, _PATH_SPD, _PATH_NONE = 0, 1, 2


@dataclass(frozen=True)
class Effort:
    """A contiguous run above a speed threshold (docs/02_MATH.md §10).

    ``start_s`` / ``end_s`` bracket the run on the given time axis: the end is the first sample
    after the run (one median interval past the last sample when the run reaches the series end).
    ``distance_m`` integrates the given speeds over the whole bracket, merged gaps included.
    """

    start_s: float
    end_s: float
    peak_mps: float
    distance_m: float


# --------------------------------------------------------------------------------------------
# Play clock
# --------------------------------------------------------------------------------------------


class _PeriodClock:
    """Map between (period, t_s) and play seconds ``u``; see the module docstring.

    The offline reference and the online engine use this same class, so their clock views agree
    at every tick provided they have seen the same events.
    """

    def __init__(self) -> None:
        self._len: dict[int, float] = {}

    def observe(self, event_type: EventType, period: int, t_s: float) -> None:
        """Learn a period length from its period_end event (t_s on the match clock, seconds)."""
        if event_type == EventType.PERIOD_END:
            self._len[period] = t_s - PERIOD_START_S[period]

    def _length(self, period: int) -> float:
        return self._len.get(period, _STANDARD_PERIOD_LEN_S[period])

    def start_u(self, period: int) -> float:
        """Play second at which ``period`` starts."""
        return sum(self._length(p) for p in range(1, period))

    def to_u(self, period: int, t_s: float) -> float:
        """Play seconds of a sample or event given on the match clock."""
        return self.start_u(period) + (t_s - PERIOD_START_S[period])

    def period_at_u(self, u_s: float) -> int:
        """Period containing play second ``u_s`` (clamped to 1-4)."""
        period = 1
        for p in (2, 3, 4):
            if u_s >= self.start_u(p):
                period = p
        return period


# --------------------------------------------------------------------------------------------
# Effort detection (docs/02_MATH.md §10)
# --------------------------------------------------------------------------------------------


def _median_dt(t_s: FloatArray) -> float:
    """Median sampling interval, seconds; NaN with fewer than 2 samples."""
    if t_s.size < 2:
        return math.nan
    return float(np.median(np.diff(t_s)))


def _effort_arrays(
    t_s: FloatArray,
    speed_mps: FloatArray,
    threshold_mps: float,
    min_duration_s: float,
    merge_gap_s: float,
    default_dt_s: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Vectorised effort detection; returns (start_s, end_s, peak_mps, distance_m) per effort.

    A run is a maximal contiguous block of samples with speed >= threshold (NaN counts as below
    threshold). Runs separated by a below-threshold gap shorter than ``merge_gap_s`` are merged;
    the gap duration is the time from the first below-threshold sample after a run to the first
    sample of the next. A (merged) effort is kept when its bracket ``end_s - start_s`` lasts at
    least ``min_duration_s`` (with 1e-9 s tolerance for clock arithmetic).
    """
    empty = np.zeros(0)
    n = t_s.size
    if n == 0:
        return empty, empty, empty, empty
    above: BoolArray = np.asarray(speed_mps >= threshold_mps)
    if not bool(above.any()):
        return empty, empty, empty, empty
    turns = np.diff(above.astype(np.int8))
    a_idx = np.flatnonzero(turns == 1) + 1
    b_idx = np.flatnonzero(turns == -1)
    if above[0]:
        a_idx = np.concatenate([[0], a_idx])
    if above[-1]:
        b_idx = np.concatenate([b_idx, [n - 1]])
    if a_idx.size > 1:
        gaps = t_s[a_idx[1:]] - t_s[b_idx[:-1] + 1]
        split = np.flatnonzero(gaps >= merge_gap_s) + 1
        grp_lo = np.concatenate([[0], split])
        grp_hi = np.concatenate([split, [a_idx.size]])
    else:
        grp_lo = np.zeros(1, dtype=np.intp)
        grp_hi = np.ones(1, dtype=np.intp)
    a_g = a_idx[grp_lo]
    b_g = b_idx[grp_hi - 1]
    nxt = b_g + 1
    starts = t_s[a_g]
    ends = np.where(nxt < n, t_s[np.minimum(nxt, n - 1)], t_s[b_g] + default_dt_s)
    keep = (ends - starts) >= (min_duration_s - _EPS_S)
    step = np.concatenate([np.diff(t_s), [default_dt_s]])
    peaks = np.asarray(
        [float(np.nanmax(speed_mps[a : b + 1])) for a, b in zip(a_g[keep], b_g[keep], strict=True)]
    )
    dists = np.asarray(
        [
            float(np.nansum(speed_mps[a : b + 1] * step[a : b + 1]))
            for a, b in zip(a_g[keep], b_g[keep], strict=True)
        ]
    )
    return (
        np.asarray(starts[keep], dtype=np.float64),
        np.asarray(ends[keep], dtype=np.float64),
        np.asarray(peaks, dtype=np.float64),
        np.asarray(dists, dtype=np.float64),
    )


def detect_efforts(
    t_s: FloatArray,
    speed_mps: FloatArray,
    threshold_mps: float,
    min_duration_s: float,
    merge_gap_s: float,
) -> list[Effort]:
    """Efforts with speed >= threshold lasting >= min_duration_s; gaps < merge_gap_s merged.

    ``t_s`` must be strictly increasing (seconds); NaN speeds count as below threshold. See
    ``_effort_arrays`` for the exact bracket and duration conventions.
    """
    t = np.asarray(t_s, dtype=np.float64)
    v = np.asarray(speed_mps, dtype=np.float64)
    if t.size != v.size:
        raise ValueError(f"t_s and speed_mps must have the same length, got {t.size} and {v.size}")
    dt = _median_dt(t)
    starts, ends, peaks, dists = _effort_arrays(
        t, v, threshold_mps, min_duration_s, merge_gap_s, dt if math.isfinite(dt) else 0.0
    )
    return [
        Effort(float(s), float(e), float(p), float(d))
        for s, e, p, d in zip(starts, ends, peaks, dists, strict=True)
    ]


# --------------------------------------------------------------------------------------------
# Resampling and smoothing (docs/02_MATH.md §10)
# --------------------------------------------------------------------------------------------


def _grid_size(u0: float, u_last: float, dt_s: float) -> int:
    """Number of regular grid points from ``u0`` with step ``dt_s`` not exceeding ``u_last``."""
    return math.floor((u_last - u0) / dt_s + _EPS_S) + 1


def _savgol_window(window_s: float, dt_s: float, polyorder: int, n_points: int) -> int:
    """Odd Savitzky-Golay window length in samples for a window of ``window_s`` seconds.

    Shrinks to the series length when the series is shorter; 0 when the series is too short for
    the polynomial order (smoothing then yields NaN).
    """
    w = max(round(window_s / dt_s), polyorder + 1)
    if w % 2 == 0:
        w += 1
    if w > n_points:
        w = n_points if n_points % 2 == 1 else n_points - 1
    return w if w > polyorder else 0


def _smooth_positions(
    gx: FloatArray, gy: FloatArray, window: int, polyorder: int, dt_s: float
) -> FloatArray:
    """Speed (m/s) from the Savitzky-Golay derivative of both grid-space position series."""
    if window == 0:
        return np.full(gx.shape, np.nan)
    dx = np.asarray(signal.savgol_filter(gx, window, polyorder, deriv=1, delta=dt_s), np.float64)
    dy = np.asarray(signal.savgol_filter(gy, window, polyorder, deriv=1, delta=dt_s), np.float64)
    return np.asarray(np.hypot(dx, dy), dtype=np.float64)


def _smooth_speeds(gs: FloatArray, window: int, polyorder: int, dt_s: float) -> FloatArray:
    """Provided speed (m/s) smoothed with the same Savitzky-Golay window (docs/02_MATH.md §10)."""
    if window == 0:
        return np.full(gs.shape, np.nan)
    return np.asarray(signal.savgol_filter(gs, window, polyorder, deriv=0, delta=dt_s), np.float64)


def smooth_speed(
    t_s: FloatArray, x_m: FloatArray, y_m: FloatArray, window_s: float, polyorder: int
) -> FloatArray:
    """Savitzky-Golay smoothing of positions, then speed = |dp/dt| in m/s.

    Handles irregular sampling by resampling to the median sampling interval first, so the result
    lives on a regular grid starting at ``t_s[0]`` with that step (the same length as the input
    when the sampling is already regular). NaN where the series is too short for the polynomial
    order or the sampling interval is degenerate.
    """
    t = np.asarray(t_s, dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if t.size != x.size or t.size != y.size:
        raise ValueError("t_s, x_m and y_m must have the same length")
    if t.size == 0:
        return np.zeros(0)
    dt = _median_dt(t)
    if not math.isfinite(dt) or dt <= 0.0:
        return np.full(t.shape, np.nan)
    grid = t[0] + dt * np.arange(_grid_size(float(t[0]), float(t[-1]), dt), dtype=np.float64)
    gx = np.interp(grid, t, x)
    gy = np.interp(grid, t, y)
    win = _savgol_window(window_s, dt, polyorder, grid.size)
    return _smooth_positions(gx, gy, win, polyorder, dt)


def _nearest_samples(u_s: FloatArray, grid: FloatArray, dt_s: float) -> tuple[IntArray, BoolArray]:
    """Index of the sample nearest each grid point, and whether it lies within half an interval
    (grid points inside a sampling gap are not observed)."""
    idx = np.searchsorted(u_s, grid)
    right = np.minimum(idx, u_s.size - 1)
    left = np.maximum(idx.astype(np.intp) - 1, 0)
    d_left = np.abs(grid - u_s[left])
    d_right = np.abs(u_s[right] - grid)
    take_right = d_right < d_left
    near = np.where(take_right, right, left)
    dist = np.where(take_right, d_right, d_left)
    return near, dist <= 0.5 * dt_s * (1.0 + _EPS_S)


# --------------------------------------------------------------------------------------------
# Resampled per-player series
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _SeriesData:
    """One player's prefix resampled onto a regular play-clock grid.

    ``grid`` ends at the last incorporated sample; ``speed`` (m/s) is NaN where it cannot be
    computed; ``valid`` marks grid points that are observed (a real sample within half the
    sampling interval), detected, finite and physically plausible; ``ident`` is the identity
    confidence of the nearest sample. The cumulative arrays carry one leading zero, so a window
    sum over grid[lo:hi] is ``cum[hi] - cum[lo]``. ``src_a`` / ``src_b`` hold the interpolated
    grid-space series (x, y or speed, empty) that the incremental extension smooths.
    """

    n_samples: int
    first_u: float
    dt: float
    window: int
    path: int
    grid: FloatArray
    speed: FloatArray
    observed: BoolArray
    det: BoolArray
    valid: BoolArray
    ident: FloatArray
    cum_valid: FloatArray
    cum_hsr: FloatArray
    cum_spr: FloatArray
    src_a: FloatArray
    src_b: FloatArray


def _empty_series(n_samples: int = 0, first_u: float = math.nan) -> _SeriesData:
    e = np.zeros(0)
    eb = np.zeros(0, dtype=np.bool_)
    return _SeriesData(
        n_samples, first_u, math.nan, 0, _PATH_NONE, e, e, eb, eb, eb, e, e, e, e, e, e
    )


def _build_series(
    u_s: FloatArray,
    x_m: FloatArray,
    y_m: FloatArray,
    v_src: FloatArray,
    detected: BoolArray,
    ident: FloatArray,
    cfg: Config,
) -> _SeriesData:
    """Resample a sample prefix onto its regular grid and smooth the speed (m/s).

    This is the batch computation; ``_SampleBuffer`` extends its result incrementally with
    bit-identical values.
    """
    f = cfg.features
    n = int(u_s.size)
    if n == 0:
        return _empty_series()
    first_u = float(u_s[0])
    if n < 2:
        return _empty_series(n, first_u)
    dt = _median_dt(u_s)
    if not math.isfinite(dt) or dt <= 0.0:
        return _empty_series(n, first_u)
    grid = first_u + dt * np.arange(_grid_size(first_u, float(u_s[-1]), dt), dtype=np.float64)
    window = _savgol_window(f.smoothing_window_s, dt, f.smoothing_polyorder, grid.size)
    if bool(np.all(np.isfinite(x_m))) and bool(np.all(np.isfinite(y_m))):
        path = _PATH_POS
        src_a = np.interp(grid, u_s, x_m)
        src_b = np.interp(grid, u_s, y_m)
        speed = _smooth_positions(src_a, src_b, window, f.smoothing_polyorder, dt)
    elif bool(np.all(np.isfinite(v_src))):
        path = _PATH_SPD
        src_a = np.interp(grid, u_s, v_src)
        src_b = np.zeros(0)
        speed = _smooth_speeds(src_a, window, f.smoothing_polyorder, dt)
    else:
        path = _PATH_NONE
        src_a = np.zeros(0)
        src_b = np.zeros(0)
        speed = np.full(grid.shape, np.nan)
    near, observed = _nearest_samples(u_s, grid, dt)
    det = detected[near]
    ident_g = ident[near]
    valid = observed & det & np.isfinite(speed) & (speed <= f.max_plausible_speed_mps)
    zero = np.zeros(1)
    cum_valid = np.concatenate([zero, np.cumsum(valid.astype(np.float64))])
    hsr = np.where(valid & (speed >= f.hsr_threshold_mps), speed, 0.0) * dt
    spr = np.where(valid & (speed >= f.sprint_threshold_mps), speed, 0.0) * dt
    cum_hsr = np.concatenate([zero, np.cumsum(hsr)])
    cum_spr = np.concatenate([zero, np.cumsum(spr)])
    return _SeriesData(
        n,
        first_u,
        dt,
        window,
        path,
        grid,
        speed,
        observed,
        det,
        valid,
        ident_g,
        cum_valid,
        cum_hsr,
        cum_spr,
        src_a,
        src_b,
    )


def _extend_series(
    cache: _SeriesData,
    u_s: FloatArray,
    x_m: FloatArray,
    y_m: FloatArray,
    v_src: FloatArray,
    detected: BoolArray,
    ident: FloatArray,
    all_pos: bool,
    all_spd: bool,
    cfg: Config,
) -> _SeriesData | None:
    """Extend ``cache`` to a longer prefix of the same stream; None when a rebuild is required
    (sampling interval, smoothing window or speed source changed).

    A growing prefix changes smoothed values only within half a window of the old end
    (Savitzky-Golay edge fits), so the tail is recomputed and everything earlier is kept; the
    result is bit-identical to ``_build_series`` on the longer prefix.
    """
    f = cfg.features
    if cache.n_samples < 2 or cache.window == 0 or cache.grid.size == 0:
        return None
    dt = _median_dt(u_s)
    if not math.isfinite(dt) or dt <= 0.0 or dt != cache.dt:
        return None
    path = _PATH_POS if all_pos else (_PATH_SPD if all_spd else _PATH_NONE)
    if path != cache.path:
        return None
    n_grid = _grid_size(cache.first_u, float(u_s[-1]), dt)
    if n_grid < cache.grid.size:
        return None
    window = _savgol_window(f.smoothing_window_s, dt, f.smoothing_polyorder, n_grid)
    if window != cache.window:
        return None
    if n_grid == cache.grid.size:
        # New samples within the last grid interval: the grid and the series are unchanged.
        return _SeriesData(
            int(u_s.size),
            cache.first_u,
            dt,
            window,
            cache.path,
            cache.grid,
            cache.speed,
            cache.observed,
            cache.det,
            cache.valid,
            cache.ident,
            cache.cum_valid,
            cache.cum_hsr,
            cache.cum_spr,
            cache.src_a,
            cache.src_b,
        )
    n_old = cache.grid.size
    h = window // 2
    g_new = cache.first_u + dt * np.arange(n_old, n_grid, dtype=np.float64)
    if cache.path == _PATH_POS:
        src_a = np.concatenate([cache.src_a, np.interp(g_new, u_s, x_m)])
        src_b = np.concatenate([cache.src_b, np.interp(g_new, u_s, y_m)])
    elif cache.path == _PATH_SPD:
        src_a = np.concatenate([cache.src_a, np.interp(g_new, u_s, v_src)])
        src_b = cache.src_b
    else:
        src_a = cache.src_a
        src_b = cache.src_b
    tail_start = max(0, n_old - 2 * h)
    if cache.path == _PATH_POS:
        tail = _smooth_positions(
            src_a[tail_start:], src_b[tail_start:], window, f.smoothing_polyorder, dt
        )
    elif cache.path == _PATH_SPD:
        tail = _smooth_speeds(src_a[tail_start:], window, f.smoothing_polyorder, dt)
    else:
        tail = np.full(n_grid - tail_start, np.nan)
    r0 = tail_start + h
    speed = np.concatenate([cache.speed[:r0], tail[h:]])
    near, obs_new = _nearest_samples(u_s, g_new, dt)
    observed = np.concatenate([cache.observed, obs_new])
    det = np.concatenate([cache.det, detected[near]])
    ident_g = np.concatenate([cache.ident, ident[near]])
    grid = np.concatenate([cache.grid, g_new])
    valid = np.empty(n_grid, dtype=np.bool_)
    valid[:r0] = cache.valid[:r0]
    valid[r0:] = (
        observed[r0:]
        & det[r0:]
        & np.isfinite(speed[r0:])
        & (speed[r0:] <= f.max_plausible_speed_mps)
    )
    cum_valid = np.empty(n_grid + 1)
    cum_hsr = np.empty(n_grid + 1)
    cum_spr = np.empty(n_grid + 1)
    cum_valid[: r0 + 1] = cache.cum_valid[: r0 + 1]
    cum_hsr[: r0 + 1] = cache.cum_hsr[: r0 + 1]
    cum_spr[: r0 + 1] = cache.cum_spr[: r0 + 1]
    cum_valid[r0 + 1 :] = cum_valid[r0] + np.cumsum(valid[r0:].astype(np.float64))
    hsr = np.where(valid[r0:] & (speed[r0:] >= f.hsr_threshold_mps), speed[r0:], 0.0) * dt
    spr = np.where(valid[r0:] & (speed[r0:] >= f.sprint_threshold_mps), speed[r0:], 0.0) * dt
    cum_hsr[r0 + 1 :] = cum_hsr[r0] + np.cumsum(hsr)
    cum_spr[r0 + 1 :] = cum_spr[r0] + np.cumsum(spr)
    return _SeriesData(
        int(u_s.size),
        cache.first_u,
        dt,
        window,
        cache.path,
        grid,
        speed,
        observed,
        det,
        valid,
        ident_g,
        cum_valid,
        cum_hsr,
        cum_spr,
        src_a,
        src_b,
    )


class _GrowVec:
    """Float64 vector with doubling capacity; ``view()`` is a zero-copy prefix."""

    def __init__(self) -> None:
        self._buf = np.zeros(256)
        self._n = 0

    @property
    def n(self) -> int:
        return self._n

    def append(self, value: float) -> None:
        if self._n == self._buf.size:
            new = np.empty(self._buf.size * 2)
            new[: self._n] = self._buf[: self._n]
            self._buf = new
        self._buf[self._n] = value
        self._n += 1

    def view(self) -> FloatArray:
        return self._buf[: self._n]

    def replace(self, values: FloatArray) -> None:
        self._buf = values
        self._n = int(values.size)


class ReadGuard(Protocol):
    """Checks every buffer read; raises when ``cutoff_u`` (play seconds) is not allowed."""

    def check_read(self, player_id: str, cutoff_u: float) -> None: ...


class _SampleBuffer:
    """Append-mostly store of one player's samples with the cached resampled series.

    ``prefix(cutoff_u)`` is the only read path: it returns the series built from samples with
    u <= cutoff_u only (play seconds), extending the cache incrementally. The extension is
    bit-identical to a from-scratch build of the same prefix. A ``guard`` sees every read first.
    """

    def __init__(self, cfg: Config, player_id: str = "", guard: ReadGuard | None = None) -> None:
        self._cfg = cfg
        self._player_id = player_id
        self._guard = guard
        self._u = _GrowVec()
        self._x = _GrowVec()
        self._y = _GrowVec()
        self._v = _GrowVec()
        self._det = _GrowVec()
        self._idn = _GrowVec()
        self._sorted = True
        self._all_pos = True
        self._all_spd = True
        self._built_k = 0
        self._built_cutoff = -math.inf
        self._cache: _SeriesData | None = None

    @property
    def n_samples(self) -> int:
        return self._u.n

    def add(
        self, u_s: float, x_m: float, y_m: float, speed_mps: float, detected: float, identity: float
    ) -> None:
        """Append one sample; out-of-order arrival marks the buffer for a rebuild."""
        if self._u.n and u_s < self._u.view()[-1]:
            self._sorted = False
        self._u.append(u_s)
        self._x.append(x_m)
        self._y.append(y_m)
        self._v.append(speed_mps)
        self._det.append(detected)
        self._idn.append(identity)
        self._all_pos = self._all_pos and math.isfinite(x_m) and math.isfinite(y_m)
        self._all_spd = self._all_spd and math.isfinite(speed_mps)

    def _columns(
        self, k: int
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray, FloatArray]:
        det = np.asarray(self._det.view()[:k] > 0.5, dtype=np.bool_)
        return (
            self._u.view()[:k],
            self._x.view()[:k],
            self._y.view()[:k],
            self._v.view()[:k],
            det,
            self._idn.view()[:k],
        )

    def _sort(self) -> None:
        order = np.argsort(self._u.view(), kind="stable")
        for vec in (self._u, self._x, self._y, self._v, self._det, self._idn):
            vec.replace(np.asarray(vec.view()[order], dtype=np.float64))
        self._sorted = True
        self._built_k = 0
        self._built_cutoff = -math.inf
        self._cache = None

    def prefix(self, cutoff_u: float) -> _SeriesData:
        """Resampled series of the prefix u <= cutoff_u (play seconds); never reads beyond."""
        if self._guard is not None:
            self._guard.check_read(self._player_id, cutoff_u)
        if not self._sorted:
            self._sort()
        u_all = self._u.view()
        k = int(np.searchsorted(u_all, cutoff_u, side="right"))
        cache = self._cache
        if cache is None or cutoff_u < self._built_cutoff or k < self._built_k:
            u, x, y, v, d, i = self._columns(k)
            cache = _build_series(u, x, y, v, d, i, self._cfg)
        elif k > self._built_k:
            u, x, y, v, d, i = self._columns(k)
            ext = _extend_series(cache, u, x, y, v, d, i, self._all_pos, self._all_spd, self._cfg)
            cache = ext if ext is not None else _build_series(u, x, y, v, d, i, self._cfg)
        self._cache = cache
        self._built_k = k
        self._built_cutoff = cutoff_u
        return cache


# --------------------------------------------------------------------------------------------
# Snapshot computation (shared by the offline reference and the online engine)
# --------------------------------------------------------------------------------------------


def _snapshot_from_series(
    *,
    match_id: str,
    player_id: str,
    period: int,
    as_of_u: float,
    cutoff_u: float,
    series: _SeriesData,
    disrupt_u: FloatArray,
    vmax_mps: float | None,
    cfg: Config,
) -> FeatureSnapshot:
    """FeatureSnapshot from a resampled prefix and the known disruption event times.

    All times are play-clock seconds; windows end at ``cutoff_u = as_of_u - delta``. Guards the
    blanking rule: raises RuntimeError if the series or the used events reach beyond the cutoff.
    """
    g = series.grid
    n = int(g.size)
    cut = float(cutoff_u)
    if n and float(g[-1]) > cut + _EPS_S:
        raise RuntimeError(
            f"feature computation for {player_id} read samples beyond the cutoff "
            f"({float(g[-1])} > {cut})"
        )
    d_hi = int(np.searchsorted(disrupt_u, cut, side="right"))
    d_lo = int(np.searchsorted(disrupt_u, cut - _W5_S, side="right"))
    if d_hi and float(disrupt_u[d_hi - 1]) > cut + _EPS_S:
        raise RuntimeError(f"feature computation for {player_id} read events beyond the cutoff")

    f = cfg.features
    q = cfg.quality

    def valid_count(lo: int) -> float:
        return float(series.cum_valid[n] - series.cum_valid[lo])

    def coverage(window_s: float) -> float:
        if n == 0 or not math.isfinite(series.first_u) or not math.isfinite(series.dt):
            return 0.0
        eff = min(window_s, cut - series.first_u)
        if eff <= 0.0:
            return 0.0
        lo = int(np.searchsorted(g, cut - window_s, side="right"))
        return float(min(1.0, valid_count(lo) * series.dt / eff))

    features: dict[str, float | None] = {name: None for name in FEATURE_NAMES}
    coverage_d: dict[str, float] = {
        "hsr_5m": coverage(_W5_S),
        "spr_5m": coverage(_W5_S),
        "recovery_10m": coverage(_W10_S),
        "hir_drop_15m": coverage(_W15_S),
        "nearmax_cum": coverage(math.inf),
        "disrupt_5m": 1.0,
    }

    if n:
        lo5 = int(np.searchsorted(g, cut - _W5_S, side="right"))
        if valid_count(lo5) > 0.0:
            features["hsr_5m"] = float(series.cum_hsr[n] - series.cum_hsr[lo5])
            features["spr_5m"] = float(series.cum_spr[n] - series.cum_spr[lo5])

    speed_ok = np.where(series.valid, series.speed, np.nan)

    if n and math.isfinite(series.dt):
        ctx = f.effort_merge_gap_s + 2.0 * series.dt
        lo10 = int(np.searchsorted(g, cut - _W10_S - ctx, side="left"))
        starts, ends, _, _ = _effort_arrays(
            g[lo10:],
            speed_ok[lo10:],
            f.hsr_threshold_mps,
            f.effort_min_duration_s,
            f.effort_merge_gap_s,
            series.dt,
        )
        keep = starts > (cut - _W10_S)
        st, en = starts[keep], ends[keep]
        if st.size >= 2:
            features["recovery_10m"] = float(np.median(st[1:] - en[:-1]))

    if n:
        lo15 = int(np.searchsorted(g, cut - _W15_S, side="right"))
        hi_den = int(np.searchsorted(g, series.first_u + _W15_S, side="right"))
        den = float(series.cum_hsr[hi_den])
        if den > 0.0 and valid_count(lo15) > 0.0:
            # A zero reference distance (no HSR in his first 15 minutes) makes the ratio
            # undefined; the feature stays missing and falls back to the usual state later.
            num = float(series.cum_hsr[n] - series.cum_hsr[lo15])
            features["hir_drop_15m"] = num / den

    if n and vmax_mps is not None and math.isfinite(series.dt) and valid_count(0) > 0.0:
        thr = f.nearmax_fraction * vmax_mps
        starts, _, _, _ = _effort_arrays(
            g, speed_ok, thr, f.effort_min_duration_s, f.effort_merge_gap_s, series.dt
        )
        features["nearmax_cum"] = float(starts.size)

    features["disrupt_5m"] = float(d_hi - d_lo)

    identity = 1.0
    if n:
        lo_id = int(np.searchsorted(g, cut - _W15_S, side="right"))
        if valid_count(lo_id) > 0.0:
            identity = float(np.min(np.where(series.valid[lo_id:], series.ident[lo_id:], np.inf)))

    low_cov = any(coverage_d[name] < q.min_coverage for name in FEATURE_NAMES)
    ident_low = identity < q.min_identity
    reasons = tuple(
        code for code, bad in (("low_coverage", low_cov), ("identity_low", ident_low)) if bad
    )
    if all(coverage_d[name] < q.min_coverage for name in TRACKING_FEATURES):
        quality = Quality.PREMATCH_ONLY
    elif reasons:
        quality = Quality.DEGRADED
    else:
        quality = Quality.GOOD
    return FeatureSnapshot(
        match_id=match_id,
        player_id=player_id,
        period=period,
        as_of_t_s=float(as_of_u),
        cutoff_t_s=cut,
        features=features,
        z={},
        coverage=coverage_d,
        identity_confidence=identity,
        quality=quality,
        quality_reasons=reasons,
    )


# --------------------------------------------------------------------------------------------
# Offline reference
# --------------------------------------------------------------------------------------------


def _num_col(df: pd.DataFrame, name: str) -> FloatArray:
    """Column as float64; missing values become NaN."""
    return np.asarray(
        pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan),
        dtype=np.float64,
    )


def _clocks_and_disrupt_upto(
    events: pd.DataFrame, as_of_u: float
) -> tuple[_PeriodClock, _PeriodClock, FloatArray]:
    """Period clocks and sorted disruption-event play times from the events up to ``as_of_u``.

    Returns the clock known at ``as_of_u`` (only the period ends observed so far), the clock from
    every period_end in the frame, and the play times of the available disruption events.
    Availability is judged on the full clock: it orders events in real time across period breaks,
    where the match clock is ambiguous. Every available event maps identically under both clocks
    (a period starts only after every previous period_end), so the full clock adds no future
    information to the features; it only excludes what has not happened yet at ``as_of_u``.
    """
    known = _PeriodClock()
    full = _PeriodClock()
    disrupt: list[float] = []
    if events.shape[0]:
        period = np.asarray(events["period"].to_numpy(), dtype=np.intp)
        t_s = _num_col(events, "t_s")
        types = events["type"].to_numpy()
        for i in range(events.shape[0]):
            full.observe(EventType(str(types[i])), int(period[i]), float(t_s[i]))
        starts = np.asarray([full.start_u(p) for p in (1, 2, 3, 4)])
        origin = np.asarray([PERIOD_START_S[p] for p in (1, 2, 3, 4)])
        u = starts[period - 1] + (t_s - origin[period - 1])
        for i in np.argsort(u, kind="stable"):
            if float(u[i]) > as_of_u:
                break
            et = EventType(str(types[i]))
            known.observe(et, int(period[i]), float(t_s[i]))
            if et in _DISRUPT_TYPES:
                disrupt.append(float(u[i]))
    return known, full, np.asarray(disrupt, dtype=np.float64)


def _frame_series(
    sub: pd.DataFrame, clock: _PeriodClock, cutoff_u: float, cfg: Config
) -> _SeriesData:
    """Resampled series of one player's rows with play time <= cutoff_u (no look-ahead).

    ``clock`` is the full clock of the frame: it orders samples in real time across period
    breaks, so rows not yet played at the cutoff stay out (see ``_clocks_and_disrupt_upto``).
    """
    if sub.shape[0] == 0:
        return _empty_series()
    period = np.asarray(sub["period"].to_numpy(), dtype=np.intp)
    t_s = _num_col(sub, "t_s")
    starts = np.asarray([clock.start_u(p) for p in (1, 2, 3, 4)])
    origin = np.asarray([PERIOD_START_S[p] for p in (1, 2, 3, 4)])
    u = starts[period - 1] + (t_s - origin[period - 1])
    keep = u <= cutoff_u
    order = np.argsort(u[keep], kind="stable")
    u_s = np.asarray(u[keep][order], dtype=np.float64)
    detected = np.asarray(sub["detected"].to_numpy(), dtype=np.bool_)[keep][order]
    return _build_series(
        u_s,
        _num_col(sub, "x_m")[keep][order],
        _num_col(sub, "y_m")[keep][order],
        _num_col(sub, "speed_mps")[keep][order],
        np.asarray(detected, dtype=np.bool_),
        _num_col(sub, "identity_confidence")[keep][order],
        cfg,
    )


def compute_features_offline(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    player_id: str,
    as_of_t_s: float,
    vmax_mps: float | None,
    cfg: Config,
) -> FeatureSnapshot:
    """Batch reference: raw features F1-F6 (docs/02_MATH.md §10) for one player at ``as_of_t_s``.

    ``as_of_t_s`` is on the play clock (module docstring); windows end at
    ``as_of_t_s - cfg.engine.blanking_delta_s`` and no data beyond that cutoff is read.
    ``vmax_mps`` is the player's peak speed from past matches (None: nearmax_cum stays missing).
    ``z`` is left empty here; the risk engine (``engine.risk_engine``) fills it and the
    norm-related quality reasons when it consumes the snapshot.
    """
    as_of_u = float(as_of_t_s)
    known, full, disrupt = _clocks_and_disrupt_upto(events, as_of_u)
    cut = as_of_u - float(cfg.engine.blanking_delta_s)
    sub = samples.loc[samples["player_id"] == player_id]
    if sub.shape[0]:
        match_id = str(sub["match_id"].iloc[0])
    elif events.shape[0]:
        match_id = str(events["match_id"].iloc[0])
    else:
        raise ValueError("cannot determine match_id: no samples and no events")
    series = _frame_series(sub, full, cut, cfg)
    return _snapshot_from_series(
        match_id=match_id,
        player_id=player_id,
        period=known.period_at_u(as_of_u),
        as_of_u=as_of_u,
        cutoff_u=cut,
        series=series,
        disrupt_u=disrupt,
        vmax_mps=vmax_mps,
        cfg=cfg,
    )


# --------------------------------------------------------------------------------------------
# Online engine
# --------------------------------------------------------------------------------------------


class StreamFeatureEngine:
    """Online engine with per-player sample buffers; ``tick`` every ``features.feature_tick_s``.

    Samples and events are pushed in stream order; ``tick(as_of_t_s)`` (play-clock seconds) emits
    one FeatureSnapshot per player currently on the pitch, tracked from samples plus substitution
    and red-card events. Output is identical to ``compute_features_offline`` at the same as_of.
    A ``read_guard`` (the replay's look-ahead guard) checks every read of the sample buffers.
    """

    def __init__(
        self,
        cfg: Config,
        match_id: str,
        vmax_by_player: dict[str, float],
        read_guard: ReadGuard | None = None,
    ) -> None:
        self._cfg = cfg
        self._guard = read_guard
        self._match_id = match_id
        self._vmax = dict(vmax_by_player)
        self._delta = float(cfg.engine.blanking_delta_s)
        self._clock = _PeriodClock()
        self._buffers: dict[str, _SampleBuffer] = {}
        self._on_pitch: set[str] = set()
        self._disrupt_u: list[float] = []

    def push_samples(self, samples: Iterable[TrackingSample]) -> None:
        """Buffer tracking samples of this match; a first sample puts the player on the pitch."""
        for s in samples:
            if s.match_id != self._match_id:
                raise ValueError(
                    f"sample for match {s.match_id!r} pushed to engine for {self._match_id!r}"
                )
            u = self._clock.to_u(s.period, s.t_s)
            buf = self._buffers.get(s.player_id)
            if buf is None:
                buf = _SampleBuffer(self._cfg, s.player_id, self._guard)
                self._buffers[s.player_id] = buf
            buf.add(
                u,
                s.x_m if s.x_m is not None else math.nan,
                s.y_m if s.y_m is not None else math.nan,
                s.speed_mps if s.speed_mps is not None else math.nan,
                1.0 if s.detected else 0.0,
                s.identity_confidence,
            )
            self._on_pitch.add(s.player_id)

    def push_event(self, event: MatchEvent) -> None:
        """Apply one event: period lengths, disruption count, substitutions and red cards."""
        if event.match_id != self._match_id:
            raise ValueError(
                f"event for match {event.match_id!r} pushed to engine for {self._match_id!r}"
            )
        u = self._clock.to_u(event.period, event.t_s)
        self._clock.observe(event.type, event.period, event.t_s)
        if event.type in _DISRUPT_TYPES:
            bisect.insort(self._disrupt_u, u)
        if (
            event.type == EventType.SUBSTITUTION
            and event.player_id is not None
            and event.player_in_id is not None
        ):
            self._on_pitch.discard(event.player_id)
            self._on_pitch.add(event.player_in_id)
        elif event.type == EventType.RED_CARD and event.player_id is not None:
            self._on_pitch.discard(event.player_id)

    def tick(self, as_of_t_s: float) -> list[FeatureSnapshot]:
        """One snapshot per player currently on the pitch (play-clock ``as_of_t_s``, seconds)."""
        as_of_u = float(as_of_t_s)
        cut = as_of_u - self._delta
        period = self._clock.period_at_u(as_of_u)
        disrupt = np.asarray(self._disrupt_u, dtype=np.float64)
        out: list[FeatureSnapshot] = []
        for pid in sorted(self._on_pitch):
            buf = self._buffers.get(pid)
            if buf is None:
                buf = self._buffers[pid] = _SampleBuffer(self._cfg, pid, self._guard)
            out.append(
                _snapshot_from_series(
                    match_id=self._match_id,
                    player_id=pid,
                    period=period,
                    as_of_u=as_of_u,
                    cutoff_u=cut,
                    series=buf.prefix(cut),
                    disrupt_u=disrupt,
                    vmax_mps=self._vmax.get(pid),
                    cfg=self._cfg,
                )
            )
        return out
