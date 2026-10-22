"""Synthetic data with known ground truth (docs/05_VALIDATION.md §1).

Two paths:
- ``generate_person_minutes``: fast path for model tests (no tracking, features simulated directly).
- ``generate_match``: full path with tracking at ``synth.tick_hz``, events, lineups, pre-match
  risks and labels.

Both are deterministic given ``seed`` and draw injuries from the true model of docs/02_MATH.md §4.
In both, the live state behind the hazard is a latent AR(1) process per player-match (SD units);
the full path does not derive it from its own tracking, so its features carry no signal until a
later milestone links the two.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy import signal

from livrisk import markov, riskmath
from livrisk.config import Config, load_config
from livrisk.contracts import (
    EventSource,
    EventType,
    InjuryLabel,
    LabelSource,
    Lineup,
    LineupPlayer,
    MatchEvent,
    PositionGroup,
    PreMatchRisk,
    Tissue,
    TrackingSample,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

DEFAULT_FEATURES: tuple[str, ...] = (
    "hsr_5m",
    "spr_5m",
    "recovery_10m",
    "hir_drop_15m",
    "nearmax_cum",
    "disrupt_5m",
)

#: Regular minutes per player-match in the fast path (no stoppage modelled there).
_FAST_MINUTES: int = 90
#: Burn-in added before the recorded AR(1) features so they start stationary.
_AR1_BURN: int = 200
#: Player-matches generated per vectorised block in the fast path.
_CHUNK: int = 1000
#: Marker for "no event in this player-match".
_NEVER: int = 10**9

# Match scaffolding of the full path. These shape the synthetic match only; the hazard truth is
# in ``SynthTruth``.
_GOALS_PER_MATCH: float = 2.6
_YELLOWS_PER_MATCH: float = 3.5
_REDS_PER_TEAM_MATCH: float = 0.08
_MAX_SUBS_PER_TEAM: int = 5
#: Whole minutes of stoppage, inclusive ranges; the announcement equals the time played.
_STOPPAGE_1H_MIN: tuple[int, int] = (1, 4)
_STOPPAGE_2H_MIN: tuple[int, int] = (2, 6)
_INJURY_STOPPAGE_DELAY_S: tuple[float, float] = (5.0, 30.0)
_INJURY_EXIT_DELAY_S: tuple[float, float] = (60.0, 180.0)
#: An injured player slows to this speed (within the deceleration limit) until he leaves.
_INJURED_SPEED_MPS: float = 0.5
#: Heading random walk; per-tick SD is this times sqrt(dt) so paths do not depend on the rate.
_HEADING_SD_RAD_PER_SQRT_S: float = 0.38
_PITCH_L_M: float = 105.0
_PITCH_W_M: float = 68.0

#: Squad per team: 11 starters (index 0 is the goalkeeper), then 9 substitutes.
_SQUAD: tuple[PositionGroup, ...] = (
    PositionGroup.GK,
    PositionGroup.CB,
    PositionGroup.CB,
    PositionGroup.FB,
    PositionGroup.FB,
    PositionGroup.MF,
    PositionGroup.MF,
    PositionGroup.MF,
    PositionGroup.W,
    PositionGroup.W,
    PositionGroup.FW,
    PositionGroup.GK,
    PositionGroup.CB,
    PositionGroup.FB,
    PositionGroup.MF,
    PositionGroup.MF,
    PositionGroup.W,
    PositionGroup.W,
    PositionGroup.FW,
    PositionGroup.FW,
)
_N_STARTERS: int = 11
_TEAM_IDS: tuple[str, str] = ("T1", "T2")


@dataclass(frozen=True)
class SynthTruth:
    """Ground truth used to generate data. Units: theta and beta on the log-hazard scale."""

    feature_names: tuple[str, ...] = DEFAULT_FEATURES
    beta: tuple[float, ...] = (0.30, 0.0, -0.20, 0.0, 0.10, 0.15)
    #: Band log-levels (8 bands, docs/02_MATH.md §2) before normalisation.
    theta: tuple[float, ...] = (0.10, -0.15, 0.20, 0.30, -0.10, 0.0, 0.15, 0.35)
    match_share: float = 0.6
    #: p7 ~ Beta(mean * c, (1 - mean) * c).
    p7_mean: float = 0.025
    p7_concentration: float = 40.0
    #: Share of match injuries that are contact injuries. Contact injuries are a competing cause
    #: with hazard contact_share / (1 - contact_share) times the non-contact baseline (no live
    #: multiplier); they end the player's risk set without counting as events (y = 0).
    contact_share: float = 0.4
    #: AR(1) coefficient of minute-level live features within a player-match.
    feature_ar1: float = 0.8
    #: Share of starters substituted inside ``sub_window_min`` when the live state stays usual.
    sub_share: float = 0.6
    #: Minute boundaries [start, end) of the second half at which tactical substitutions happen.
    sub_window_min: tuple[int, int] = (55, 85)
    #: Live feature that drives substitution and its log hazard ratio per SD. Positive values
    #: make substitution more likely under high load (informative censoring on purpose).
    sub_feature: str = "hsr_5m"
    sub_log_hr: float = 0.5

    def __post_init__(self) -> None:
        if len(self.beta) != len(self.feature_names):
            raise ValueError(f"beta must have one entry per feature ({len(self.feature_names)})")
        if len(self.theta) != riskmath.N_BANDS:
            raise ValueError(f"theta must have {riskmath.N_BANDS} entries")
        if not 0.0 < self.match_share <= 1.0:
            raise ValueError("match_share must be in (0, 1]")
        if not 0.0 < self.p7_mean < 1.0 or self.p7_concentration <= 0.0:
            raise ValueError("p7_mean must be in (0, 1) and p7_concentration > 0")
        if not 0.0 <= self.contact_share < 1.0:
            raise ValueError("contact_share must be in [0, 1)")
        if not -1.0 < self.feature_ar1 < 1.0:
            raise ValueError("feature_ar1 must be in (-1, 1)")
        if not 0.0 <= self.sub_share < 1.0:
            raise ValueError("sub_share must be in [0, 1)")
        lo, hi = self.sub_window_min
        if not 45 < lo < hi <= 90:
            raise ValueError("sub_window_min must satisfy 45 < start < end <= 90")
        if self.sub_feature not in self.feature_names:
            raise ValueError(f"sub_feature {self.sub_feature!r} is not a feature name")

    def g_levels(self) -> FloatArray:
        """True time profile g per band (mean 1 over the regular bands)."""
        return riskmath.normalise_time_profile(self.theta)

    def mean_relative_risk(self) -> float:
        """E[exp(beta . z)] for z ~ N(0, I), the stationary law of the latent live state."""
        return math.exp(0.5 * sum(b * b for b in self.beta))


@dataclass(frozen=True)
class SpeedProcess:
    """Band-dependent semi-Markov speed process (docs/05_VALIDATION.md §1, config ``synth``).

    ``jump_matrix`` has a zero diagonal; ``band_intensity`` scales entries into run, HSR and sprint
    from lower zones; ``position_intensity`` scales entries into HSR and sprint. Rows are
    renormalised after scaling. Every scaled matrix must be ergodic (``markov.is_ergodic``).
    """

    zones_mps: tuple[float, ...]
    jump_matrix: tuple[tuple[float, ...], ...]
    dwell_mean_s: tuple[float, ...]
    dwell_min_s: tuple[float, ...]
    dwell_gamma_shape: tuple[float, ...]
    band_intensity: tuple[float, ...]
    position_intensity: dict[str, float]
    accel_max_mps2: float
    decel_max_mps2: float
    burn_in_s: float


def speed_process_from_config(cfg: Config) -> SpeedProcess:
    """Read the ``synth`` section; raise ``ValueError`` if any band or position matrix is not
    ergodic."""
    s = cfg.synth
    proc = SpeedProcess(
        zones_mps=tuple(float(z) for z in s.zones_mps),
        jump_matrix=tuple(tuple(float(p) for p in row) for row in s.jump_matrix),
        dwell_mean_s=tuple(float(d) for d in s.dwell_mean_s),
        dwell_min_s=tuple(float(d) for d in s.dwell_min_s),
        dwell_gamma_shape=tuple(float(k) for k in s.dwell_gamma_shape),
        band_intensity=tuple(float(b) for b in s.band_intensity),
        position_intensity={k: float(v) for k, v in s.position_intensity.items()},
        accel_max_mps2=float(s.accel_max_mps2),
        decel_max_mps2=float(s.decel_max_mps2),
        burn_in_s=float(s.burn_in_s),
    )
    if len(proc.band_intensity) != riskmath.N_BANDS:
        raise ValueError(
            f"synth.band_intensity must have {riskmath.N_BANDS} entries, "
            f"got {len(proc.band_intensity)}"
        )
    if any(b <= 0.0 for b in proc.band_intensity) or any(
        v <= 0.0 for v in proc.position_intensity.values()
    ):
        raise ValueError("synth band and position intensities must be > 0")
    if any(m <= f for m, f in zip(proc.dwell_mean_s, proc.dwell_min_s, strict=True)):
        raise ValueError("synth.dwell_mean_s must exceed dwell_min_s in every zone")
    for band in range(len(proc.band_intensity)):
        for pos in proc.position_intensity:
            if not markov.is_ergodic(jump_matrix_for(proc, band, pos)):
                raise ValueError(f"jump matrix for band {band}, position {pos} is not ergodic")
    return proc


def jump_matrix_for(process: SpeedProcess, band: int, position: str) -> FloatArray:
    """Scaled, renormalised jump matrix for one band and position group.

    Band intensity scales entries into run/HSR/sprint (zones 3-5); position intensity scales
    entries into HSR/sprint (zones 4-5). Unknown positions get intensity 1.
    """
    if not 0 <= band < len(process.band_intensity):
        raise ValueError(f"band must be in [0, {len(process.band_intensity)}), got {band}")
    jump = np.asarray(process.jump_matrix, dtype=np.float64)
    scale = np.ones(jump.shape[1])
    scale[3:] = process.band_intensity[band]
    scale[4:] *= process.position_intensity.get(position, 1.0)
    scaled = jump * scale[None, :]
    return np.asarray(scaled / scaled.sum(axis=1, keepdims=True), dtype=np.float64)


def expected_occupancy(process: SpeedProcess, band: int, position: str) -> FloatArray:
    """Long-run share of time per zone: ``markov.semi_markov_occupancy`` of the band matrix."""
    return markov.semi_markov_occupancy(
        jump_matrix_for(process, band, position), np.asarray(process.dwell_mean_s)
    )


def _zone_targets(process: SpeedProcess, vmax_mps: float) -> FloatArray:
    """Target speed (m/s) per zone: mid-zone for bounded zones, ``vmax_mps`` for the top zone."""
    edges = process.zones_mps
    targets = np.empty(len(edges))
    targets[0] = 0.0
    for z in range(1, len(edges) - 1):
        targets[z] = 0.5 * (edges[z] + edges[z + 1])
    targets[len(edges) - 1] = vmax_mps
    return targets


def _ramp(
    v0: FloatArray, target: FloatArray, steps: FloatArray, accel_step: float, decel_step: float
) -> FloatArray:
    """Speed after ``steps`` ticks moving from ``v0`` toward ``target`` at the per-tick limits."""
    return np.asarray(
        v0 + np.clip(target - v0, -decel_step * steps, accel_step * steps), dtype=np.float64
    )


def simulate_speed(
    process: SpeedProcess,
    position: str,
    vmax_mps: float,
    duration_s: float,
    band_of_t: Callable[[float], int],
    rng: np.random.Generator,
    hz: float = 10.0,
) -> tuple[FloatArray, IntArray]:
    """Speed (m/s) and target zone per tick after ``burn_in_s``. Speed follows the zone target
    within the acceleration limits; the zone sequence follows the jump matrix of the current
    band; bout lengths are ``dwell_min + Gamma(shape, (mean - min) / shape)``, rounded to ticks.

    ``band_of_t`` maps seconds since the start of the simulated segment to a band index; it is
    called at every zone change and receives 0 during burn-in. The initial zone is drawn from the
    occupancy law of the first band, and ``burn_in_s`` of output is discarded.
    """
    if duration_s <= 0.0 or hz <= 0.0 or vmax_mps <= 0.0:
        raise ValueError("duration_s, hz and vmax_mps must be > 0")
    n_out = round(duration_s * hz)
    n_burn = round(process.burn_in_s * hz)
    total = n_out + n_burn
    targets = _zone_targets(process, vmax_mps)
    target_list = targets.tolist()
    cdfs: list[list[list[float]]] = []
    for band in range(len(process.band_intensity)):
        cdf = np.cumsum(jump_matrix_for(process, band, position), axis=1)
        cdf[:, -1] = 1.0
        cdfs.append(cdf.tolist())
    floors = process.dwell_min_s
    shapes = process.dwell_gamma_shape
    scales = [(m - f) / k for m, f, k in zip(process.dwell_mean_s, floors, shapes, strict=True)]
    accel_step = process.accel_max_mps2 / hz
    decel_step = process.decel_max_mps2 / hz

    zone = int(rng.choice(len(targets), p=expected_occupancy(process, band_of_t(0.0), position)))
    speed = target_list[zone]
    bout_zone: list[int] = []
    bout_len: list[int] = []
    bout_v0: list[float] = []
    t = 0
    while t < total:
        dwell_s = floors[zone] + float(rng.gamma(shapes[zone], scales[zone]))
        n_bout = min(max(1, round(dwell_s * hz)), total - t)
        bout_zone.append(zone)
        bout_len.append(n_bout)
        bout_v0.append(speed)
        gap = target_list[zone] - speed
        speed += max(-decel_step * n_bout, min(accel_step * n_bout, gap))
        t += n_bout
        if t < total:
            band = band_of_t(max((t - n_burn) / hz, 0.0))
            zone = bisect.bisect_right(cdfs[band][zone], float(rng.random()))

    lengths = np.asarray(bout_len, dtype=np.int64)
    zones = np.repeat(np.asarray(bout_zone, dtype=np.int64), lengths)
    v0 = np.repeat(np.asarray(bout_v0, dtype=np.float64), lengths)
    steps = (np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths) + 1).astype(
        np.float64
    )
    speeds = _ramp(v0, targets[zones], steps, accel_step, decel_step)
    return speeds[n_burn:], zones[n_burn:]


@dataclass(frozen=True)
class SynthMatch:
    """One synthetic match (full path).

    ``samples`` has the ``TrackingSample`` columns, sorted by time then player. ``person_minutes``
    is the ground truth behind the labels: one row per outfield player-minute at risk, with the
    latent live state (SD units) and the same columns as the fast path except club and season.
    ``vmax_mps`` is each squad player's true peak speed.
    """

    match_id: str
    lineups: tuple[Lineup, Lineup]
    events: tuple[MatchEvent, ...]
    samples: pd.DataFrame
    prematch: tuple[PreMatchRisk, ...]
    injuries: tuple[InjuryLabel, ...]
    person_minutes: pd.DataFrame
    vmax_mps: dict[str, float]
    truth: SynthTruth = field(default_factory=SynthTruth)


# --------------------------------------------------------------------------------------------
# True hazard, shared by both paths
# --------------------------------------------------------------------------------------------


def _ar1_features(rng: np.random.Generator, n: int, minutes: int, k: int, phi: float) -> FloatArray:
    """Stationary AR(1) z-scores, shape (n, minutes, k): sd 1, coefficient ``phi``."""
    innov = float(np.sqrt(1.0 - phi * phi))
    eps = rng.normal(0.0, 1.0, size=(n, minutes + _AR1_BURN, k))
    z = np.asarray(signal.lfilter([innov], [1.0, -phi], eps, axis=1), dtype=np.float64)
    return z[:, _AR1_BURN:, :]


def _draw_p7(rng: np.random.Generator, truth: SynthTruth, n: int) -> FloatArray:
    a = truth.p7_mean * truth.p7_concentration
    b = (1.0 - truth.p7_mean) * truth.p7_concentration
    return np.clip(rng.beta(a, b, size=n), 1e-6, 0.999)


def _log_baseline(truth: SynthTruth, p7: FloatArray) -> FloatArray:
    """ln(Lambda / 90) per player: log baseline hazard per minute at g = 1 (docs/02 §1-§2)."""
    return np.log(
        np.array(
            [
                riskmath.baseline_hazard_per_min(
                    riskmath.weekly_to_match_cumhaz(float(p), truth.match_share)
                )
                for p in p7
            ],
            dtype=np.float64,
        )
    )


def _contact_prob(truth: SynthTruth, log_base: FloatArray) -> FloatArray:
    """Per-minute probability of a contact injury, given the log baseline including g."""
    if truth.contact_share == 0.0:
        return np.zeros_like(log_base)
    ratio = truth.contact_share / (1.0 - truth.contact_share)
    return riskmath.inv_cloglog(log_base + math.log(ratio))


def _sub_prob(truth: SynthTruth, log_hr: float, z_sub: FloatArray) -> FloatArray:
    """Per-minute probability of a tactical substitution inside the window.

    The base hazard gives ``sub_share`` over the window at z = 0 and is lowered by
    exp(-log_hr^2 / 2) so the mean hazard over the stationary live state stays the same.
    """
    if truth.sub_share == 0.0:
        return np.zeros_like(z_sub)
    width = truth.sub_window_min[1] - truth.sub_window_min[0]
    log_h = math.log(riskmath.cumhaz_from_prob(truth.sub_share) / width) - 0.5 * log_hr * log_hr
    return riskmath.inv_cloglog(log_h + log_hr * z_sub)


def _categorical(codes: IntArray, labels: list[str]) -> pd.Categorical[str]:
    return pd.Categorical.from_codes(codes, categories=pd.Index(labels))


def _first_hit(hits: BoolArray) -> IntArray:
    """Index of the first True per row, or ``_NEVER``."""
    return np.where(hits.any(axis=1), hits.argmax(axis=1), _NEVER).astype(np.int64)


# --------------------------------------------------------------------------------------------
# Fast path
# --------------------------------------------------------------------------------------------


def generate_person_minutes(
    n_player_matches: int,
    seed: int,
    truth: SynthTruth | None = None,
    informative_censoring: bool = True,
    n_clubs: int = 10,
    n_seasons: int = 3,
) -> pd.DataFrame:
    """Fast path. One row per player-minute at risk, 90 regular minutes per player-match.

    Columns: match_id, player_id, club_id, season, period, t_min (minute start), band, p7,
    one column per feature name (z values, SD units), y (non-contact injury onset in this minute),
    contact (contact injury in this minute; ends the risk set), substituted_after (bool).

    Within a minute a non-contact injury takes precedence over a contact injury, and both over a
    substitution, which happens at the end of the minute. With ``informative_censoring`` the
    substitution hazard rises with the current value of ``truth.sub_feature``; without it the
    hazard is flat in the window. There is no data after a row's minute, so no look-ahead applies.
    """
    truth = truth or SynthTruth()
    if n_player_matches < 1 or n_clubs < 1 or n_seasons < 1:
        raise ValueError("n_player_matches, n_clubs and n_seasons must be >= 1")
    rng = np.random.default_rng(seed)
    beta = np.asarray(truth.beta, dtype=np.float64)
    k = len(truth.feature_names)
    minutes = np.arange(_FAST_MINUTES)
    period_of_minute = np.where(minutes < 45, 1, 2).astype(np.int64)
    band_of_minute = np.array(
        [
            riskmath.band_index(float(m), int(p))
            for m, p in zip(minutes, period_of_minute, strict=True)
        ],
        dtype=np.int64,
    )
    log_g = np.log(truth.g_levels())[band_of_minute]
    boundary = minutes + 1
    sub_open = (boundary >= truth.sub_window_min[0]) & (boundary < truth.sub_window_min[1])
    sub_log_hr = truth.sub_log_hr if informative_censoring else 0.0
    sub_k = truth.feature_names.index(truth.sub_feature)

    owners: list[IntArray] = []
    t_rows: list[IntArray] = []
    p7_rows: list[FloatArray] = []
    z_rows: list[FloatArray] = []
    y_rows: list[BoolArray] = []
    c_rows: list[BoolArray] = []
    s_rows: list[BoolArray] = []
    for start in range(0, n_player_matches, _CHUNK):
        n = min(_CHUNK, n_player_matches - start)
        p7 = _draw_p7(rng, truth, n)
        z = _ar1_features(rng, n, _FAST_MINUTES, k, truth.feature_ar1)
        log_base = _log_baseline(truth, p7)[:, None] + log_g[None, :]
        p_nc = riskmath.inv_cloglog(log_base + z @ beta)
        p_c = _contact_prob(truth, log_base)
        p_sub = np.where(sub_open[None, :], _sub_prob(truth, sub_log_hr, z[:, :, sub_k]), 0.0)
        u = rng.random((3, n, _FAST_MINUTES))
        first_nc = _first_hit(u[0] < p_nc)
        first_c = _first_hit(u[1] < p_c)
        first_sub = _first_hit(u[2] < p_sub)
        first_injury = np.minimum(first_nc, first_c)
        last = np.minimum(first_injury, np.minimum(first_sub, _FAST_MINUTES - 1))
        counts = last + 1
        rep = np.repeat(np.arange(n), counts)
        t_row = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        owners.append(start + rep)
        t_rows.append(t_row)
        p7_rows.append(p7[rep])
        z_rows.append(z[rep, t_row, :])
        y_rows.append(t_row == first_nc[rep])
        c_rows.append((t_row == first_c[rep]) & (first_c[rep] < first_nc[rep]))
        s_rows.append((t_row == first_sub[rep]) & (first_sub[rep] < first_injury[rep]))

    owner = np.concatenate(owners)
    t_all = np.concatenate(t_rows)
    z_all = np.concatenate(z_rows)
    data: dict[str, object] = {
        "match_id": _categorical(owner, [f"sm{i:06d}" for i in range(n_player_matches)]),
        "player_id": _categorical(owner, [f"sp{i:06d}" for i in range(n_player_matches)]),
        "club_id": _categorical(owner % n_clubs, [f"sc{c:02d}" for c in range(n_clubs)]),
        "season": _categorical(
            (owner // n_clubs) % n_seasons, [f"s{1 + s}" for s in range(n_seasons)]
        ),
        "period": period_of_minute[t_all],
        "t_min": t_all.astype(np.float64),
        "band": band_of_minute[t_all],
        "p7": np.concatenate(p7_rows),
    }
    for j, name in enumerate(truth.feature_names):
        data[name] = z_all[:, j]
    data["y"] = np.concatenate(y_rows).astype(np.int64)
    data["contact"] = np.concatenate(c_rows)
    data["substituted_after"] = np.concatenate(s_rows)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------------------------
# Full path
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Clock:
    """Match minutes and ticks. Play time runs without the half-time break: tick 0 is kick-off,
    tick ``p1_ticks`` is the second-half kick-off (t_s = 2700 on the match clock)."""

    hz: float
    ticks_per_min: int
    p1_minutes: int
    p2_minutes: int
    t_min: FloatArray
    period: IntArray
    band: IntArray

    @property
    def n_minutes(self) -> int:
        return self.p1_minutes + self.p2_minutes

    @property
    def p1_ticks(self) -> int:
        return self.p1_minutes * self.ticks_per_min

    @property
    def total_ticks(self) -> int:
        return self.n_minutes * self.ticks_per_min

    def at_play_s(self, u_s: float) -> tuple[int, float]:
        """Play seconds -> (period, t_s) on the continuous match clock."""
        p1_s = self.p1_minutes * 60.0
        if u_s < p1_s:
            return 1, u_s
        return 2, 2700.0 + (u_s - p1_s)

    def at_tick(self, tick: int) -> tuple[int, float]:
        if tick < self.p1_ticks:
            return 1, tick / self.hz
        return 2, 2700.0 + (tick - self.p1_ticks) / self.hz


def _make_clock(hz: float, extra_1h_min: int, extra_2h_min: int) -> _Clock:
    ticks_per_min = round(60.0 * hz)
    if abs(60.0 * hz - ticks_per_min) > 1e-9:
        raise ValueError(f"sample_hz must give a whole number of ticks per minute, got {hz}")
    p1 = 45 + extra_1h_min
    p2 = 45 + extra_2h_min
    t_min = np.concatenate([np.arange(p1), 45 + np.arange(p2)]).astype(np.float64)
    period = np.concatenate([np.ones(p1), np.full(p2, 2)]).astype(np.int64)
    band = np.array(
        [riskmath.band_index(float(t), int(p)) for t, p in zip(t_min, period, strict=True)],
        dtype=np.int64,
    )
    return _Clock(hz, ticks_per_min, p1, p2, t_min, period, band)


class _BandLookup:
    """``band_of_t`` for one spell: seconds since the spell start -> band of that match minute."""

    def __init__(self, clock: _Clock, start_tick: int) -> None:
        self._bands = clock.band.tolist()
        self._offset_s = start_tick / clock.hz

    def __call__(self, t_s: float) -> int:
        minute = int((self._offset_s + t_s) // 60.0)
        return int(self._bands[min(max(minute, 0), len(self._bands) - 1)])


@dataclass
class _Spell:
    """One player's time on the pitch, in ticks [start, end); ``injury_tick`` if injured."""

    roster: int
    start_tick: int
    end_tick: int
    injury_tick: int | None = None


@dataclass
class _TeamPlay:
    spells: list[_Spell]
    rows_roster: list[int]
    rows_minute: list[int]
    rows_y: list[bool]
    rows_contact: list[bool]
    rows_sub: list[bool]
    z: FloatArray
    p7: FloatArray
    labels: list[InjuryLabel]
    events: list[MatchEvent]


def _pick_substitute(group: PositionGroup, used: set[int]) -> int:
    """First unused outfield substitute of the same group, else the first unused one."""
    bench = [b for b in range(_N_STARTERS, len(_SQUAD)) if _SQUAD[b] != PositionGroup.GK]
    free = [b for b in bench if b not in used]
    same = [b for b in free if _SQUAD[b] == group]
    pick = (same or free)[0]
    used.add(pick)
    return pick


def _play_team(
    rng: np.random.Generator,
    truth: SynthTruth,
    clock: _Clock,
    match_id: str,
    team_id: str,
    squad_ids: Sequence[str],
    p7: FloatArray,
) -> _TeamPlay:
    """Draw one team's injuries, substitutions and red cards minute by minute.

    Order inside a minute: non-contact injury, contact injury, then a tactical substitution at the
    end of the minute, then at most one red card for the team. Goalkeepers are never at risk and
    never substituted. Injured players leave 1-3 minutes after onset and are replaced while
    substitutions remain; replacements join the risk set at their first full minute.
    """
    n_squad, n_min = len(_SQUAD), clock.n_minutes
    tpm, total = clock.ticks_per_min, clock.total_ticks
    beta = np.asarray(truth.beta, dtype=np.float64)
    z = _ar1_features(rng, n_squad, n_min, len(beta), truth.feature_ar1)
    log_base = _log_baseline(truth, p7)[:, None] + np.log(truth.g_levels())[clock.band][None, :]
    p_nc = riskmath.inv_cloglog(log_base + z @ beta)
    p_c = _contact_prob(truth, log_base)
    boundary = clock.t_min + 1.0
    window = (
        (clock.period == 2)
        & (boundary >= truth.sub_window_min[0])
        & (boundary < truth.sub_window_min[1])
    )
    sub_k = truth.feature_names.index(truth.sub_feature)
    p_sub = np.where(window[None, :], _sub_prob(truth, truth.sub_log_hr, z[:, :, sub_k]), 0.0)
    u = rng.random((3, n_squad, n_min))
    hit_nc: list[list[bool]] = (u[0] < p_nc).tolist()
    hit_c: list[list[bool]] = (u[1] < p_c).tolist()
    hit_sub: list[list[bool]] = (u[2] < p_sub).tolist()
    p_red = riskmath.prob_from_cumhaz(_REDS_PER_TEAM_MATCH / n_min)
    hit_red: list[bool] = (rng.random(n_min) < p_red).tolist()

    play = _TeamPlay([], [], [], [], [], [], z, p7, [], [])
    spells = {r: _Spell(r, 0, total) for r in range(_N_STARTERS)}
    at_risk = list(range(1, _N_STARTERS))
    joins: dict[int, list[int]] = {}
    used: set[int] = set()
    subs_left = _MAX_SUBS_PER_TEAM
    src = EventSource.SYNTHETIC

    def substitution(tick: int, out_r: int) -> int:
        new = _pick_substitute(_SQUAD[out_r], used)
        spells[new] = _Spell(new, tick, total)
        joins.setdefault(-(-tick // tpm), []).append(new)
        period, t_s = clock.at_tick(tick)
        play.events.append(
            MatchEvent(
                match_id=match_id,
                period=period,
                t_s=t_s,
                type=EventType.SUBSTITUTION,
                team_id=team_id,
                player_id=squad_ids[out_r],
                player_in_id=squad_ids[new],
                source=src,
            )
        )
        return new

    for i in range(n_min):
        at_risk = sorted(at_risk + joins.pop(i, []))
        start = i * tpm
        for r in list(at_risk):
            play.rows_roster.append(r)
            play.rows_minute.append(i)
            injured_nc, injured_c = hit_nc[r][i], hit_c[r][i]
            if injured_nc or injured_c:
                play.rows_y.append(injured_nc)
                play.rows_contact.append(not injured_nc)
                play.rows_sub.append(False)
                at_risk.remove(r)
                onset_s = start / clock.hz + float(rng.uniform(0.0, 60.0))
                onset_tick = min(int(onset_s * clock.hz), start + tpm - 1)
                exit_tick = min(
                    round((onset_s + float(rng.uniform(*_INJURY_EXIT_DELAY_S))) * clock.hz),
                    total,
                )
                stop_tick = min(
                    onset_tick + round(float(rng.uniform(*_INJURY_STOPPAGE_DELAY_S)) * clock.hz),
                    exit_tick - 1,
                )
                spells[r].injury_tick = onset_tick
                spells[r].end_tick = exit_tick
                period, t_onset = clock.at_play_s(onset_s)
                play.labels.append(
                    _make_label(rng, match_id, squad_ids[r], period, t_onset, not injured_nc)
                )
                stop_period, t_stop = clock.at_tick(stop_tick)
                play.events.append(
                    MatchEvent(
                        match_id=match_id,
                        period=stop_period,
                        t_s=t_stop,
                        type=EventType.INJURY_STOPPAGE,
                        team_id=team_id,
                        player_id=squad_ids[r],
                        source=src,
                    )
                )
                if subs_left > 0 and exit_tick < total:
                    subs_left -= 1
                    substitution(exit_tick, r)
                continue
            subbed = r < _N_STARTERS and hit_sub[r][i] and subs_left > 0
            play.rows_y.append(False)
            play.rows_contact.append(False)
            play.rows_sub.append(subbed)
            if subbed:
                subs_left -= 1
                at_risk.remove(r)
                spells[r].end_tick = start + tpm
                substitution(start + tpm, r)
        if hit_red[i] and at_risk:
            r = at_risk[int(rng.integers(len(at_risk)))]
            at_risk.remove(r)
            card_tick = start + 1 + int(rng.integers(tpm - 1))
            spells[r].end_tick = card_tick
            period, t_s = clock.at_tick(card_tick)
            play.events.append(
                MatchEvent(
                    match_id=match_id,
                    period=period,
                    t_s=t_s,
                    type=EventType.RED_CARD,
                    team_id=team_id,
                    player_id=squad_ids[r],
                    source=src,
                )
            )
    play.spells = [spells[r] for r in sorted(spells)]
    return play


def _make_label(
    rng: np.random.Generator,
    match_id: str,
    player_id: str,
    period: int,
    t_s: float,
    contact: bool,
) -> InjuryLabel:
    """One injury label at onset ``t_s``. Non-contact labels are time-loss >= 7 days."""
    if contact:
        tissue = str(
            rng.choice(np.array(["ligament", "bone", "other", "muscle"]), p=[0.4, 0.2, 0.3, 0.1])
        )
        days_lost = 1 + int(rng.gamma(1.5, 10.0))
    else:
        tissue = str(
            rng.choice(
                np.array(["muscle", "ligament", "tendon", "other"]), p=[0.7, 0.15, 0.1, 0.05]
            )
        )
        days_lost = 7 + int(rng.gamma(2.0, 6.0))
    area = {
        "muscle": ("hamstring", "adductor", "quadriceps", "calf"),
        "ligament": ("knee", "ankle"),
        "tendon": ("achilles", "patellar"),
        "bone": ("foot", "tibia"),
        "other": ("other",),
    }[tissue]
    return InjuryLabel(
        match_id=match_id,
        player_id=player_id,
        period=period,
        t_s_onset=t_s,
        uncertainty_s=float(rng.uniform(10.0, 50.0)),
        contact=contact,
        tissue=Tissue(tissue),
        body_area=str(rng.choice(np.array(area))),
        days_lost=days_lost,
        source=LabelSource.SYNTHETIC,
        annotator_agreement=float(rng.uniform(0.7, 1.0)),
    )


def _clock_events(match_id: str, clock: _Clock) -> list[MatchEvent]:
    """Kick-off, stoppage announcement and period end for both halves."""
    src = EventSource.SYNTHETIC
    events: list[MatchEvent] = []
    for period, start_s, extra_min in (
        (1, 0.0, clock.p1_minutes - 45),
        (2, 2700.0, clock.p2_minutes - 45),
    ):
        events.append(
            MatchEvent(
                match_id=match_id, period=period, t_s=start_s, type=EventType.KICKOFF, source=src
            )
        )
        events.append(
            MatchEvent(
                match_id=match_id,
                period=period,
                t_s=start_s + 2690.0,
                type=EventType.STOPPAGE_ANNOUNCED,
                stoppage_min=float(extra_min),
                source=src,
            )
        )
        events.append(
            MatchEvent(
                match_id=match_id,
                period=period,
                t_s=start_s + 2700.0 + 60.0 * extra_min,
                type=EventType.PERIOD_END,
                source=src,
            )
        )
    return events


def _incidents(
    rng: np.random.Generator,
    match_id: str,
    clock: _Clock,
    plays: dict[str, _TeamPlay],
    squad_ids: dict[str, list[str]],
) -> list[MatchEvent]:
    """Goals and yellow cards at random play times; cards go to a player on the pitch."""
    src = EventSource.SYNTHETIC
    lo, hi = clock.ticks_per_min, clock.total_ticks - clock.ticks_per_min
    events: list[MatchEvent] = []
    for _ in range(int(rng.poisson(_GOALS_PER_MATCH))):
        period, t_s = clock.at_tick(int(rng.integers(lo, hi)))
        team_id = _TEAM_IDS[int(rng.integers(2))]
        events.append(
            MatchEvent(
                match_id=match_id,
                period=period,
                t_s=t_s,
                type=EventType.GOAL,
                team_id=team_id,
                source=src,
            )
        )
    for _ in range(int(rng.poisson(_YELLOWS_PER_MATCH))):
        tick = int(rng.integers(lo, hi))
        team_id = _TEAM_IDS[int(rng.integers(2))]
        on = [s.roster for s in plays[team_id].spells if s.start_tick <= tick < s.end_tick]
        period, t_s = clock.at_tick(tick)
        events.append(
            MatchEvent(
                match_id=match_id,
                period=period,
                t_s=t_s,
                type=EventType.YELLOW_CARD,
                team_id=team_id,
                player_id=squad_ids[team_id][on[int(rng.integers(len(on)))]],
                source=src,
            )
        )
    return events


def _fold(values: FloatArray, lo: float, hi: float) -> FloatArray:
    """Reflect a path into [lo, hi] (triangle-wave folding; preserves step lengths)."""
    width = hi - lo
    return np.asarray(lo + np.abs((values - lo + width) % (2.0 * width) - width), dtype=np.float64)


def _spell_track(
    rng: np.random.Generator,
    proc: SpeedProcess,
    clock: _Clock,
    spell: _Spell,
    group: PositionGroup,
    vmax_mps: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Speed (m/s) and position (m) per tick of one spell."""
    n_ticks = spell.end_tick - spell.start_tick
    dt = 1.0 / clock.hz
    speeds, _ = simulate_speed(
        proc,
        group.value,
        vmax_mps,
        n_ticks * dt,
        _BandLookup(clock, spell.start_tick),
        rng,
        hz=clock.hz,
    )
    if spell.injury_tick is not None:
        j0 = max(spell.injury_tick - spell.start_tick, 1)
        steps = np.arange(1, n_ticks - j0 + 1, dtype=np.float64)
        v0 = np.full(steps.shape, speeds[j0 - 1])
        speeds[j0:] = _ramp(
            v0,
            np.full(steps.shape, _INJURED_SPEED_MPS),
            steps,
            proc.accel_max_mps2 * dt,
            proc.decel_max_mps2 * dt,
        )
    heading = float(rng.uniform(-np.pi, np.pi)) + np.cumsum(
        rng.normal(0.0, _HEADING_SD_RAD_PER_SQRT_S * math.sqrt(dt), size=n_ticks)
    )
    x0 = float(rng.uniform(-0.4 * _PITCH_L_M, 0.4 * _PITCH_L_M))
    y0 = float(rng.uniform(-0.4 * _PITCH_W_M, 0.4 * _PITCH_W_M))
    x_m = _fold(x0 + np.cumsum(speeds * np.cos(heading) * dt), -_PITCH_L_M / 2, _PITCH_L_M / 2)
    y_m = _fold(y0 + np.cumsum(speeds * np.sin(heading) * dt), -_PITCH_W_M / 2, _PITCH_W_M / 2)
    return speeds, x_m, y_m


_SAMPLE_COLUMNS: tuple[str, ...] = tuple(TrackingSample.model_fields)


def _samples_frame(
    match_id: str,
    clock: _Clock,
    ticks: IntArray,
    player_ids: NDArray[np.object_],
    team_ids: NDArray[np.object_],
    speeds: FloatArray,
    x_m: FloatArray,
    y_m: FloatArray,
) -> pd.DataFrame:
    n = ticks.shape[0]
    period = np.where(ticks < clock.p1_ticks, 1, 2).astype(np.int64)
    t_s = np.where(period == 1, ticks / clock.hz, 2700.0 + (ticks - clock.p1_ticks) / clock.hz)
    missing = pd.arrays.IntegerArray(np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.bool_))
    frame = pd.DataFrame(
        {
            "match_id": np.full(n, match_id, dtype=object),
            "player_id": player_ids,
            "team_id": team_ids,
            "period": period,
            "t_s": t_s.astype(np.float64),
            "x_m": x_m,
            "y_m": y_m,
            "speed_mps": speeds,
            "source": np.full(n, "synthetic", dtype=object),
            "detected": np.ones(n, dtype=np.bool_),
            "identity_confidence": np.ones(n, dtype=np.float64),
            "wall_time_ns": missing,
            "ingest_time_ns": missing.copy(),
        }
    )
    return frame[list(_SAMPLE_COLUMNS)]


def generate_match(
    seed: int,
    truth: SynthTruth | None = None,
    sample_hz: float | None = None,
    match_id: str | None = None,
    cfg: Config | None = None,
    tracking: bool = True,
) -> SynthMatch:
    """Full path: two squads of 20 (11 starters), tracking at ``sample_hz`` (default
    ``synth.tick_hz``) from the ``SpeedProcess`` of ``cfg`` (default: ``load_config()``),
    positions from a heading random walk reflected at the pitch edges (metres, centre origin),
    events, lineups, pre-match risks and injury labels.

    Stoppage time is whole minutes and the announced stoppage equals the time played. Every
    outfield minute on the pitch is at risk under the hazard of docs/02_MATH.md §4 (stoppage
    minutes use bands 3 and 7); see ``_play_team`` for the order of competing ends. Pre-match risks
    are dated the day before kick-off, strictly before the match date.

    Squad players keep their identity, position and peak speed across seeds; p7 and everything
    in the match vary with ``seed``. Labels, events and pre-match risks come from one random
    stream and tracking from another, so ``tracking=False`` returns the same match without
    samples, much faster.
    """
    truth = truth or SynthTruth()
    if cfg is None:
        cfg = load_config()
    hz = float(cfg.synth.tick_hz) if sample_hz is None else float(sample_hz)
    labels_seq, tracking_seq = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(labels_seq)
    mid = match_id or f"synth-{seed}"
    kickoff = datetime(2026, 1, 1, 20, 0, tzinfo=UTC) + timedelta(days=seed % 300)
    clock = _make_clock(
        hz,
        int(rng.integers(_STOPPAGE_1H_MIN[0], _STOPPAGE_1H_MIN[1] + 1)),
        int(rng.integers(_STOPPAGE_2H_MIN[0], _STOPPAGE_2H_MIN[1] + 1)),
    )

    squad_ids = {t: [f"{t}-P{n + 1:02d}" for n in range(len(_SQUAD))] for t in _TEAM_IDS}
    vmax: dict[str, float] = {}
    lo_v, hi_v = cfg.synth.vmax_bounds_mps
    for team_index, team_id in enumerate(_TEAM_IDS):
        squad_rng = np.random.default_rng([cfg.seeds.default, team_index])
        draws = squad_rng.normal(cfg.synth.vmax_mean_mps, cfg.synth.vmax_sd_mps, len(_SQUAD))
        for pid, v in zip(squad_ids[team_id], np.clip(draws, lo_v, hi_v), strict=True):
            vmax[pid] = float(v)

    lineups: list[Lineup] = []
    prematch: list[PreMatchRisk] = []
    plays: dict[str, _TeamPlay] = {}
    for team_id in _TEAM_IDS:
        p7 = _draw_p7(rng, truth, len(_SQUAD))
        congested = rng.random(len(_SQUAD)) < 0.1
        days_since = rng.integers(2, 8, size=len(_SQUAD))
        members = tuple(
            LineupPlayer(
                player_id=squad_ids[team_id][n],
                shirt_number=n + 1,
                position_group=group,
                starting=n < _N_STARTERS,
            )
            for n, group in enumerate(_SQUAD)
        )
        lineups.append(Lineup(match_id=mid, team_id=team_id, kickoff_utc=kickoff, players=members))
        prematch.extend(
            PreMatchRisk(
                player_id=squad_ids[team_id][n],
                as_of_date=kickoff.date() - timedelta(days=1),
                p7=float(p7[n]),
                model_version="synth-weekly-1",
                congested=bool(congested[n]),
                days_since_last_match=float(days_since[n]),
            )
            for n in range(len(_SQUAD))
        )
        plays[team_id] = _play_team(rng, truth, clock, mid, team_id, squad_ids[team_id], p7)

    events = _clock_events(mid, clock)
    for team_id in _TEAM_IDS:
        events.extend(plays[team_id].events)
    events.extend(_incidents(rng, mid, clock, plays, squad_ids))
    events.sort(key=lambda e: (e.period, e.t_s))
    injuries = [label for team_id in _TEAM_IDS for label in plays[team_id].labels]

    if tracking:
        proc = speed_process_from_config(cfg)
        samples = _track(
            np.random.default_rng(tracking_seq), proc, clock, mid, plays, squad_ids, vmax
        )
    else:
        empty = np.zeros(0, dtype=np.float64)
        no_ids = np.zeros(0, dtype=object)
        samples = _samples_frame(
            mid, clock, np.zeros(0, dtype=np.int64), no_ids, no_ids, empty, empty, empty
        )

    return SynthMatch(
        match_id=mid,
        lineups=(lineups[0], lineups[1]),
        events=tuple(events),
        samples=samples,
        prematch=tuple(prematch),
        injuries=tuple(sorted(injuries, key=lambda x: (x.period, x.t_s_onset))),
        person_minutes=_match_person_minutes(mid, clock, truth, plays, squad_ids),
        vmax_mps=vmax,
        truth=truth,
    )


def _track(
    rng: np.random.Generator,
    proc: SpeedProcess,
    clock: _Clock,
    match_id: str,
    plays: dict[str, _TeamPlay],
    squad_ids: dict[str, list[str]],
    vmax: dict[str, float],
) -> pd.DataFrame:
    """Tracking samples of every spell, sorted by tick then player id."""
    order = sorted(vmax)
    code_of = {pid: c for c, pid in enumerate(order)}
    ticks: list[IntArray] = []
    codes: list[IntArray] = []
    speeds: list[FloatArray] = []
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    for team_id in _TEAM_IDS:
        for spell in plays[team_id].spells:
            pid = squad_ids[team_id][spell.roster]
            v, x, y = _spell_track(rng, proc, clock, spell, _SQUAD[spell.roster], vmax[pid])
            ticks.append(np.arange(spell.start_tick, spell.end_tick, dtype=np.int64))
            codes.append(np.full(v.shape[0], code_of[pid], dtype=np.int64))
            speeds.append(v)
            xs.append(x)
            ys.append(y)
    tick = np.concatenate(ticks)
    code = np.concatenate(codes)
    sort = np.lexsort((code, tick))
    ids = np.asarray(order, dtype=object)
    team_of = np.asarray([pid.split("-", 1)[0] for pid in order], dtype=object)
    return _samples_frame(
        match_id,
        clock,
        tick[sort],
        ids[code[sort]],
        team_of[code[sort]],
        np.concatenate(speeds)[sort],
        np.concatenate(xs)[sort],
        np.concatenate(ys)[sort],
    )


def _match_person_minutes(
    match_id: str,
    clock: _Clock,
    truth: SynthTruth,
    plays: dict[str, _TeamPlay],
    squad_ids: dict[str, list[str]],
) -> pd.DataFrame:
    """At-risk player-minutes of both teams, sorted by player then minute."""
    codes: list[IntArray] = []
    minutes: list[IntArray] = []
    zs: list[FloatArray] = []
    p7s: list[FloatArray] = []
    flags: list[BoolArray] = []
    for t, team_id in enumerate(_TEAM_IDS):
        play = plays[team_id]
        roster = np.asarray(play.rows_roster, dtype=np.int64)
        minute = np.asarray(play.rows_minute, dtype=np.int64)
        # Codes in this order sort like the player ids ("T1-P01" ... "T2-P20").
        codes.append(t * len(_SQUAD) + roster)
        minutes.append(minute)
        zs.append(play.z[roster, minute, :])
        p7s.append(play.p7[roster])
        flags.append(
            np.asarray([play.rows_y, play.rows_contact, play.rows_sub], dtype=np.bool_).reshape(
                3, -1
            )
        )
    code = np.concatenate(codes)
    minute_all = np.concatenate(minutes)
    sort = np.lexsort((minute_all, code))
    code, minute_all = code[sort], minute_all[sort]
    z_all = np.concatenate(zs)[sort]
    flag = np.concatenate(flags, axis=1)[:, sort]
    ids = np.asarray([pid for team_id in _TEAM_IDS for pid in squad_ids[team_id]], dtype=object)
    teams = np.repeat(np.asarray(_TEAM_IDS, dtype=object), len(_SQUAD))
    data: dict[str, object] = {
        "match_id": np.full(sort.shape[0], match_id, dtype=object),
        "player_id": ids[code],
        "team_id": teams[code],
        "period": clock.period[minute_all],
        "t_min": clock.t_min[minute_all],
        "band": clock.band[minute_all],
        "p7": np.concatenate(p7s)[sort],
    }
    for j, name in enumerate(truth.feature_names):
        data[name] = z_all[:, j]
    data["y"] = flag[0].astype(np.int64)
    data["contact"] = flag[1]
    data["substituted_after"] = flag[2]
    return pd.DataFrame(data)


# --------------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------------


def _model_frame(models: Sequence[BaseModel], model_type: type[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame(
        [m.model_dump(mode="json") for m in models], columns=list(model_type.model_fields)
    )


def match_tables(match: SynthMatch) -> dict[str, pd.DataFrame]:
    """The match as flat tables (samples, events, lineups, prematch, injuries, person_minutes)
    with contract field names; lineups have one row per squad player."""
    lineups = pd.DataFrame(
        [
            {
                "match_id": lu.match_id,
                "team_id": lu.team_id,
                "kickoff_utc": lu.kickoff_utc,
                **p.model_dump(mode="json"),
            }
            for lu in match.lineups
            for p in lu.players
        ]
    )
    prematch = _model_frame(match.prematch, PreMatchRisk)
    prematch["as_of_date"] = pd.to_datetime(prematch["as_of_date"]).dt.date
    return {
        "samples": match.samples,
        "events": _model_frame(match.events, MatchEvent),
        "lineups": lineups,
        "prematch": prematch,
        "injuries": _model_frame(match.injuries, InjuryLabel),
        "person_minutes": match.person_minutes,
    }


def write_match(match: SynthMatch, out_dir: str | Path) -> dict[str, Path]:
    """Write ``match_tables`` as Parquet files ``<name>.parquet`` under ``out_dir``.

    The bytes depend only on the match, so the same seed and config give identical files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in match_tables(match).items():
        path = out / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths
