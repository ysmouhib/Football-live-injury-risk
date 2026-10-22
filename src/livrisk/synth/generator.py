"""Synthetic data with known ground truth (docs/05_VALIDATION.md §1).

Two paths:
- ``generate_person_minutes``: fast path for model tests (no tracking, features simulated directly).
- ``generate_match``: full path with 10 Hz tracking, events, lineups, pre-match risks, labels.

Both are deterministic given ``seed`` and draw injuries from the true model of docs/02_MATH.md §4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from livrisk.config import Config
from livrisk.contracts import InjuryLabel, Lineup, MatchEvent, PreMatchRisk

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_FEATURES: tuple[str, ...] = (
    "hsr_5m",
    "spr_5m",
    "recovery_10m",
    "hir_drop_15m",
    "nearmax_cum",
    "disrupt_5m",
)


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
    raise NotImplementedError("planned for M2")


def jump_matrix_for(process: SpeedProcess, band: int, position: str) -> FloatArray:
    """Scaled, renormalised jump matrix for one band and position group."""
    raise NotImplementedError("planned for M2")


def expected_occupancy(process: SpeedProcess, band: int, position: str) -> FloatArray:
    """Long-run share of time per zone: ``markov.semi_markov_occupancy`` of the band matrix."""
    raise NotImplementedError("planned for M2")


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
    band; bout lengths are ``dwell_min + Gamma(shape, (mean - min) / shape)``."""
    raise NotImplementedError("planned for M2")


@dataclass(frozen=True)
class SynthMatch:
    """One synthetic match (full path). ``samples`` has the TrackingSample columns."""

    match_id: str
    lineups: tuple[Lineup, Lineup]
    events: tuple[MatchEvent, ...]
    samples: pd.DataFrame
    prematch: tuple[PreMatchRisk, ...]
    injuries: tuple[InjuryLabel, ...]
    truth: SynthTruth = field(default_factory=SynthTruth)


def generate_person_minutes(
    n_player_matches: int,
    seed: int,
    truth: SynthTruth | None = None,
    informative_censoring: bool = True,
    n_clubs: int = 10,
    n_seasons: int = 3,
) -> pd.DataFrame:
    """Fast path. One row per player-minute at risk.

    Columns: match_id, player_id, club_id, season, period, t_min (minute start), band, p7,
    one column per feature name (z values, SD units), y (non-contact injury onset in this minute),
    contact (contact injury in this minute; ends the risk set), substituted_after (bool).
    """
    raise NotImplementedError("planned for M2")


def generate_match(
    seed: int,
    truth: SynthTruth | None = None,
    sample_hz: float = 10.0,
    match_id: str | None = None,
) -> SynthMatch:
    """Full path: 22 players at ``sample_hz`` with the ``SpeedProcess`` of the default config,
    positions from a heading random walk inside the pitch, events, lineups and labels."""
    raise NotImplementedError("planned for M2")
