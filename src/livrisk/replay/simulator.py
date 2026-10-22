"""Replay simulator and look-ahead guard (docs/05_VALIDATION.md §3)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from livrisk.config import Config
from livrisk.model.live_hazard import FittedHazard
from livrisk.synth.generator import SynthMatch


class LookAheadError(RuntimeError):
    """Raised when any component reads data with t_s > as_of - delta."""


@dataclass(frozen=True)
class ReplaySettings:
    speed: float = 50.0  # times real time
    latency_s: float = 5.0
    dropout_rate: float = 0.0  # share of samples dropped at random
    burst_s: float = 0.0  # length of one dropout burst per half, seconds
    clock_drift_s: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class ReplayResult:
    features: pd.DataFrame
    risks: pd.DataFrame
    alerts: pd.DataFrame
    displayed: pd.DataFrame  # what the screen showed and when, after stale rules


def replay_match(
    match: SynthMatch, fitted: FittedHazard, cfg: Config, settings: ReplaySettings
) -> ReplayResult:
    """Run the live path on a recorded or synthetic match with a simulated clock."""
    raise NotImplementedError("planned for M9")
