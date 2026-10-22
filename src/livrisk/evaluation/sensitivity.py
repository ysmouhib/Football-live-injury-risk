"""Diagnostics and sensitivity analyses for Gate C (docs/05_VALIDATION.md §2).

- Input attribution: calibration of the pre-match input separately from the live layer.
- Substitution intensity: how strongly live signals drive substitutions (censoring diagnostic).
- Tipping point: how much higher the substituted players' hazard would have to be to erase each
  live effect.
- Aalen-Johansen cumulative incidence of non-contact and contact injury by match minute.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from livrisk.config import Config


@dataclass(frozen=True)
class InputAttribution:
    """Calibration of the time-profile model M1 on test rows, before any live signal."""

    oe_m1: float
    oe_m1_ci: tuple[float, float]
    #: Slope on ln(Lambda) when it enters as a covariate instead of a fixed offset (ideal 1).
    input_slope: float
    input_slope_ci: tuple[float, float]
    verdict: str  # "input_ok" | "input_miscalibrated"


def input_attribution(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> InputAttribution:
    """Fit M1 with ln(Lambda) as a covariate (prior N(1, input_slope_prior_sd^2)) on ``train``,
    report the slope and M1's O/E on ``test`` with player-cluster bootstrap CIs."""
    raise NotImplementedError("planned for M10")


def substitution_intensity(
    person_minutes: pd.DataFrame, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Discrete-time cloglog model of being substituted in minute j given the same live signals
    and band; returns coefficient, SD and hazard ratio per signal."""
    raise NotImplementedError("planned for M10")


def tipping_point(
    person_minutes: pd.DataFrame,
    feature_names: Sequence[str],
    cfg: Config,
    seed: int,
    multipliers: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0),
) -> pd.DataFrame:
    """For each multiplier psi, impute the substituted players' remaining minutes with hazard
    psi times the model hazard, refit, and report each beta with its interval. The tipping point
    of a signal is the smallest psi at which its interval includes 0."""
    raise NotImplementedError("planned for M10")


def aalen_johansen(person_minutes: pd.DataFrame) -> pd.DataFrame:
    """Cumulative incidence by match minute of non-contact and contact injury, with substitution
    and match end as censoring. Columns: t_min, cif_noncontact, cif_contact."""
    raise NotImplementedError("planned for M10")
