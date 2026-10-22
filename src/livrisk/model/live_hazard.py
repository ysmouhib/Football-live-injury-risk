"""Live hazard model, offline fitting (docs/02_MATH.md §4).

Discrete-time complementary log-log model with the pre-match hazard as an offset; MAP estimation
with normal priors; Laplace covariance. Use ``riskmath.cloglog_loglik_terms``,
``riskmath.cloglog_score_terms`` and ``riskmath.fisher_weight``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Priors:
    """Prior SDs (docs/02_MATH.md §4)."""

    alpha_sd: float = 0.5
    theta_sd: float = 0.5605
    eta_sd: float = 0.5
    beta_sd: float = 0.3537


@dataclass(frozen=True)
class Design:
    """Person-minute design. ``offset`` = ln(Lambda_i * dt / 90)."""

    X_live: FloatArray  # matrix name; N815 does not apply to dataclass fields
    offset: FloatArray
    y: FloatArray
    band: IntArray
    player_id: NDArray[np.str_]
    weights: FloatArray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class FittedHazard:
    """Fitted parameters.

    ``theta`` are normalised band log-levels (mean exp = 1 over regular bands); ``cov`` is the
    Laplace covariance over (alpha, theta free params, beta), with beta last."""

    feature_names: tuple[str, ...]
    alpha: float
    theta: tuple[float, ...]
    beta: tuple[float, ...]
    param_names: tuple[str, ...]
    cov: tuple[tuple[float, ...], ...]
    match_share_prior: float
    config_hash: str
    version: str

    def g_levels(self) -> FloatArray:
        """g per band (8,), via ``riskmath.normalise_time_profile``."""
        raise NotImplementedError("planned for M6")


def build_design(
    person_minutes: pd.DataFrame,
    feature_names: Sequence[str],
    match_share_prior: float,
    reliability: dict[str, float] | None = None,
    weights_col: str | None = None,
) -> Design:
    """Build the design from M2-style person-minute rows (z already computed)."""
    raise NotImplementedError("planned for M6")


def fit_map(design: Design, priors: Priors, config_hash: str, version: str) -> FittedHazard:
    """MAP via L-BFGS-B with analytic gradient; sum-to-zero theta (regular bands); Laplace cov."""
    raise NotImplementedError("planned for M6")


@dataclass(frozen=True)
class AuditReport:
    """Importance-sampling check of the Laplace intervals (docs/02_MATH.md §5)."""

    khat: float
    ess: float
    max_rel_quantile_diff: float
    passed: bool
    #: Normalised importance weights for the engine's stored draws, used only when not passed.
    draw_weights: tuple[float, ...] | None


def interval_audit(
    fitted: FittedHazard,
    design: Design,
    priors: Priors,
    battery: Sequence[tuple[float, float]],
    seed: int,
    draws: int = 4000,
    proposal_df: float = 5.0,
) -> AuditReport:
    """Draw from a Student-t around the MAP with the Laplace covariance, weight by exact posterior
    over proposal, compute Pareto k (Zhang and Stephens estimator on the largest weights) and the
    importance-corrected 5th/95th percentiles of P15 and R_rest for each (cumhaz, t_min) in
    ``battery``, using ``riskmath.weighted_quantile`` and ``riskmath.importance_ess``."""
    raise NotImplementedError("planned for M6")


def save_fitted(fitted: FittedHazard, path: str | Path) -> None:
    raise NotImplementedError("planned for M6")


def load_fitted(path: str | Path) -> FittedHazard:
    raise NotImplementedError("planned for M6")
