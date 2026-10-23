"""Nested comparators of docs/05_VALIDATION.md §2, fitted on training rows, scored on test rows.

- M0: pre-match Lambda spread evenly over the minutes (g = 1, beta = 0); nothing is fitted.
- M1: M0 x fitted time profile g(t) (alpha and the band levels by MAP, no live features).
- M2: M1 x live multiplier on the main features (the full model of docs/02_MATH.md §4).

All predictions are per-minute hazards (lambda * Delta) from ``live_hazard.predict_hazard``.
Fits see training rows only; the caller applies the training-data policy before fitting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from livrisk.config import Config, config_hash
from livrisk.model.live_hazard import (
    Design,
    FittedHazard,
    Priors,
    build_design,
    fit_map,
    predict_hazard,
)

FloatArray = NDArray[np.float64]

MODELS: tuple[str, ...] = ("m0", "m1", "m2")


def designs(
    person_minutes: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> tuple[Design, Design]:
    """(M1 design without features, M2 design with ``feature_names``) of the same rows."""
    full = build_design(person_minutes, feature_names, float(cfg.prematch.match_share_prior))
    return without_features(full), full


def without_features(design: Design) -> Design:
    """The same rows with no live features (the M1 design)."""
    return replace(design, feature_names=(), x=design.x[:, :0])


def with_covariates(design: Design, names: Sequence[str], x: FloatArray) -> Design:
    """The same rows and offset with other covariates (benchmarks, diagnostics)."""
    return replace(design, feature_names=tuple(names), x=np.asarray(x, dtype=np.float64))


@dataclass(frozen=True, eq=False)
class ModelFits:
    """The fitted M1 and M2 of one training set (M0 has no parameters)."""

    m1: FittedHazard
    m2: FittedHazard


def fit_models(
    train: pd.DataFrame,
    feature_names: Sequence[str],
    cfg: Config,
    version: str,
    sample_weight: FloatArray | None = None,
) -> ModelFits:
    """MAP fits of M1 and M2 with the priors of ``model.priors`` on training rows."""
    d1, d2 = designs(train, feature_names, cfg)
    priors = Priors.from_config(cfg)
    h = config_hash(cfg)
    return ModelFits(
        m1=fit_map(d1, priors, config_hash=h, version=f"{version}-m1", sample_weight=sample_weight),
        m2=fit_map(d2, priors, config_hash=h, version=f"{version}-m2", sample_weight=sample_weight),
    )


def predict_models(
    fits: ModelFits, rows: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> dict[str, FloatArray]:
    """Per-minute hazards of M0, M1 and M2 for ``rows`` (minutes at risk, no data after them)."""
    d1, d2 = designs(rows, feature_names, cfg)
    return {
        "m0": np.asarray(np.exp(d1.offset), dtype=np.float64),
        "m1": predict_hazard(fits.m1, d1),
        "m2": predict_hazard(fits.m2, d2),
    }
