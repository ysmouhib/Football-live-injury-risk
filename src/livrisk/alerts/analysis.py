"""Alert operating-point analysis (docs/02_MATH.md §7.1).

The alert rules of ``alerts.logic`` turn a player's RR path into a finite chain over
(RR level, consecutive-update counters, alert state). Given a per-update transition matrix of RR
levels, a forward recursion (``markov.propagate``) over one match gives, per player and without
the team budget: expected Check openings, expected Discuss upgrades and expected flips between
open and cleared. Team figures multiply by the players on the pitch; the budget effect and the
probability of exhausting it by minute m use a Poisson approximation on the team openings and
are checked against replay (M9).

There is no stationary regime here: a match is about 95 updates and the budget is consumed, so
the analysis is finite-horizon by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from livrisk.config import Config

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OperatingPoint:
    """Expected alert behaviour per team-match for one alert configuration."""

    expected_checks: float
    expected_discuss_upgrades: float
    expected_flaps: float
    #: P(team budget exhausted by update u), u = 0 .. updates_per_match.
    p_budget_exhausted: tuple[float, ...]


def rr_level(rr: float, edges: Sequence[float]) -> int:
    """Index of the RR level: 0 below ``edges[0]``, len(edges) at or above ``edges[-1]``."""
    raise NotImplementedError("planned for M8")


def estimate_rr_level_matrix(
    rr_paths: Sequence[Sequence[float]], edges: Sequence[float]
) -> FloatArray:
    """Per-update transition matrix of RR levels from replayed RR paths (synthetic in development,
    shadow-mode paths later), using ``markov.empirical_transition_matrix``."""
    raise NotImplementedError("planned for M8")


def operating_point(
    rr_level_matrix: FloatArray, init: FloatArray, r_rest_ok: float, cfg: Config
) -> OperatingPoint:
    """Finite-horizon forward recursion over the alert chain for ``alerts.analysis`` settings.

    ``r_rest_ok`` is the share of updates whose R_rest clears the Check floor (independence
    assumed and stated in the report)."""
    raise NotImplementedError("planned for M8")
