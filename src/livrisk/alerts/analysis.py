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

import itertools
import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

from livrisk import markov
from livrisk.config import Config

FloatArray = NDArray[np.float64]

#: Alert states of the chain: no open alert, open Check, open Discuss.
_NONE, _CHECK, _DISCUSS = 0, 1, 2


@dataclass(frozen=True)
class OperatingPoint:
    """Expected alert behaviour per team-match for one alert configuration.

    ``expected_checks`` counts alert openings at either level (each opening is one card and one
    budget slot); ``expected_discuss_upgrades`` counts Check -> Discuss upgrades (free);
    ``expected_flaps`` counts open-state transitions (openings plus clearings).
    ``p_budget_exhausted[u]`` is the probability that team openings reach
    ``alerts.budget_per_match`` within the first u updates (u = 0 .. updates_per_match).
    """

    expected_checks: float
    expected_discuss_upgrades: float
    expected_flaps: float
    #: P(team budget exhausted by update u), u = 0 .. updates_per_match.
    p_budget_exhausted: tuple[float, ...]


def replay_agreement_rel(cfg: Config) -> float:
    """Required relative agreement between this analysis and replay (``alerts.analysis``)."""
    return float(cfg.alerts.analysis.replay_agreement_rel)


def _checked_edges(edges: Sequence[float]) -> tuple[float, ...]:
    """Validated RR level edges (positive, strictly ascending)."""
    e = tuple(float(x) for x in edges)
    if not e or any(x <= 0.0 for x in e) or any(b <= a for a, b in itertools.pairwise(e)):
        raise ValueError("rr level edges must be positive and strictly ascending")
    return e


def rr_level(rr: float, edges: Sequence[float]) -> int:
    """Index of the RR level: 0 below ``edges[0]``, len(edges) at or above ``edges[-1]``.

    ``rr`` is the engine's bounded relative risk (dimensionless, > 0); an RR exactly on an edge
    belongs to the level above it.
    """
    if not math.isfinite(rr) or rr <= 0.0:
        raise ValueError(f"rr must be positive and finite, got {rr}")
    return bisect_right(_checked_edges(edges), rr)


def estimate_rr_level_matrix(
    rr_paths: Sequence[Sequence[float]], edges: Sequence[float]
) -> FloatArray:
    """Per-update transition matrix of RR levels from replayed RR paths.

    ``rr_paths`` are per-player sequences of RR values at successive risk updates
    (dimensionless); transitions are counted within each path only, then row-normalised by
    ``markov.empirical_transition_matrix``. Raises ``ValueError`` when a level is never
    transitioned from in any path (its row is unidentifiable).
    """
    e = _checked_edges(edges)
    n = len(e) + 1
    counts = np.zeros((n, n), dtype=np.float64)
    for path in rr_paths:
        levels = np.asarray([rr_level(rr, e) for rr in path], dtype=np.int64)
        if levels.size >= 2:
            np.add.at(counts, (levels[:-1], levels[1:]), 1.0)
    return markov.empirical_transition_matrix(np.zeros(0, dtype=np.int64), n, prior_counts=counts)


def _threshold_level(threshold: float, edges: tuple[float, ...]) -> int:
    """RR level cut representing ``rr >= threshold`` exactly: the first level at or above it.

    The threshold must coincide with an edge, otherwise the rule is not representable on the
    level grid and the analysis would silently approximate it.
    """
    if threshold not in edges:
        raise ValueError(
            f"alert threshold {threshold} must equal an rr_level_edges entry for the "
            "operating-point chain to represent the rule exactly"
        )
    return bisect_right(edges, threshold)


def operating_point(
    rr_level_matrix: FloatArray, init: FloatArray, r_rest_ok: float, cfg: Config
) -> OperatingPoint:
    """Finite-horizon forward recursion over the alert chain for ``alerts.analysis`` settings.

    ``rr_level_matrix`` is the per-update transition matrix of RR levels (see
    ``estimate_rr_level_matrix``); ``init`` the level distribution at the first update;
    ``r_rest_ok`` the share of updates whose R_rest clears the Check floor (treated as
    independent of the level, an assumption stated in the report). Alert thresholds must
    coincide with ``rr_level_edges`` (``_threshold_level``); quality is assumed good or
    degraded throughout. The chain mirrors ``alerts.logic``: streaks capped at the persistence
    requirements, hysteresis clearing, upgrades without a floor re-gate, openings gated by the
    floor. The team budget enters only through the Poisson approximation on openings.
    """
    p = markov.as_transition_matrix(rr_level_matrix)
    edges = _checked_edges(cfg.alerts.analysis.rr_level_edges)
    n = len(edges) + 1
    if p.shape != (n, n):
        raise ValueError(f"rr_level_matrix must be {n}x{n} for {len(edges)} edges")
    x_init = np.asarray(init, dtype=np.float64)
    if x_init.shape != (n,):
        raise ValueError(f"init must have one entry per RR level ({n})")
    if not 0.0 <= r_rest_ok <= 1.0:
        raise ValueError(f"r_rest_ok must be in [0, 1], got {r_rest_ok}")
    alerts = cfg.alerts
    pc = int(alerts.check.persist_updates)
    pd = int(alerts.discuss.persist_updates)
    c_check = _threshold_level(float(alerts.check.rr_min), edges)
    c_discuss = _threshold_level(float(alerts.discuss.rr_min), edges)
    c_clear = _threshold_level(float(alerts.clear_rr_below), edges)
    states = list(
        itertools.product(range(n), range(pc + 1), range(pd + 1), range(pc + 1), range(3))
    )
    index = {s: i for i, s in enumerate(states)}
    size = len(states)
    t_mat = np.zeros((size, size))
    e_open = np.zeros((size, size))
    e_upgrade = np.zeros((size, size))
    e_clear = np.zeros((size, size))
    for level, sc, sd, cl, a in states:
        i = index[(level, sc, sd, cl, a)]
        for lp in range(n):
            prob = float(p[level, lp])
            if prob <= 0.0:
                continue
            scp = min(sc + 1, pc) if lp >= c_check else 0
            sdp = min(sd + 1, pd) if lp >= c_discuss else 0
            clp = min(cl + 1, pc) if lp < c_clear else 0
            branches: list[tuple[float, int, FloatArray | None]]
            if a != _NONE and clp >= pc:
                branches = [(prob, _NONE, e_clear)]
            elif a == _CHECK and sdp >= pd:
                branches = [(prob, _DISCUSS, e_upgrade)]
            elif a == _NONE and (sdp >= pd or scp >= pc):
                a_pass = _DISCUSS if sdp >= pd else _CHECK
                branches = [
                    (prob * r_rest_ok, a_pass, e_open),
                    (prob * (1.0 - r_rest_ok), _NONE, None),
                ]
            else:
                branches = [(prob, a, None)]
            for q, a_next, event in branches:
                j = index[(lp, scp, sdp, clp, a_next)]
                t_mat[i, j] += q
                if event is not None:
                    event[i, j] += q
    # First update: counters come from the first level alone; an opening is possible only
    # when a persistence requirement is 1.
    x0 = np.zeros(size)
    open_first = 0.0
    for level in range(n):
        if x_init[level] <= 0.0:
            continue
        scp = min(1, pc) if level >= c_check else 0
        sdp = min(1, pd) if level >= c_discuss else 0
        clp = min(1, pc) if level < c_clear else 0
        if sdp >= pd or scp >= pc:
            a_pass = _DISCUSS if sdp >= pd else _CHECK
            x0[index[(level, scp, sdp, clp, a_pass)]] += x_init[level] * r_rest_ok
            x0[index[(level, scp, sdp, clp, _NONE)]] += x_init[level] * (1.0 - r_rest_ok)
            open_first += float(x_init[level]) * r_rest_ok
        else:
            x0[index[(level, scp, sdp, clp, _NONE)]] += x_init[level]
    updates = int(cfg.alerts.analysis.updates_per_match)
    xs = markov.propagate(x0, t_mat, steps=updates - 1)
    open_ev = np.concatenate(([open_first], xs[:-1] @ e_open @ np.ones(size)))
    upgrade_ev = np.concatenate(([0.0], xs[:-1] @ e_upgrade @ np.ones(size)))
    clear_ev = np.concatenate(([0.0], xs[:-1] @ e_clear @ np.ones(size)))
    players = float(cfg.alerts.analysis.players_per_team)
    budget = int(alerts.budget_per_match)
    lam = players * np.concatenate(([0.0], np.cumsum(open_ev)))
    p_exhausted = poisson.sf(budget - 1, lam)
    return OperatingPoint(
        expected_checks=float(players * open_ev.sum()),
        expected_discuss_upgrades=float(players * upgrade_ev.sum()),
        expected_flaps=float(players * (open_ev.sum() + clear_ev.sum())),
        p_budget_exhausted=tuple(float(x) for x in p_exhausted),
    )
