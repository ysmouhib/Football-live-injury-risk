"""Finite Markov chain utilities (docs/02_MATH.md §13).

Used by the synthetic speed process (M2), norm diagnostics (M4) and the alert operating-point
analysis (M8). Pure functions on small dense matrices; rows of a transition matrix sum to 1.

Terminology: a finite chain is ergodic when it is irreducible (every state reaches every other)
and aperiodic. It then has a unique stationary distribution pi with pi P = pi, and P^t(x, .)
converges to pi from every start x.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_transition_matrix(p: ArrayLike, tol: float = 1e-9) -> FloatArray:
    """Validate a square row-stochastic matrix and return it as float64.

    Raises ``ValueError`` for a non-square or empty matrix, negative or non-finite entries, or a
    row that does not sum to 1 within ``tol``.
    """
    m = np.asarray(p, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        raise ValueError("transition matrix must be square and non-empty")
    if not np.all(np.isfinite(m)) or np.any(m < -tol):
        raise ValueError("transition matrix entries must be finite and >= 0")
    if not np.allclose(m.sum(axis=1), 1.0, rtol=0.0, atol=tol * m.shape[0]):
        raise ValueError("every row of a transition matrix must sum to 1")
    return np.clip(m, 0.0, None)


def _reach(adj: NDArray[np.bool_], start: int) -> NDArray[np.bool_]:
    seen = np.zeros(adj.shape[0], dtype=bool)
    seen[start] = True
    queue = deque([start])
    while queue:
        i = queue.popleft()
        for j in np.flatnonzero(adj[i] & ~seen):
            seen[j] = True
            queue.append(int(j))
    return seen


def is_irreducible(p: ArrayLike) -> bool:
    """True when every state can reach every other state."""
    adj = as_transition_matrix(p) > 0.0
    return bool(_reach(adj, 0).all() and _reach(adj.T, 0).all())


def period(p: ArrayLike) -> int:
    """Period of an irreducible chain: gcd of the lengths of its cycles (1 = aperiodic)."""
    m = as_transition_matrix(p)
    if not is_irreducible(m):
        raise ValueError("period is defined here for irreducible chains only")
    adj = m > 0.0
    level = np.full(m.shape[0], -1, dtype=np.int64)
    level[0] = 0
    queue = deque([0])
    while queue:
        i = queue.popleft()
        for j in np.flatnonzero(adj[i]):
            if level[j] < 0:
                level[j] = level[i] + 1
                queue.append(int(j))
    g = 0
    for i, j in zip(*np.nonzero(adj), strict=True):
        g = math.gcd(g, int(abs(level[i] + 1 - level[j])))
    return g


def is_ergodic(p: ArrayLike) -> bool:
    """Irreducible and aperiodic."""
    return is_irreducible(p) and period(p) == 1


def stationary_distribution(p: ArrayLike) -> FloatArray:
    """The unique pi with pi P = pi and sum(pi) = 1, for an irreducible chain.

    Periodic chains are accepted (pi still exists and is unique), but P^t does not converge to it.
    """
    m = as_transition_matrix(p)
    if not is_irreducible(m):
        raise ValueError("stationary distribution is unique only for irreducible chains")
    n = m.shape[0]
    a = m.T - np.eye(n)
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(a, b)
    pi = np.clip(pi, 0.0, None)
    return np.asarray(pi / pi.sum(), dtype=np.float64)


def slem(p: ArrayLike) -> float:
    """Second-largest eigenvalue modulus. 1.0 for periodic or reducible chains; 0 for n = 1."""
    m = as_transition_matrix(p)
    if m.shape[0] == 1:
        return 0.0
    mods = np.sort(np.abs(np.linalg.eigvals(m)))[::-1]
    return float(min(mods[1], 1.0))


def tv_distance(p: ArrayLike, q: ArrayLike) -> float:
    """Total-variation distance 0.5 * sum |p - q| between two distributions."""
    pp = np.asarray(p, dtype=np.float64)
    qq = np.asarray(q, dtype=np.float64)
    if pp.shape != qq.shape:
        raise ValueError("distributions must have the same shape")
    return float(0.5 * np.abs(pp - qq).sum())


def mixing_time(p: ArrayLike, eps: float = 0.25, max_steps: int = 100_000) -> int:
    """Smallest t with max over starts x of TV(P^t(x, .), pi) <= eps (exact, by powering).

    In steps of the chain. Raises ``ValueError`` if the chain is not ergodic or the bound is not
    reached within ``max_steps``.
    """
    m = as_transition_matrix(p)
    if not is_ergodic(m):
        raise ValueError("mixing time is defined for ergodic chains")
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must be in (0, 1)")
    pi = stationary_distribution(m)
    pt = np.eye(m.shape[0])
    for t in range(max_steps + 1):
        if 0.5 * np.abs(pt - pi).sum(axis=1).max() <= eps:
            return t
        pt = pt @ m
    raise ValueError(f"mixing time exceeds {max_steps} steps")


def semi_markov_occupancy(jump_matrix: ArrayLike, mean_dwell: ArrayLike) -> FloatArray:
    """Long-run share of time per state of a semi-Markov process.

    ``jump_matrix`` is the embedded chain of state changes (its diagonal may be zero) and
    ``mean_dwell`` the mean sojourn per state (any time unit). With q the stationary law of the
    jump chain, the share of time in state i is q_i * m_i / sum_j q_j * m_j.
    """
    q = stationary_distribution(jump_matrix)
    m = np.asarray(mean_dwell, dtype=np.float64)
    if m.shape != q.shape or np.any(m <= 0.0):
        raise ValueError("mean_dwell must be positive, one value per state")
    w = q * m
    return np.asarray(w / w.sum(), dtype=np.float64)


def jump_chain(p: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Split a Markov chain on a fixed time step into its jump chain and mean dwell (in steps).

    A Markov chain is the semi-Markov process whose dwell in state i is geometric with mean
    1 / (1 - P_ii). Raises ``ValueError`` for absorbing states (P_ii = 1).
    """
    m = as_transition_matrix(p)
    stay = np.diag(m).copy()
    if np.any(stay >= 1.0):
        raise ValueError("absorbing states have no jump chain")
    jumps = m.copy()
    np.fill_diagonal(jumps, 0.0)
    jumps = jumps / (1.0 - stay)[:, None]
    return np.asarray(jumps, dtype=np.float64), np.asarray(1.0 / (1.0 - stay), dtype=np.float64)


def propagate(
    init: ArrayLike, matrices: ArrayLike | Sequence[ArrayLike], steps: int | None = None
) -> FloatArray:
    """State distributions over time; row 0 is ``init``.

    ``matrices`` is either one transition matrix applied ``steps`` times (time-homogeneous) or a
    sequence of matrices applied in order (time-inhomogeneous; ``steps`` must then be None).
    """
    x0 = np.asarray(init, dtype=np.float64)
    if x0.ndim != 1 or np.any(x0 < 0.0) or not math.isclose(float(x0.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("init must be a probability vector")
    arr = np.asarray(matrices, dtype=np.float64)
    if arr.ndim == 2:
        if steps is None or steps < 0:
            raise ValueError("steps must be >= 0 for a single matrix")
        mats = [as_transition_matrix(arr)] * steps
    elif arr.ndim == 3:
        if steps is not None:
            raise ValueError("steps must be None when a sequence of matrices is given")
        mats = [as_transition_matrix(a) for a in arr]
    else:
        raise ValueError("matrices must be one matrix or a sequence of matrices")
    out = np.empty((len(mats) + 1, x0.shape[0]))
    out[0] = x0
    for t, mt in enumerate(mats, start=1):
        if mt.shape[0] != x0.shape[0]:
            raise ValueError("matrix size does not match init")
        out[t] = out[t - 1] @ mt
    return out


def empirical_transition_matrix(
    states: Sequence[int] | ArrayLike, n_states: int, prior_counts: ArrayLike | None = None
) -> FloatArray:
    """Row-normalised transition counts of a state sequence, plus optional Dirichlet pseudo-counts.

    ``prior_counts`` (n_states x n_states) is typically kappa * P_position, which shrinks a
    player's sparse rows toward his position group. A row with no counts and no prior raises.
    """
    s = np.asarray(states, dtype=np.int64)
    if s.ndim != 1 or np.any(s < 0) or np.any(s >= n_states):
        raise ValueError("states must be integers in [0, n_states)")
    counts = np.zeros((n_states, n_states))
    np.add.at(counts, (s[:-1], s[1:]), 1.0)
    if prior_counts is not None:
        prior = np.asarray(prior_counts, dtype=np.float64)
        if prior.shape != counts.shape or np.any(prior < 0.0):
            raise ValueError("prior_counts must be a non-negative n_states x n_states matrix")
        counts = counts + prior
    rows = counts.sum(axis=1)
    if np.any(rows <= 0.0):
        missing = np.flatnonzero(rows <= 0.0).tolist()
        raise ValueError(f"no transitions and no prior for states {missing}")
    return np.asarray(counts / rows[:, None], dtype=np.float64)


def ar1_effective_n(n: int, phi: float) -> float:
    """Effective number of independent values in the mean of n consecutive AR(1) values.

    n_eff = n / (1 + 2 * sum_{k=1}^{n-1} (1 - k/n) * phi^k); equals n when phi = 0 and tends to
    n (1 - phi) / (1 + phi) for large n.
    """
    if n < 1 or not -1.0 < phi < 1.0:
        raise ValueError("n >= 1 and -1 < phi < 1 required")
    k = np.arange(1, n)
    inflation = 1.0 + 2.0 * float(np.sum((1.0 - k / n) * phi**k))
    return n / inflation
