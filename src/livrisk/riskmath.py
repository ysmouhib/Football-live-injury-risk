"""Reference implementation of docs/02_MATH.md.

Every formula used by the engine lives here. Other modules import these functions; they never
re-implement them. Section numbers refer to docs/02_MATH.md.

Conventions
-----------
- Probabilities are floats in [0, 1]; hazards are non-negative.
- Match time ``t_min`` is minutes since kick-off of the match: first half 0-45(+stoppage),
  second half 45-90(+stoppage). ``period`` is 1 or 2 (3, 4 = extra time).
- Small probabilities are handled with ``log1p``/``expm1``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Number of time bands in the match profile (docs/02_MATH.md §2).
N_BANDS: int = 8
#: Indices of the six regular 15-minute bands; the mean of exp(theta) over these is 1.
REGULAR_BANDS: tuple[int, ...] = (0, 1, 2, 4, 5, 6)
#: Regular match length used to spread the match cumulative hazard (minutes).
REGULAR_MINUTES: float = 90.0
#: Player-spells per team per match: 11 players x six 15-minute spells (§7).
PLAYER_SPELLS_PER_TEAM_MATCH: int = 66


# --------------------------------------------------------------------------------------------
# Probability <-> cumulative hazard
# --------------------------------------------------------------------------------------------


def cumhaz_from_prob(p: float) -> float:
    """Cumulative hazard H such that P(event) = 1 - exp(-H).

    Args:
        p: probability in [0, 1).
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"probability must be in [0, 1), got {p}")
    return -math.log1p(-p)


def prob_from_cumhaz(h: float) -> float:
    """P(event) = 1 - exp(-H), computed stably for small H."""
    if h < 0.0:
        raise ValueError(f"cumulative hazard must be >= 0, got {h}")
    return -math.expm1(-h)


def weekly_to_match_cumhaz(p7: float, match_share: float = 0.6) -> float:
    """§1. Cumulative hazard of a match injury over 90 regular minutes.

    Lambda_i = a * (-ln(1 - p7)).

    Args:
        p7: calibrated probability of an injury (>= 7 days) in the next 7 days.
        match_share: share ``a`` of the weekly hazard falling in one match (prior 0.6).
    """
    if not 0.0 < match_share <= 1.0:
        raise ValueError(f"match_share must be in (0, 1], got {match_share}")
    return match_share * cumhaz_from_prob(p7)


# --------------------------------------------------------------------------------------------
# Time profile g(t)
# --------------------------------------------------------------------------------------------


def band_index(t_min: float, period: int) -> int:
    """§2. Time band for match minute ``t_min`` in ``period``.

    Extra time (periods 3, 4) uses band 7 (end of match), a conservative choice.
    """
    if period == 1:
        if t_min < 15.0:
            return 0
        if t_min < 30.0:
            return 1
        if t_min < 45.0:
            return 2
        return 3
    if period == 2:
        if t_min < 60.0:
            return 4
        if t_min < 75.0:
            return 5
        if t_min < 90.0:
            return 6
        return 7
    if period in (3, 4):
        return 7
    raise ValueError(f"period must be 1, 2, 3 or 4, got {period}")


def normalise_time_profile(theta: ArrayLike) -> FloatArray:
    """§2. Convert band log-levels ``theta`` (length 8) into g levels with mean 1 over the regular
    bands. Returns exp(theta) / mean_regular(exp(theta))."""
    th = np.asarray(theta, dtype=np.float64)
    if th.shape != (N_BANDS,):
        raise ValueError(f"theta must have shape ({N_BANDS},), got {th.shape}")
    levels = np.exp(th)
    scale = float(np.mean(levels[list(REGULAR_BANDS)]))
    return np.asarray(levels / scale, dtype=np.float64)


def flat_time_profile() -> FloatArray:
    """g = 1 in every band (used before the profile is fitted)."""
    return np.ones(N_BANDS, dtype=np.float64)


# --------------------------------------------------------------------------------------------
# Baseline and live hazard
# --------------------------------------------------------------------------------------------


def baseline_hazard_per_min(match_cumhaz: float, g_level: float = 1.0) -> float:
    """§2. lambda_0(t) = (Lambda / 90) * g(t), per minute."""
    if match_cumhaz < 0.0 or g_level < 0.0:
        raise ValueError("match_cumhaz and g_level must be >= 0")
    return match_cumhaz / REGULAR_MINUTES * g_level


def shrink_feature(z: ArrayLike, reliability: ArrayLike, clip: float = 4.0) -> FloatArray:
    """§3. z* = rho * clip(z, -clip, clip). NaN (missing) becomes 0 (usual state)."""
    zz = np.asarray(z, dtype=np.float64)
    rho = np.asarray(reliability, dtype=np.float64)
    if np.any((rho < 0.0) | (rho > 1.0)):
        raise ValueError("reliability must be in [0, 1]")
    zz = np.where(np.isnan(zz), 0.0, np.clip(zz, -clip, clip))
    return np.asarray(rho * zz, dtype=np.float64)


def relative_risk(beta: ArrayLike, z_star: ArrayLike, rr_max: float | None = None) -> float:
    """§3. RR = exp(beta . z*); optionally bounded to [1/rr_max, rr_max] (engine safety bound)."""
    b = np.asarray(beta, dtype=np.float64)
    zs = np.asarray(z_star, dtype=np.float64)
    if b.shape != zs.shape:
        raise ValueError(f"beta {b.shape} and z_star {zs.shape} must have the same shape")
    rr = math.exp(float(b @ zs))
    if rr_max is not None:
        if rr_max < 1.0:
            raise ValueError("rr_max must be >= 1")
        rr = min(max(rr, 1.0 / rr_max), rr_max)
    return rr


def zscore(x: float, mean: float, sd: float) -> float:
    """Player-relative z-score; returns NaN when sd <= 0 or x is NaN."""
    if not sd > 0.0 or math.isnan(x):
        return math.nan
    return (x - mean) / sd


def shrink_mean(n: float, xbar: float, kappa: float, pop_mean: float) -> float:
    """§3. Empirical-Bayes norm: (n * xbar + kappa * mu_pos) / (n + kappa).

    ``n`` is the (effective) number of past matches, not the number of windows.
    """
    if n < 0 or kappa <= 0.0:
        raise ValueError("n must be >= 0 and kappa > 0")
    if n == 0:
        return pop_mean
    return (n * xbar + kappa * pop_mean) / (n + kappa)


def shrink_sd(n: float, sd_raw: float, kappa: float, pop_sd: float) -> float:
    """§3. sqrt((n * s_raw^2 + kappa * s_pos^2) / (n + kappa))."""
    if n < 0 or kappa <= 0.0 or pop_sd <= 0.0:
        raise ValueError("n >= 0, kappa > 0 and pop_sd > 0 required")
    if n == 0 or math.isnan(sd_raw):
        return pop_sd
    return math.sqrt((n * sd_raw**2 + kappa * pop_sd**2) / (n + kappa))


def recency_weights(ages_matches: ArrayLike, half_life_matches: float) -> FloatArray:
    """§3. Weight of each past match: 2^(-age / half_life), age in matches (0 = most recent)."""
    ages = np.asarray(ages_matches, dtype=np.float64)
    if half_life_matches <= 0.0 or np.any(ages < 0.0):
        raise ValueError("half_life_matches > 0 and ages >= 0 required")
    return np.asarray(np.exp2(-ages / half_life_matches), dtype=np.float64)


def kish_effective_n(weights: ArrayLike) -> float:
    """§3. Kish effective sample size (sum w)^2 / sum w^2; 0 for no weight."""
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0.0):
        raise ValueError("weights must be >= 0")
    s2 = float(np.sum(w * w))
    return 0.0 if s2 == 0.0 else float(np.sum(w)) ** 2 / s2


def weighted_mean_sd(values: ArrayLike, weights: ArrayLike) -> tuple[float, float]:
    """§3. Weighted mean and SD with reliability weights (unbiased for equal weights).

    var = sum w (x - m)^2 / (V1 - V2 / V1), V1 = sum w, V2 = sum w^2. SD is NaN when fewer than
    two values carry weight.
    """
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.shape != w.shape or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("values and non-negative weights of the same shape, sum > 0, required")
    v1 = float(w.sum())
    v2 = float(np.sum(w * w))
    mean = float(np.sum(w * x)) / v1
    denom = v1 - v2 / v1
    if np.count_nonzero(w) < 2 or denom <= 0.0:
        return mean, math.nan
    var = float(np.sum(w * (x - mean) ** 2)) / denom
    return mean, math.sqrt(var)


# --------------------------------------------------------------------------------------------
# Online outputs (§5)
# --------------------------------------------------------------------------------------------


def window_probability(hazards_per_min: ArrayLike, dt_min: float = 1.0) -> float:
    """P(event in window) = 1 - exp(-sum(lambda) * dt)."""
    lam = np.asarray(hazards_per_min, dtype=np.float64)
    if np.any(lam < 0.0):
        raise ValueError("hazards must be >= 0")
    return prob_from_cumhaz(float(lam.sum()) * dt_min)


def minute_schedule(
    t_from_min: float,
    period: int,
    horizon_min: float | None,
    stoppage_1h_min: float = 2.0,
    stoppage_2h_min: float = 3.0,
) -> list[tuple[float, int]]:
    """Minutes (start time, period) from ``t_from_min`` forward, in 1-minute steps.

    With ``horizon_min=None`` the schedule runs to the end of the second half including stoppage.
    A first-half call continues through half-time into the full second half (no hazard during the
    break). Extra time is not projected (calls in periods 3-4 return the remaining 15-min block).
    """
    out: list[tuple[float, int]] = []
    if period == 1:
        end_1h = 45.0 + stoppage_1h_min
        t = t_from_min
        while t < end_1h:
            out.append((t, 1))
            t += 1.0
        t = 45.0
        while t < 90.0 + stoppage_2h_min:
            out.append((t, 2))
            t += 1.0
    elif period == 2:
        t = t_from_min
        while t < 90.0 + stoppage_2h_min:
            out.append((t, 2))
            t += 1.0
    elif period in (3, 4):
        t = t_from_min
        block_end = (105.0 if period == 3 else 120.0) + 1.0
        while t < block_end:
            out.append((t, period))
            t += 1.0
    else:
        raise ValueError(f"period must be 1-4, got {period}")
    if horizon_min is not None:
        out = out[: math.ceil(horizon_min)]
    return out


def projected_probability(
    match_cumhaz: float,
    g_levels: ArrayLike,
    rr: float,
    t_from_min: float,
    period: int,
    horizon_min: float | None,
    stoppage_1h_min: float = 2.0,
    stoppage_2h_min: float = 3.0,
) -> float:
    """§5. P15 (``horizon_min=15``) or R_rest (``horizon_min=None``), holding RR constant.

    Use ``rr=1.0`` for R_usual.
    """
    g = np.asarray(g_levels, dtype=np.float64)
    if g.shape != (N_BANDS,):
        raise ValueError(f"g_levels must have shape ({N_BANDS},)")
    base = match_cumhaz / REGULAR_MINUTES
    total = 0.0
    for t, per in minute_schedule(
        t_from_min, period, horizon_min, stoppage_1h_min, stoppage_2h_min
    ):
        total += base * float(g[band_index(t, per)]) * rr
    return prob_from_cumhaz(total)


# --------------------------------------------------------------------------------------------
# Discrete-time cloglog model (§4)
# --------------------------------------------------------------------------------------------


def cloglog(p: ArrayLike) -> FloatArray:
    """ln(-ln(1 - p))."""
    pp = np.asarray(p, dtype=np.float64)
    if np.any((pp <= 0.0) | (pp >= 1.0)):
        raise ValueError("p must be in (0, 1)")
    return np.asarray(np.log(-np.log1p(-pp)), dtype=np.float64)


def inv_cloglog(eta: ArrayLike) -> FloatArray:
    """1 - exp(-exp(eta)), stable."""
    e = np.asarray(eta, dtype=np.float64)
    return np.asarray(-np.expm1(-np.exp(e)), dtype=np.float64)


def cloglog_loglik_terms(eta: ArrayLike, y: ArrayLike) -> FloatArray:
    """Per-row log-likelihood: y * ln(1 - e^{-mu}) - (1 - y) * mu, with mu = exp(eta)."""
    mu = np.exp(np.asarray(eta, dtype=np.float64))
    yy = np.asarray(y, dtype=np.float64)
    log_p = np.log(-np.expm1(-mu))
    return np.asarray(yy * log_p - (1.0 - yy) * mu, dtype=np.float64)


def cloglog_score_terms(eta: ArrayLike, y: ArrayLike) -> FloatArray:
    """d loglik / d eta = y * mu e^{-mu} / (1 - e^{-mu}) - (1 - y) * mu."""
    mu = np.exp(np.asarray(eta, dtype=np.float64))
    yy = np.asarray(y, dtype=np.float64)
    ratio = mu * np.exp(-mu) / (-np.expm1(-mu))
    return np.asarray(yy * ratio - (1.0 - yy) * mu, dtype=np.float64)


def fisher_weight(mu: ArrayLike) -> FloatArray:
    """Fisher weight for cloglog Bernoulli: mu^2 e^{-mu} / (1 - e^{-mu}) (-> mu as mu -> 0)."""
    m = np.asarray(mu, dtype=np.float64)
    return np.asarray(m * m * np.exp(-m) / (-np.expm1(-m)), dtype=np.float64)


def prior_sd_for_multiplier(multiplier: float = 2.0, prob: float = 0.95) -> float:
    """Normal prior SD such that |coef| <= ln(multiplier) with probability ``prob`` (two-sided)."""
    if multiplier <= 1.0 or not 0.0 < prob < 1.0:
        raise ValueError("multiplier > 1 and prob in (0, 1) required")
    z = NormalDist().inv_cdf(0.5 + prob / 2.0)
    return math.log(multiplier) / z


# --------------------------------------------------------------------------------------------
# Alerts, decisions, sample size (§7, §8)
# --------------------------------------------------------------------------------------------


def alert_ppv(recall: float, base_rate: float, flag_share: float) -> float:
    """§7. PPV = r * pi / x."""
    if not (0.0 <= recall <= 1.0 and 0.0 <= base_rate <= 1.0 and 0.0 < flag_share <= 1.0):
        raise ValueError("recall, base_rate in [0,1]; flag_share in (0,1]")
    return min(1.0, recall * base_rate / flag_share)


def alerts_per_match(flag_share: float, player_spells: int = PLAYER_SPELLS_PER_TEAM_MATCH) -> float:
    """§7. Expected alerts per team per match = x * 66."""
    return flag_share * player_spells


def action_threshold(cost: float, benefit: float) -> float:
    """§7. p* = C / (C + B)."""
    if cost < 0.0 or benefit <= 0.0:
        raise ValueError("cost >= 0 and benefit > 0 required")
    return cost / (cost + benefit)


def schoenfeld_events(hazard_ratio: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """§8. Events needed to detect ``hazard_ratio`` per SD of one standardised covariate."""
    if hazard_ratio <= 0.0 or hazard_ratio == 1.0:
        raise ValueError("hazard_ratio must be > 0 and != 1")
    nd = NormalDist()
    z = nd.inv_cdf(1.0 - alpha / 2.0) + nd.inv_cdf(power)
    return math.ceil(z * z / math.log(hazard_ratio) ** 2)


# --------------------------------------------------------------------------------------------
# Calibration (§9)
# --------------------------------------------------------------------------------------------


def platt_calibrate(p: ArrayLike, a: float, b: float) -> FloatArray:
    """sigma(a + b * logit(p))."""
    pp = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    logit = np.log(pp) - np.log1p(-pp)
    return np.asarray(1.0 / (1.0 + np.exp(-(a + b * logit))), dtype=np.float64)


def beta_calibrate(p: ArrayLike, a: float, b: float, c: float) -> FloatArray:
    """Beta calibration: 1 / (1 + 1 / (e^c * p^a / (1 - p)^b))."""
    pp = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    log_odds = c + a * np.log(pp) - b * np.log1p(-pp)
    return np.asarray(1.0 / (1.0 + np.exp(-log_odds)), dtype=np.float64)


def observed_expected_ratio(
    y: Sequence[float] | ArrayLike, p: Sequence[float] | ArrayLike
) -> float:
    """§9. O/E = sum(y) / sum(p)."""
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(p, dtype=np.float64)
    expected = float(pp.sum())
    if expected <= 0.0:
        raise ValueError("sum of predicted probabilities must be > 0")
    return float(yy.sum()) / expected


# --------------------------------------------------------------------------------------------
# Intervals and their audit (§5)
# --------------------------------------------------------------------------------------------


def log_cumhaz_sd_from_interval(p7_lo: float, p7_hi: float, level: float = 0.90) -> float:
    """§5. SD of ln H, H = -ln(1 - p7), implied by a central ``level`` interval [p7_lo, p7_hi].

    The engine draws ln(Lambda) ~ N(ln Lambda_hat, sd^2) alongside the parameter draws; the match
    share a0 cancels on the log scale.
    """
    if not 0.0 < p7_lo <= p7_hi < 1.0 or not 0.0 < level < 1.0:
        raise ValueError("0 < p7_lo <= p7_hi < 1 and 0 < level < 1 required")
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    return (math.log(-math.log1p(-p7_hi)) - math.log(-math.log1p(-p7_lo))) / (2.0 * z)


def importance_ess(log_weights: ArrayLike) -> float:
    """§5. Effective sample size of importance weights given on the log scale (stable)."""
    lw = np.asarray(log_weights, dtype=np.float64)
    if lw.ndim != 1 or lw.size == 0 or not np.all(np.isfinite(lw)):
        raise ValueError("log_weights must be a non-empty finite 1-D array")
    return kish_effective_n(np.exp(lw - float(np.max(lw))))


def weighted_quantile(values: ArrayLike, weights: ArrayLike, q: float) -> float:
    """§5. Quantile of a weighted sample (inverse of the weighted empirical CDF, midpoint rule)."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.shape != w.shape or x.ndim != 1 or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("1-D values and non-negative weights of the same shape required")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    keep = w > 0.0
    x, w = x[keep], w[keep]
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    if xs.size == 1:
        return float(xs[0])
    cdf = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return float(np.interp(q, cdf, xs))


# --------------------------------------------------------------------------------------------
# Evaluation helpers (§9)
# --------------------------------------------------------------------------------------------


def holm_adjust(p_values: ArrayLike) -> FloatArray:
    """§9. Holm step-down adjusted p-values (family-wise error control), in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must be a 1-D array in [0, 1]")
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


# --------------------------------------------------------------------------------------------
# Research track (§14): not part of the main fit
# --------------------------------------------------------------------------------------------


def decayed_event_load(
    event_times_s: ArrayLike, as_of_s: float, delta_s: float, tau_s: float
) -> float:
    """§14 R2. sum exp(-(t_cut - t_e) / tau) over events with t_e <= t_cut = as_of - delta."""
    t = np.asarray(event_times_s, dtype=np.float64)
    if tau_s <= 0.0 or delta_s < 0.0:
        raise ValueError("tau_s > 0 and delta_s >= 0 required")
    cut = as_of_s - delta_s
    past = t[t <= cut]
    return float(np.sum(np.exp(-(cut - past) / tau_s)))
