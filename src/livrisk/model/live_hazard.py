"""Live hazard model: offline MAP fit with Laplace covariance (docs/02_MATH.md §4, §5.1).

The discrete-time cloglog model of docs/02_MATH.md §4 is fitted by maximum a posteriori with
an analytic gradient; uncertainty comes from a Laplace approximation (inverse observed Fisher
information plus prior precision). ``interval_audit`` checks the approximation once per model
version by self-normalised importance sampling (§5.1).

Parameters are ordered psi = (alpha, theta_0..theta_7, beta_1..beta_K), feature coefficients
last. Fitting uses the free parameterisation of §4: theta_6 is solved from the sum-to-zero
constraint over the regular bands, theta_6 = -(theta_0 + theta_1 + theta_2 + theta_4 + theta_5).
After the fit the band levels are converted to the §2 normalisation (mean of exp(theta) equal
to 1 over the regular bands) and the level is absorbed into alpha. Because of the exact linear
relation between the theta rows, the covariance has rank 8 + K; draws from the Laplace
approximation need an eigenvalue factorisation, not a Cholesky one.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln

from livrisk import riskmath
from livrisk.config import Config, IntervalAuditConfig

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Band whose level is solved from the sum-to-zero constraint of docs/02_MATH.md §4.
SOLVED_BAND: int = 6
#: Bands fitted as free parameters (all except SOLVED_BAND), in fitting order.
FREE_BANDS: tuple[int, ...] = tuple(b for b in range(riskmath.N_BANDS) if b != SOLVED_BAND)
#: Regular bands other than the solved one; their fitted levels sum to -theta_6.
SUMMED_BANDS: tuple[int, ...] = tuple(b for b in riskmath.REGULAR_BANDS if b != SOLVED_BAND)
#: Slots of the summed bands inside FREE_BANDS.
_SUMMED_SLOTS: tuple[int, ...] = tuple(FREE_BANDS.index(b) for b in SUMMED_BANDS)

_REQUIRED_COLUMNS = (
    "match_id",
    "player_id",
    "t_min",
    "band",
    "p7",
    "y",
    "contact",
    "substituted_after",
)

#: Draws processed per block when scoring proposal draws in the audit (bounds memory).
_AUDIT_CHUNK: int = 64

#: Clip on the linear predictor when scoring audit draws. Heavy-tailed proposal draws can
#: reach eta where exp(eta) overflows; their posterior mass is zero either way.
_ETA_MAX: float = 100.0


@dataclass(frozen=True)
class Priors:
    """Independent normal prior SDs of docs/02_MATH.md §4, on the log-hazard scale.

    Defaults match configs/default.yaml model.priors: theta_sd = ln(3)/1.96 (a band level at
    most x3 or /3 at 95 %), beta_sd = ln(2)/1.96 (a 1-SD change at most doubles or halves risk
    at 95 %). eta_sd is carried for the congestion interaction of §2, which is off by default
    and not fitted here.
    """

    alpha_sd: float = 0.5
    theta_sd: float = 0.5605
    eta_sd: float = 0.5
    beta_sd: float = 0.3537

    def __post_init__(self) -> None:
        if min(self.alpha_sd, self.theta_sd, self.eta_sd, self.beta_sd) <= 0.0:
            raise ValueError("prior SDs must be > 0")

    @classmethod
    def from_config(cls, cfg: Config) -> Priors:
        """Priors from the ``model.priors`` section of a loaded config."""
        p = cfg.model.priors
        return cls(alpha_sd=p.alpha_sd, theta_sd=p.theta_sd, eta_sd=p.eta_sd, beta_sd=p.beta_sd)


@dataclass(frozen=True)
class Design:
    """Training design for the cloglog fit of docs/02_MATH.md §4.

    One row per player-minute at risk. ``offset`` is ln(Lambda_i * Delta / 90) with the model
    interval Delta = 1 min; ``band`` is the §2 band index; ``player_id`` identifies the player
    of each row (used later by clustered resampling and substitution diagnostics). Rows already
    end at substitution, red card, match end or a contact injury, so every row contributes with
    its label ``y`` (§6); ``build_design`` validates that censoring.
    """

    feature_names: tuple[str, ...]
    x: FloatArray
    offset: FloatArray
    y: IntArray
    band: IntArray
    player_id: NDArray[np.str_]
    match_share: float

    def __post_init__(self) -> None:
        n = self.y.shape[0]
        if self.x.shape != (n, len(self.feature_names)):
            raise ValueError(
                f"x has shape {self.x.shape}, expected ({n}, {len(self.feature_names)})"
            )
        for name, arr in (("offset", self.offset), ("band", self.band)):
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
        if self.player_id.shape != (n,):
            raise ValueError(f"player_id has shape {self.player_id.shape}, expected ({n},)")

    @property
    def n_rows(self) -> int:
        """Number of person-minute rows."""
        return int(self.y.shape[0])

    @property
    def n_events(self) -> int:
        """Number of non-contact injury onsets in the design."""
        return int(self.y.sum())

    @property
    def n_features(self) -> int:
        """K, the number of live features."""
        return len(self.feature_names)


def build_design(
    person_minutes: pd.DataFrame, feature_names: Sequence[str], match_share: float
) -> Design:
    """Design matrix from at-risk person-minutes (synthetic fast/full path or real data).

    ``match_share`` is the share ``a`` of the weekly hazard falling in one match (docs/02 §1);
    the per-row offset is ln(weekly_to_match_cumhaz(p7, match_share) / 90). Training rows must
    already be censored per §6: a contact injury or a substitution ends the player's rows, and
    that last row still counts as exposure (y = 0 unless a non-contact injury starts in it).
    """
    names = tuple(feature_names)
    if not 1 <= len(names) <= 6:
        raise ValueError(f"the main fit uses 1 <= K <= 6 features, got {len(names)}")
    missing = [c for c in _REQUIRED_COLUMNS if c not in person_minutes.columns]
    missing += [c for c in names if c not in person_minutes.columns]
    if missing:
        raise ValueError(f"person_minutes is missing columns: {missing}")

    y = person_minutes["y"].to_numpy(dtype=np.int64)
    band = person_minutes["band"].to_numpy(dtype=np.int64)
    p7 = person_minutes["p7"].to_numpy(dtype=np.float64)
    x = person_minutes.loc[:, list(names)].to_numpy(dtype=np.float64)
    if np.any((y != 0) & (y != 1)):
        raise ValueError("y must be 0 or 1")
    if np.any((band < 0) | (band >= riskmath.N_BANDS)):
        raise ValueError(f"band must be in [0, {riskmath.N_BANDS})")
    if np.any((p7 <= 0.0) | (p7 >= 1.0)):
        raise ValueError("p7 must be in (0, 1)")
    if not np.all(np.isfinite(x)):
        raise ValueError("feature matrix contains non-finite values")

    uniq, inverse = np.unique(p7, return_inverse=True)
    lam = np.array(
        [riskmath.weekly_to_match_cumhaz(float(u), match_share) for u in uniq],
        dtype=np.float64,
    )
    offset = np.log(lam[inverse] / riskmath.REGULAR_MINUTES)

    ends = person_minutes["contact"].to_numpy(dtype=np.bool_) | person_minutes[
        "substituted_after"
    ].to_numpy(dtype=np.bool_)
    if np.any(ends):
        last = (
            person_minutes.groupby(["match_id", "player_id"], observed=True)["t_min"]
            .transform("max")
            .to_numpy(dtype=np.float64)
        )
        t_min = person_minutes["t_min"].to_numpy(dtype=np.float64)
        if bool(np.any(ends & (t_min < last))):
            raise ValueError(
                "found rows after a contact injury or a substitution within a player-match; "
                "training rows must be censored at the event (docs/02_MATH.md §6)"
            )

    player_id = np.asarray(person_minutes["player_id"].astype(str).to_numpy(), dtype=np.str_)
    return Design(
        feature_names=names,
        x=x,
        offset=offset,
        y=y,
        band=band,
        player_id=player_id,
        match_share=float(match_share),
    )


def _prior_sds(priors: Priors, n_features: int) -> FloatArray:
    """Prior SD per free parameter: alpha, the FREE_BANDS theta levels, then beta."""
    sds = [priors.alpha_sd] + [priors.theta_sd] * len(FREE_BANDS)
    sds += [priors.beta_sd] * n_features
    return np.asarray(sds, dtype=np.float64)


def _theta_full(phi: FloatArray) -> FloatArray:
    """All 8 band levels from the free vector; theta_6 solved from the §4 constraint."""
    theta = np.zeros(riskmath.N_BANDS, dtype=np.float64)
    theta[list(FREE_BANDS)] = phi[1 : 1 + len(FREE_BANDS)]
    theta[SOLVED_BAND] = -float(np.sum(phi[[1 + j for j in _SUMMED_SLOTS]]))
    return theta


def _linear_predictor(phi: FloatArray, design: Design) -> FloatArray:
    """eta = offset + alpha + theta_band + x . beta, mu = exp(eta) per docs/02_MATH.md §4."""
    theta = _theta_full(phi)
    beta = phi[1 + len(FREE_BANDS) :]
    eta = design.offset + phi[0] + theta[design.band] + design.x @ beta
    return np.asarray(eta, dtype=np.float64)


def neg_log_posterior(
    phi: FloatArray,
    design: Design,
    priors: Priors,
    sample_weight: FloatArray | None = None,
) -> float:
    """Negative log posterior (up to a constant) at the free parameter vector.

    ``phi`` is (alpha, theta_b for b in FREE_BANDS, beta_1..beta_K); all values on the
    log-hazard scale. ``sample_weight`` multiplies the per-row log-likelihood terms (the IPCW
    sensitivity of §4; off by default). The rows are training data: only matches free of shown
    alerts, or censored at the first shown alert (docs/02 §6).
    """
    eta = _linear_predictor(phi, design)
    terms = riskmath.cloglog_loglik_terms(eta, design.y)
    loglik = float(terms.sum()) if sample_weight is None else float(terms @ sample_weight)
    sds = _prior_sds(priors, design.n_features)
    return -loglik + 0.5 * float(np.sum((phi / sds) ** 2))


def neg_log_posterior_grad(
    phi: FloatArray,
    design: Design,
    priors: Priors,
    sample_weight: FloatArray | None = None,
) -> FloatArray:
    """Analytic gradient of ``neg_log_posterior``, vectorised over rows.

    Score terms come from ``riskmath.cloglog_score_terms``; the band block is aggregated with
    bincounts and mapped to the free parameterisation (a band-6 row contributes with a minus
    sign to the summed bands, §4).
    """
    eta = _linear_predictor(phi, design)
    s = riskmath.cloglog_score_terms(eta, design.y)
    if sample_weight is not None:
        s = s * sample_weight
    s_band = np.bincount(design.band, weights=s, minlength=riskmath.N_BANDS)
    g_theta = s_band[list(FREE_BANDS)]
    g_theta[list(_SUMMED_SLOTS)] -= s_band[SOLVED_BAND]
    g_beta = design.x.T @ s
    grad_ll = np.concatenate([[float(s.sum())], g_theta, g_beta])
    sds = _prior_sds(priors, design.n_features)
    return np.asarray(-grad_ll + phi / sds**2, dtype=np.float64)


def _expansion_matrix(n_features: int) -> FloatArray:
    """Linear map from the free vector to the full parameter vector (9 + K entries)."""
    n_free = 1 + len(FREE_BANDS) + n_features
    mat = np.zeros((1 + riskmath.N_BANDS + n_features, n_free), dtype=np.float64)
    mat[0, 0] = 1.0
    for slot, band in enumerate(FREE_BANDS):
        mat[1 + band, 1 + slot] = 1.0
    for slot in _SUMMED_SLOTS:
        mat[1 + SOLVED_BAND, 1 + slot] = -1.0
    mat[1 + riskmath.N_BANDS :, 1 + len(FREE_BANDS) :] = np.eye(n_features)
    return mat


def _laplace_covariance(
    phi_hat: FloatArray, design: Design, priors: Priors, sample_weight: FloatArray | None
) -> FloatArray:
    """Laplace covariance over the full parameter vector.

    Inverse of the observed Fisher information (``riskmath.fisher_weight``) plus prior
    precision, assembled in the free parameterisation and mapped to the full vector. The
    result has rank 8 + K: the theta rows satisfy the §4 constraint exactly.
    """
    mu = np.exp(_linear_predictor(phi_hat, design))
    w_row = riskmath.fisher_weight(mu)
    if sample_weight is not None:
        w_row = w_row * sample_weight
    k = design.n_features
    n_free = 1 + len(FREE_BANDS) + k
    c_band = np.bincount(design.band, weights=w_row, minlength=riskmath.N_BANDS)
    xw_band = np.zeros((riskmath.N_BANDS, k), dtype=np.float64)
    for j in range(k):
        xw_band[:, j] = np.bincount(
            design.band, weights=w_row * design.x[:, j], minlength=riskmath.N_BANDS
        )
    in_summed = np.zeros(len(FREE_BANDS), dtype=np.bool_)
    in_summed[list(_SUMMED_SLOTS)] = True

    info = np.zeros((n_free, n_free), dtype=np.float64)
    n_th = len(FREE_BANDS)
    info[0, 0] = float(w_row.sum())
    a_theta = c_band[list(FREE_BANDS)] - c_band[SOLVED_BAND] * in_summed
    info[0, 1 : 1 + n_th] = a_theta
    info[1 : 1 + n_th, 0] = a_theta
    info[1 : 1 + n_th, 1 : 1 + n_th] = np.diag(c_band[list(FREE_BANDS)]) + c_band[
        SOLVED_BAND
    ] * np.outer(in_summed, in_summed)
    a_beta = design.x.T @ w_row
    info[0, 1 + n_th :] = a_beta
    info[1 + n_th :, 0] = a_beta
    theta_beta = xw_band[list(FREE_BANDS)] - xw_band[SOLVED_BAND] * in_summed[:, None]
    info[1 : 1 + n_th, 1 + n_th :] = theta_beta
    info[1 + n_th :, 1 : 1 + n_th] = theta_beta.T
    info[1 + n_th :, 1 + n_th :] = (design.x * w_row[:, None]).T @ design.x
    sds = _prior_sds(priors, k)
    info[np.diag_indices_from(info)] += 1.0 / sds**2

    cov_free = np.linalg.inv(info)
    cov = _expansion_matrix(k) @ cov_free @ _expansion_matrix(k).T
    return np.asarray(0.5 * (cov + cov.T), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class FittedHazard:
    """MAP point and Laplace covariance of the docs/02_MATH.md §4 model.

    ``point`` and the rows and columns of ``cov`` follow ``param_names``: (alpha,
    theta_0..theta_7, beta_<feature>...). The theta levels carry the §2 normalisation (mean of
    exp(theta) equal to 1 over the regular bands) with the level absorbed into alpha. ``cov``
    has exact rank 8 + K because theta_6 is solved from the other regular bands; posterior
    draws need an eigenvalue factorisation of ``cov``.
    """

    param_names: tuple[str, ...]
    point: FloatArray
    cov: FloatArray
    feature_names: tuple[str, ...]
    priors: Priors
    config_hash: str
    version: str
    n_rows: int
    n_events: int

    @property
    def alpha(self) -> float:
        """Match-share correction on the log-hazard scale (0 reproduces the prior share)."""
        return float(self.point[0])

    @property
    def theta(self) -> FloatArray:
        """Band log-levels (8), normalised per docs/02_MATH.md §2."""
        return np.asarray(self.point[1 : 1 + riskmath.N_BANDS], dtype=np.float64)

    @property
    def beta(self) -> FloatArray:
        """Live coefficients, log hazard ratio per SD, in ``feature_names`` order."""
        return np.asarray(self.point[1 + riskmath.N_BANDS :], dtype=np.float64)

    def g_levels(self) -> FloatArray:
        """Time profile g per band, mean 1 over the regular bands (docs/02_MATH.md §2)."""
        return riskmath.normalise_time_profile(self.theta)

    def save(self, path: str | Path) -> None:
        """Write the fit as JSON: point, covariance, priors, config hash and version."""
        payload = {
            "param_names": list(self.param_names),
            "point": [float(v) for v in self.point],
            "cov": [[float(v) for v in row] for row in self.cov],
            "feature_names": list(self.feature_names),
            "priors": {
                "alpha_sd": self.priors.alpha_sd,
                "theta_sd": self.priors.theta_sd,
                "eta_sd": self.priors.eta_sd,
                "beta_sd": self.priors.beta_sd,
            },
            "config_hash": self.config_hash,
            "version": self.version,
            "n_rows": self.n_rows,
            "n_events": self.n_events,
        }
        Path(path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> FittedHazard:
        """Read a fit written by ``save``; raise ``ValueError`` on inconsistent shapes."""
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        param_names = tuple(str(v) for v in raw["param_names"])
        point = np.asarray(raw["point"], dtype=np.float64)
        cov = np.asarray(raw["cov"], dtype=np.float64)
        n_par = len(param_names)
        if point.shape != (n_par,) or cov.shape != (n_par, n_par):
            raise ValueError(
                f"inconsistent shapes: {n_par} names, point {point.shape}, cov {cov.shape}"
            )
        priors_raw = raw["priors"]
        return cls(
            param_names=param_names,
            point=point,
            cov=cov,
            feature_names=tuple(str(v) for v in raw["feature_names"]),
            priors=Priors(
                alpha_sd=float(priors_raw["alpha_sd"]),
                theta_sd=float(priors_raw["theta_sd"]),
                eta_sd=float(priors_raw["eta_sd"]),
                beta_sd=float(priors_raw["beta_sd"]),
            ),
            config_hash=str(raw["config_hash"]),
            version=str(raw["version"]),
            n_rows=int(raw["n_rows"]),
            n_events=int(raw["n_events"]),
        )


def fit_map(
    design: Design,
    priors: Priors,
    *,
    config_hash: str,
    version: str,
    sample_weight: FloatArray | None = None,
) -> FittedHazard:
    """MAP fit of the §4 cloglog model by L-BFGS-B with the analytic gradient.

    ``sample_weight`` carries optional per-row IPCW weights (the substitution sensitivity of
    §4; the weights themselves come from the substitution-intensity diagnostic, planned for
    M10). Raises ``RuntimeError`` if the optimiser does not converge: a fit that failed to
    converge must never reach the engine. Training rows only, per docs/02 §6.
    """
    k = design.n_features
    weight = None
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != (design.n_rows,) or not np.all(np.isfinite(weight)):
            raise ValueError("sample_weight must be finite with one entry per row")
        if np.any(weight < 0.0) or not np.any(weight > 0.0):
            raise ValueError("sample_weight must be >= 0 with at least one positive entry")
    result = minimize(
        neg_log_posterior,
        np.zeros(1 + len(FREE_BANDS) + k, dtype=np.float64),
        args=(design, priors, weight),
        method="L-BFGS-B",
        jac=neg_log_posterior_grad,
        options={"maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"hazard fit did not converge: {result.message}")
    phi_hat = np.asarray(result.x, dtype=np.float64)
    cov = _laplace_covariance(phi_hat, design, priors, weight)

    point = _expansion_matrix(k) @ phi_hat
    level = math.log(float(np.mean(np.exp(point[[1 + b for b in riskmath.REGULAR_BANDS]]))))
    point[0] += level
    point[1 : 1 + riskmath.N_BANDS] -= level

    param_names = (
        "alpha",
        *(f"theta_{b}" for b in range(riskmath.N_BANDS)),
        *(f"beta_{name}" for name in design.feature_names),
    )
    logger.info(
        "hazard fit: %d rows, %d events, %d iterations", design.n_rows, design.n_events, result.nit
    )
    return FittedHazard(
        param_names=param_names,
        point=np.asarray(point, dtype=np.float64),
        cov=cov,
        feature_names=design.feature_names,
        priors=priors,
        config_hash=config_hash,
        version=version,
        n_rows=design.n_rows,
        n_events=design.n_events,
    )


@dataclass(frozen=True, eq=False)
class AuditReport:
    """Outcome of the docs/02_MATH.md §5.1 interval audit for one model version.

    ``passed`` is khat < khat_max and every battery percentile within quantile_rel_tol of the
    Laplace one. ``proposal_draws`` and ``weights`` (self-normalised) are the draws the engine
    falls back to when the audit fails: it then reports weighted percentiles instead of Laplace
    ones.
    """

    passed: bool
    khat: float
    ess: float
    n_draws: int
    tail_draws: int
    battery_cases: int
    max_rel_diff: float
    proposal_draws: FloatArray
    weights: FloatArray


def _gpdfit_zs(x: FloatArray) -> float:
    """GPD shape k of non-negative exceedances, Zhang & Stephens (2009) empirical-Bayes fit.

    Posterior mean over a profile-likelihood grid for the rate b = -k / sigma, followed by the
    standard weakly informative shrinkage of k toward 0.5 (10 pseudo-observations), as used by
    the PSIS reference implementations. Returns 0 for a degenerate (all-zero) tail.
    """
    n: int = x.shape[0]
    x_max = float(x[-1])
    if x_max <= 0.0:
        return 0.0
    x_quartile = float(x[int(n / 4 + 0.5) - 1])
    if x_quartile <= 0.0:
        x_quartile = x_max
    grid = 30 + int(math.sqrt(n))
    j = np.arange(1, grid + 1, dtype=np.float64)
    b = (1.0 - np.sqrt(grid / (j - 0.5))) / (3.0 * x_quartile) + 1.0 / x_max
    k_grid = np.mean(np.log1p(-b[:, None] * x[None, :]), axis=1)
    log_post = n * (np.log(-b / k_grid) - k_grid - 1.0)
    w = np.exp(log_post - float(np.max(log_post)))
    w /= w.sum()
    b_post = float(b @ w)
    k_hat = float(np.mean(np.log1p(-b_post * x)))
    return (n * k_hat + 10.0 * 0.5) / (n + 10.0)


def _pareto_khat(log_weights: FloatArray) -> tuple[float, int]:
    """Pareto k-hat of the upper tail of the self-normalised importance weights.

    Tail size M = min(S/5, 3 sqrt(S)) (docs/02_MATH.md §5.1); the GPD is fitted to the
    exceedances of the M largest weights over the largest weight below the tail. Returns
    (khat, M).
    """
    lw = np.asarray(log_weights, dtype=np.float64)
    s = lw.shape[0]
    m = int(min(s // 5, 3.0 * math.sqrt(s)))
    if m < 5:
        raise ValueError(f"need at least 5 tail weights, got tail size {m}")
    w_sorted = np.sort(np.exp(lw - float(np.max(lw))))
    threshold = float(w_sorted[s - m - 1])
    exceedances = w_sorted[s - m :] - threshold
    return _gpdfit_zs(exceedances), m


def _to_free_batch(draws: FloatArray, n_features: int) -> FloatArray:
    """Free parameterisation of stored-space draws (undoes the §2 level shift)."""
    theta = draws[:, 1 : 1 + riskmath.N_BANDS]
    shift = (theta[:, SOLVED_BAND] + theta[:, list(SUMMED_BANDS)].sum(axis=1)) / float(
        len(riskmath.REGULAR_BANDS)
    )
    phi = np.empty((draws.shape[0], 1 + len(FREE_BANDS) + n_features), dtype=np.float64)
    phi[:, 0] = draws[:, 0] + shift
    phi[:, 1 : 1 + len(FREE_BANDS)] = theta[:, list(FREE_BANDS)] - shift[:, None]
    phi[:, 1 + len(FREE_BANDS) :] = draws[:, 1 + riskmath.N_BANDS :]
    return phi


def _log_posterior_draws(draws: FloatArray, design: Design, priors: Priors) -> FloatArray:
    """Unnormalised log posterior per stored-space draw, scored in blocks over draws.

    With y in {0, 1} the row terms of ``riskmath.cloglog_loglik_terms`` sum to
    sum_events(ln(1 - e^-mu) + mu) - sum_all(mu), so only the event rows need the log term and
    mu = exp(eta) is evaluated once per element; the event rows still go through
    ``riskmath.cloglog_loglik_terms``. The prior is evaluated in the free parameterisation the
    fit was made in.
    """
    s = draws.shape[0]
    k = design.n_features
    out = np.empty(s, dtype=np.float64)
    event = np.flatnonzero(design.y == 1)
    x_ev, band_ev, off_ev = design.x[event], design.band[event], design.offset[event]
    ones_ev = np.ones((event.size, 1), dtype=np.float64)
    beta_slice = slice(1 + riskmath.N_BANDS, None)
    for lo in range(0, s, _AUDIT_CHUNK):
        hi = min(lo + _AUDIT_CHUNK, s)
        d = draws[lo:hi]
        eta = design.x @ d[:, beta_slice].T
        eta += design.offset[:, None]
        eta += d[:, 0][None, :]
        eta += d[:, 1 : 1 + riskmath.N_BANDS].T[design.band]
        np.minimum(eta, _ETA_MAX, out=eta)
        np.exp(eta, out=eta)
        ll_events = riskmath.cloglog_loglik_terms(
            off_ev[:, None]
            + d[:, 0][None, :]
            + d[:, 1 : 1 + riskmath.N_BANDS].T[band_ev]
            + x_ev @ d[:, beta_slice].T,
            ones_ev,
        ).sum(axis=0)
        out[lo:hi] = ll_events + eta[event].sum(axis=0) - eta.sum(axis=0)
    phi = _to_free_batch(draws, k)
    sds = _prior_sds(priors, k)
    out -= 0.5 * np.sum((phi / sds[None, :]) ** 2, axis=1)
    return out


def _window_band_counts(t_min: float, period: int, horizon_min: float | None) -> FloatArray:
    """Minutes per band in the §5 window from (t_min, period), schedule from riskmath."""
    counts = np.zeros(riskmath.N_BANDS, dtype=np.float64)
    for t, p in riskmath.minute_schedule(t_min, period, horizon_min):
        counts[riskmath.band_index(t, p)] += 1.0
    return counts


def _battery(design: Design, n_cases: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Deterministic (Lambda, minute) battery for the audit.

    Lambda levels are 20 quantiles of the player-match cumulative hazards in the design,
    crossed with an even grid of minutes; per case the minute counts per band of the P15
    (15 min) and R_rest (to match end) windows. Returns (lambda, counts_p15, counts_rest).
    """
    lam_rows = np.exp(design.offset) * riskmath.REGULAR_MINUTES
    lam_levels = np.quantile(np.unique(lam_rows), (np.arange(20) + 0.5) / 20.0)
    minutes = np.linspace(3.0, 88.0, -(-n_cases // 20))
    grid = [
        (level, float(t), 1 if t < 45.0 else 2)
        for level in (float(v) for v in lam_levels)
        for t in minutes
    ]
    if len(grid) > n_cases:
        pick = np.floor(np.arange(n_cases) * len(grid) / n_cases).astype(np.int64)
        grid = [grid[int(i)] for i in pick]
    lam = np.array([g[0] for g in grid], dtype=np.float64)
    counts_p15 = np.array([_window_band_counts(g[1], g[2], 15.0) for g in grid])
    counts_rest = np.array([_window_band_counts(g[1], g[2], None) for g in grid])
    return lam, counts_p15, counts_rest


def _battery_probs(draws: FloatArray, lam: FloatArray, counts: FloatArray) -> FloatArray:
    """Window injury probability per case (rows) and draw (columns).

    1 - exp(-Lambda * exp(alpha) * (counts . g) / 90), z at the usual state (RR = 1), matching
    ``riskmath.projected_probability`` with rr = 1.
    """
    alpha = draws[:, 0]
    g = np.exp(draws[:, 1 : 1 + riskmath.N_BANDS])
    base = counts @ g.T / riskmath.REGULAR_MINUTES
    cum = lam[:, None] * np.exp(alpha)[None, :] * base
    return np.asarray(-np.expm1(-cum), dtype=np.float64)


def _battery_max_rel_diff(
    draws: FloatArray,
    weights: FloatArray,
    weights_laplace: FloatArray,
    lam: FloatArray,
    counts_p15: FloatArray,
    counts_rest: FloatArray,
) -> float:
    """Largest relative |IS - Laplace| difference over cases, windows and the 5th/95th
    percentiles. Both percentile sets come from the same proposal draws; the Laplace side is
    re-weighted to the Gaussian target, so sampling noise shared by the two cancels."""
    worst = 0.0
    for counts in (counts_p15, counts_rest):
        p = _battery_probs(draws, lam, counts)
        for c in range(lam.shape[0]):
            for q in (0.05, 0.95):
                q_is = riskmath.weighted_quantile(p[c], weights, q)
                q_lap = riskmath.weighted_quantile(p[c], weights_laplace, q)
                worst = max(worst, abs(q_is - q_lap) / q_lap)
    return worst


def _psd_eigh(cov: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Non-zero eigenvalues and eigenvectors of a symmetric PSD matrix.

    The Laplace covariance has exact rank 8 + K (the theta rows satisfy the sum-to-zero
    constraint of docs/02_MATH.md §4), so a numerical rank tolerance decides which eigenvalues
    count as non-zero.
    """
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    tol = float(vals[-1]) * cov.shape[0] * np.finfo(np.float64).eps
    keep = vals > max(tol, 0.0)
    return np.asarray(vals[keep], dtype=np.float64), np.asarray(vecs[:, keep], dtype=np.float64)


def laplace_factor(cov: FloatArray) -> FloatArray:
    """Factor ``factor`` with ``factor @ factor.T == cov``, from the non-zero eigenspace.

    Posterior draws from the Laplace approximation are ``point + z @ factor.T`` with ``z``
    standard normal. Because the covariance of a ``FittedHazard`` is exactly singular, draws
    must come from this factor, not from a Cholesky factorisation.
    """
    vals, basis = _psd_eigh(cov)
    return np.asarray(basis * np.sqrt(vals)[None, :], dtype=np.float64)


def interval_audit(
    fitted: FittedHazard, design: Design, cfg: IntervalAuditConfig, *, seed: int = 0
) -> AuditReport:
    """§5.1 audit of the Laplace approximation for one fitted model version.

    Draws cfg.draws vectors from a multivariate Student-t (cfg.proposal_df degrees of freedom)
    centred at the MAP point with the Laplace covariance as scale, scores them under the true
    log posterior of ``design`` and computes the Pareto k-hat of the tail weights and the ESS.
    On a deterministic battery of cfg.battery_matches (Lambda, minute) cases at the usual live
    state, the importance-weighted 5th and 95th percentiles of P15 and R_rest are compared
    with the Laplace ones, the latter estimated from the same proposal draws re-weighted to
    the Gaussian target (shared draws cancel sampling noise in the comparison). Passes when
    khat < cfg.khat_max and every percentile agrees within cfg.quantile_rel_tol (relative).
    Deterministic given ``seed``.
    """
    k = len(fitted.feature_names)
    if design.n_features != k:
        raise ValueError("design and fitted model disagree on the number of features")
    psi = np.asarray(fitted.point, dtype=np.float64)
    cov = np.asarray(fitted.cov, dtype=np.float64)
    vals, basis = _psd_eigh(cov)
    rank = int(vals.size)
    factor = basis * np.sqrt(vals)[None, :]

    s = int(cfg.draws)
    df_t = float(cfg.proposal_df)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((s, rank))
    scale = np.sqrt(df_t / rng.chisquare(df_t, size=s))
    draws = psi + (z * scale[:, None]) @ factor.T

    log_post = _log_posterior_draws(draws, design, fitted.priors)
    q_proj = (draws - psi) @ basis / np.sqrt(vals)[None, :]
    maha = np.einsum("ij,ij->i", q_proj, q_proj)
    log_prop = (
        gammaln(0.5 * (df_t + rank))
        - gammaln(0.5 * df_t)
        - 0.5 * rank * math.log(df_t * math.pi)
        - 0.5 * float(np.log(vals).sum())
        - 0.5 * (df_t + rank) * np.log1p(maha / df_t)
    )
    log_w = log_post - log_prop
    ess = riskmath.importance_ess(log_w)
    khat, tail = _pareto_khat(log_w)
    w = np.exp(log_w - float(np.max(log_w)))
    w /= w.sum()
    # Weights targeting the Gaussian (Laplace) law from the same proposal draws.
    log_w_laplace = -0.5 * maha - log_prop
    w_laplace = np.exp(log_w_laplace - float(np.max(log_w_laplace)))
    w_laplace /= w_laplace.sum()

    lam, counts_p15, counts_rest = _battery(design, int(cfg.battery_matches))
    max_rel = _battery_max_rel_diff(draws, w, w_laplace, lam, counts_p15, counts_rest)

    passed = bool(khat < cfg.khat_max and max_rel <= cfg.quantile_rel_tol)
    if not passed:
        logger.warning(
            "interval audit failed: khat=%.3f (max %.2f), max rel diff %.3f (tol %.2f)",
            khat,
            cfg.khat_max,
            max_rel,
            cfg.quantile_rel_tol,
        )
    return AuditReport(
        passed=passed,
        khat=float(khat),
        ess=float(ess),
        n_draws=s,
        tail_draws=tail,
        battery_cases=int(lam.shape[0]),
        max_rel_diff=max_rel,
        proposal_draws=draws,
        weights=w,
    )
