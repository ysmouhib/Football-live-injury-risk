"""Diagnostics and sensitivity analyses for Gate C (docs/05_VALIDATION.md §2; docs/02_MATH.md §4,
§9; D-013).

- Input attribution: calibration of the pre-match input separately from the live layer.
- Substitution intensity: how strongly live signals drive substitutions (censoring diagnostic),
  and the stabilised IPCW weights built from it.
- Tipping point: how much higher the substituted players' hazard would have to be to erase each
  live effect.
- Aalen-Johansen cumulative incidence of non-contact and contact injury by match minute.

All fits reuse the MAP and Laplace machinery of ``model.live_hazard`` on purpose-built designs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from livrisk import riskmath
from livrisk.config import Config, config_hash
from livrisk.evaluation.bootstrap import percentile_interval, ratio_draws
from livrisk.evaluation.criteria import CI_LEVEL, OE_RANGE, TIPPING_LEVEL
from livrisk.evaluation.metrics import spell_table
from livrisk.evaluation.models import designs, fit_models, with_covariates
from livrisk.model.live_hazard import Design, FittedHazard, Priors, fit_map, predict_hazard

FloatArray = NDArray[np.float64]

#: Weak priors of the substitution-intensity model (near maximum likelihood on the log scale).
_DIAGNOSTIC_PRIORS = Priors(alpha_sd=10.0, theta_sd=10.0, eta_sd=0.5, beta_sd=10.0)
#: Upper percentile at which IPCW weights are truncated (docs/02_MATH.md §4).
_IPCW_TRUNCATE_PCT: float = 99.0
#: Regular match length for imputed minutes after a substitution.
_REGULAR_END_MIN: int = 90


def _z(level: float) -> float:
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def _beta_table(fit: FittedHazard, level: float) -> pd.DataFrame:
    """Coefficient, SD and interval of every live coefficient of a fit."""
    k0 = 1 + riskmath.N_BANDS
    sd = np.sqrt(np.clip(np.diag(fit.cov)[k0:], 0.0, None))
    z = _z(level)
    return pd.DataFrame(
        {
            "signal": list(fit.feature_names),
            "coef": fit.beta,
            "sd": sd,
            "hr": np.exp(fit.beta),
            "lo": fit.beta - z * sd,
            "hi": fit.beta + z * sd,
        }
    )


@dataclass(frozen=True)
class InputAttribution:
    """Calibration of the time-profile model M1 on test rows, before any live signal."""

    oe_m1: float
    oe_m1_ci: tuple[float, float]
    #: Slope on ln(Lambda) when it enters as a covariate instead of a fixed offset (ideal 1).
    input_slope: float
    input_slope_ci: tuple[float, float]
    verdict: str  # "input_ok" | "input_miscalibrated"


def input_attribution(
    train: pd.DataFrame, test: pd.DataFrame, cfg: Config, n_boot: int = 1000, seed: int = 0
) -> InputAttribution:
    """Fit M1 with ln(Lambda) as a covariate (prior N(1, input_slope_prior_sd^2)) on ``train``,
    report the slope and M1's O/E on ``test`` with player-cluster bootstrap CIs.

    The covariate is centred ln(Lambda) on top of the usual offset, so its coefficient is the
    slope minus 1 with a N(0, sd^2) prior. ``input_miscalibrated`` when the slope interval
    excludes 1 or M1's O/E lies outside the Gate C range (docs/02_MATH.md §9).
    """
    d1_train, _ = designs(train, (), cfg)
    ln_lam = d1_train.offset + math.log(riskmath.REGULAR_MINUTES)
    centred = (ln_lam - float(ln_lam.mean()))[:, None]
    priors = replace(Priors.from_config(cfg), beta_sd=float(cfg.evaluation.input_slope_prior_sd))
    fit = fit_map(
        with_covariates(d1_train, ["ln_lambda"], centred),
        priors,
        config_hash=config_hash(cfg),
        version="input-attribution",
    )
    sd = math.sqrt(max(float(fit.cov[-1, -1]), 0.0))
    z = _z(CI_LEVEL)
    slope = 1.0 + float(fit.beta[0])

    m1 = fit_models(train, (), cfg, "input-attribution").m1
    d1_test, _ = designs(test, (), cfg)
    spells = spell_table(test.assign(_m1=predict_hazard(m1, d1_test)), "_m1")
    point, draws = ratio_draws(
        spells["y"].to_numpy(dtype=np.float64),
        spells["_m1"].to_numpy(dtype=np.float64),
        spells["player_id"].astype(str).to_numpy(),
        n_boot,
        seed,
    )
    oe = percentile_interval(point, draws, CI_LEVEL)
    slope_ci = (slope - z * sd, slope + z * sd)
    bad = not slope_ci[0] <= 1.0 <= slope_ci[1] or not OE_RANGE[0] <= oe.point <= OE_RANGE[1]
    return InputAttribution(
        oe_m1=oe.point,
        oe_m1_ci=(oe.lo, oe.hi),
        input_slope=slope,
        input_slope_ci=slope_ci,
        verdict="input_miscalibrated" if bad else "input_ok",
    )


def _substitution_design(
    person_minutes: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> tuple[Design, NDArray[np.bool_]]:
    """Design of being substituted after minute j: rows of the bands that saw a substitution,
    no pre-match offset, label ``substituted_after``. Returns the design and its row mask."""
    sub = person_minutes["substituted_after"].to_numpy(dtype=np.bool_)
    band = person_minutes["band"].to_numpy(dtype=np.int64)
    rows = np.isin(band, np.unique(band[sub]))
    _, full = designs(person_minutes.loc[rows], feature_names, cfg)
    design = replace(full, offset=np.zeros(full.n_rows), y=sub[rows].astype(np.int64))
    return design, rows


def substitution_intensity(
    person_minutes: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> pd.DataFrame:
    """Discrete-time cloglog model of being substituted in minute j given the same live signals
    and band; returns coefficient, SD and hazard ratio per signal.

    Fitted on the bands where substitutions occur, with weak priors (a diagnostic close to
    maximum likelihood). Columns: signal, coef, sd, hr, lo, hi (95 % Laplace interval).
    """
    design, _ = _substitution_design(person_minutes, feature_names, cfg)
    fit = fit_map(design, _DIAGNOSTIC_PRIORS, config_hash=config_hash(cfg), version="subst")
    return _beta_table(fit, CI_LEVEL)


def ipcw_weights(
    person_minutes: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> FloatArray:
    """Stabilised inverse-probability-of-censoring weights per row (docs/02_MATH.md §4).

    w = P(not yet substituted | band) / P(not yet substituted | band, live signals), both from
    the substitution-intensity model over the minutes before the row in its player-match,
    truncated at the 99th percentile. Rows outside the substitution bands have no hazard of
    their own but carry the weight accumulated before them.
    """
    design, rows = _substitution_design(person_minutes, feature_names, cfg)
    h = config_hash(cfg)
    den = fit_map(design, _DIAGNOSTIC_PRIORS, config_hash=h, version="ipcw-den")
    bare = with_covariates(design, (), design.x[:, :0])
    num = fit_map(bare, _DIAGNOSTIC_PRIORS, config_hash=h, version="ipcw-num")
    # P(no substitution in a minute) = exp(-mu), so the log weight is a sum of hazard gaps.
    gap = np.zeros(person_minutes.shape[0], dtype=np.float64)
    gap[rows] = predict_hazard(den, design) - predict_hazard(num, bare)
    order = np.lexsort(
        (
            person_minutes["t_min"].to_numpy(dtype=np.float64),
            person_minutes["period"].to_numpy(dtype=np.int64),
            person_minutes["player_id"].astype(str).to_numpy(),
            person_minutes["match_id"].astype(str).to_numpy(),
        )
    )
    key = (
        person_minutes["match_id"].astype(str) + "\0" + person_minutes["player_id"].astype(str)
    ).to_numpy()[order]
    g = gap[order]
    before = pd.Series(g).groupby(key, sort=False).cumsum().to_numpy() - g
    w = np.empty_like(g)
    w[order] = np.exp(before)
    cap = float(np.percentile(w, _IPCW_TRUNCATE_PCT))
    return np.asarray(np.minimum(w, cap), dtype=np.float64)


def ipcw_fit(
    person_minutes: pd.DataFrame, feature_names: Sequence[str], cfg: Config
) -> pd.DataFrame:
    """Live coefficients of M2 refitted with IPCW weights (sensitivity), 95 % intervals."""
    weights = ipcw_weights(person_minutes, feature_names, cfg)
    fit = fit_models(person_minutes, feature_names, cfg, "ipcw", sample_weight=weights).m2
    return _beta_table(fit, CI_LEVEL)


def _imputed_minutes(person_minutes: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Minutes after each substitution up to the end of regular time, live state usual (z = 0).

    Rows carry the player-match's p7 under a separate match key, so they never sit behind the
    substitution of their own player-match.
    """
    last = person_minutes.loc[person_minutes["substituted_after"].to_numpy(dtype=np.bool_)]
    rows: list[dict[str, object]] = []
    for rec in last.to_dict("records"):
        for t in range(int(rec["t_min"]) + 1, _REGULAR_END_MIN):
            period = 1 if t < 45 else 2
            row: dict[str, object] = {
                "match_id": f"{rec['match_id']}+imputed",
                "player_id": str(rec["player_id"]),
                "period": period,
                "t_min": float(t),
                "band": riskmath.band_index(float(t), period),
                "p7": float(rec["p7"]),
                "y": 0,
                "contact": False,
                "substituted_after": False,
            }
            row.update({f: 0.0 for f in feature_names})
            rows.append(row)
    columns = [
        "match_id",
        "player_id",
        "period",
        "t_min",
        "band",
        "p7",
        "y",
        "contact",
        "substituted_after",
        *feature_names,
    ]
    return pd.DataFrame(rows, columns=columns)


def tipping_point(
    person_minutes: pd.DataFrame,
    feature_names: Sequence[str],
    cfg: Config,
    seed: int,
    multipliers: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0),
) -> pd.DataFrame:
    """For each multiplier psi, impute the substituted players' remaining minutes with hazard
    psi times the model hazard, refit, and report each beta with its interval. The tipping point
    of a signal is the smallest psi at which its interval includes 0.

    The remaining minutes run to the end of regular time with the live state unknown, taken as
    his usual state (z = 0): the scenario is a removal for reasons the data do not contain. One
    seeded imputation per psi; events end the imputed minutes. Columns: psi, signal, coef, lo,
    hi (90 % Laplace interval), includes_zero.
    """
    names = list(feature_names)
    base = fit_models(person_minutes, names, cfg, "tipping-base").m2
    imputed = _imputed_minutes(person_minutes, names)
    observed = person_minutes[[*imputed.columns]].copy()
    observed["match_id"] = observed["match_id"].astype(str)
    observed["player_id"] = observed["player_id"].astype(str)
    tables: list[pd.DataFrame] = []
    _, d_imp = designs(imputed, names, cfg)
    mu_usual = predict_hazard(base, d_imp)
    keys = (imputed["match_id"] + "\0" + imputed["player_id"]).to_numpy()
    for i, psi in enumerate(multipliers):
        rng = np.random.default_rng([seed, i])
        p = riskmath.inv_cloglog(np.log(psi * mu_usual))
        hit = rng.random(p.shape[0]) < p
        hit_series = pd.Series(hit.astype(np.int64))
        before = hit_series.groupby(keys, sort=False).cumsum().to_numpy() - hit.astype(np.int64)
        keep = before == 0
        rows = imputed.loc[keep].copy()
        rows["y"] = hit[keep].astype(np.int64)
        augmented = pd.concat([observed, rows], ignore_index=True)
        fit = fit_models(augmented, names, cfg, f"tipping-{psi:g}").m2
        table = _beta_table(fit, TIPPING_LEVEL)[["signal", "coef", "lo", "hi"]]
        table.insert(0, "psi", float(psi))
        table["includes_zero"] = (table["lo"] <= 0.0) & (table["hi"] >= 0.0)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def tipping_points(table: pd.DataFrame) -> dict[str, float | None]:
    """Smallest psi whose interval includes 0, per signal (None when no psi erases it)."""
    out: dict[str, float | None] = {}
    for signal, g in table.groupby("signal", sort=False):
        hit = g.loc[g["includes_zero"], "psi"]
        out[str(signal)] = float(hit.min()) if hit.shape[0] else None
    return out


def aalen_johansen(person_minutes: pd.DataFrame, hazard_col: str | None = None) -> pd.DataFrame:
    """Cumulative incidence by match minute of non-contact and contact injury, with substitution
    and match end as censoring. Columns: t_min, cif_noncontact, cif_contact.

    Also period and n_at_risk; with ``hazard_col`` (per-minute model hazards) the incidence the
    model implies at the observed risk sets, cif_noncontact_model. Minutes are ordered by
    period, then minute.
    """
    pm = person_minutes
    work = pd.DataFrame(
        {
            "period": pm["period"].to_numpy(dtype=np.int64),
            "t_min": pm["t_min"].to_numpy(dtype=np.float64),
            "y": pm["y"].to_numpy(dtype=np.float64),
            "c": pm["contact"].to_numpy(dtype=np.float64),
        }
    )
    if hazard_col is not None:
        mu = pm[hazard_col].to_numpy(dtype=np.float64)
        work["p"] = riskmath.inv_cloglog(np.log(mu))
    g = work.groupby(["period", "t_min"], sort=True)
    agg = g.agg(n=("y", "size"), d1=("y", "sum"), d2=("c", "sum")).reset_index()
    n = agg["n"].to_numpy(dtype=np.float64)
    d1 = agg["d1"].to_numpy(dtype=np.float64)
    d2 = agg["d2"].to_numpy(dtype=np.float64)
    surv = np.cumprod(1.0 - (d1 + d2) / n)
    surv_before = np.concatenate([[1.0], surv[:-1]])
    out = pd.DataFrame(
        {
            "period": agg["period"],
            "t_min": agg["t_min"],
            "n_at_risk": agg["n"],
            "cif_noncontact": np.cumsum(surv_before * d1 / n),
            "cif_contact": np.cumsum(surv_before * d2 / n),
        }
    )
    if hazard_col is not None:
        mean_p = g["p"].mean().to_numpy(dtype=np.float64)
        out["cif_noncontact_model"] = np.cumsum(surv_before * mean_p)
    return out
