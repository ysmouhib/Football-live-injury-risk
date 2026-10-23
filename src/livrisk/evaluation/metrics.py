"""Evaluation metrics (docs/05_VALIDATION.md §2; docs/02_MATH.md §9).

Everything is scored at spell level: 15-minute blocks of a player-match counted from each
period's start; stoppage minutes form their own spell. A spell's label is 1 when a non-contact
injury starts in it. Its probability is 1 - exp(-sum of per-minute hazards) over the minutes the
player would have been at risk in it: the observed minutes plus, in a spell that ends with the
onset, the rest of the spell at the onset minute's hazard. Without that extension the spell
holding an injury would get a lower probability because the injury happened. Exposure that
stops at a substitution, a contact injury or the end of the data is not extended (censoring).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

from livrisk import riskmath

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

#: Match minute at which each period starts, and at which its regular time ends (index = period).
_PERIOD_START_MIN = np.asarray([0.0, 0.0, 45.0, 90.0, 105.0])
_REGULAR_END_MIN = np.asarray([0.0, 45.0, 90.0, 105.0, 120.0])
#: Columns carried from the minutes to their spell when present.
_CARRY: tuple[str, ...] = ("club_id", "season", "team_id", "position")
#: Probabilities are clipped to this distance from 0 and 1 inside the scores.
_P_EPS: float = 1e-15
_KEYS: list[str] = ["match_id", "player_id", "period", "spell"]


def _spell_keys(person_minutes: pd.DataFrame, spell_min: int) -> pd.DataFrame:
    period = person_minutes["period"].to_numpy(dtype=np.int64)
    t_min = person_minutes["t_min"].to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "match_id": person_minutes["match_id"].astype(str).to_numpy(),
            "player_id": person_minutes["player_id"].astype(str).to_numpy(),
            "period": period,
            "spell": ((t_min - _PERIOD_START_MIN[period]) // spell_min).astype(np.int64),
        }
    )


def spell_index(person_minutes: pd.DataFrame, spell_min: int = 15) -> IntArray:
    """Spell number of every minute row: the row of ``spell_table`` it belongs to."""
    keys = _spell_keys(person_minutes, spell_min)
    return np.asarray(keys.groupby(_KEYS, sort=True).ngroup(), dtype=np.int64)


def spell_table(
    person_minutes: pd.DataFrame,
    pred_col: str | Sequence[str],
    spell_min: int = 15,
    mean_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate minute-level hazards to spells: y_spell, p_spell = 1 - exp(-sum lambda).

    ``pred_col`` names one or more columns of per-minute hazards (lambda * Delta); see the
    module docstring for the exposure of a spell that ends with the onset. Columns: match_id,
    player_id, period, spell (index within the period), t0_min (first minute), n_min (observed
    minutes), y, onset_t_min (NaN when none), the carried columns (club_id, season, team_id,
    position) when present, the spell means of ``mean_cols``, then one spell probability per
    prediction column under the same name. Rows are ordered like ``spell_index``.
    """
    cols = [pred_col] if isinstance(pred_col, str) else list(pred_col)
    pm = person_minutes
    keys = _spell_keys(pm, spell_min)
    gid = np.asarray(keys.groupby(_KEYS, sort=True).ngroup(), dtype=np.int64)
    n = int(np.max(gid)) + 1 if gid.size else 0
    order = np.argsort(gid, kind="stable")
    first = order[np.searchsorted(gid[order], np.arange(n))]
    out = keys.iloc[first].reset_index(drop=True)
    t_min = pm["t_min"].to_numpy(dtype=np.float64)
    y = pm["y"].to_numpy(dtype=np.float64)
    count = np.bincount(gid, minlength=n).astype(np.float64)
    t0 = np.full(n, np.inf)
    np.minimum.at(t0, gid, t_min)
    event = y > 0.5
    onset = np.full(n, np.nan)
    onset[gid[event]] = t_min[event]
    out["t0_min"] = t0
    out["n_min"] = count.astype(np.int64)
    out["y"] = (np.bincount(gid, weights=y, minlength=n) > 0.5).astype(np.int64)
    out["onset_t_min"] = onset
    for c in _CARRY:
        if c in pm.columns:
            out[c] = pm[c].to_numpy()[first]
    for c in mean_cols:
        out[c] = np.bincount(gid, weights=pm[c].to_numpy(dtype=np.float64), minlength=n) / count

    period = out["period"].to_numpy(dtype=np.int64)
    start = _PERIOD_START_MIN[period] + out["spell"].to_numpy(dtype=np.float64) * spell_min
    regular_end = _REGULAR_END_MIN[period]
    period_last = (
        pd.Series(t_min).groupby([keys["match_id"], keys["period"]]).transform("max").to_numpy()
    )
    last = np.full(n, -np.inf)
    np.maximum.at(last, gid, period_last)
    planned_end = np.where(
        start < regular_end, np.minimum(start + spell_min, regular_end), last + 1.0
    )
    remaining = np.where(out["y"].to_numpy() == 1, np.maximum(planned_end - onset - 1.0, 0.0), 0.0)
    for c in cols:
        h = pm[c].to_numpy(dtype=np.float64)
        total = np.bincount(gid, weights=h, minlength=n)
        total = total + remaining * np.bincount(gid, weights=h * y, minlength=n)
        with np.errstate(divide="ignore"):
            out[c] = riskmath.inv_cloglog(np.log(total))
    return out


def _arrays(y: ArrayLike, p: ArrayLike) -> tuple[FloatArray, FloatArray]:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.clip(np.asarray(p, dtype=np.float64), _P_EPS, 1.0 - _P_EPS)
    if yy.shape != pp.shape or yy.ndim != 1:
        raise ValueError("y and p must be 1-D arrays of the same length")
    return yy, pp


def log_loss_terms(y: ArrayLike, p: ArrayLike) -> FloatArray:
    """Per-row log loss -[y ln p + (1 - y) ln(1 - p)] (natural log)."""
    yy, pp = _arrays(y, p)
    return np.asarray(-(yy * np.log(pp) + (1.0 - yy) * np.log1p(-pp)), dtype=np.float64)


def log_loss(y: ArrayLike, p: ArrayLike) -> float:
    """Mean log loss (natural log)."""
    return float(np.mean(log_loss_terms(y, p)))


def brier(y: ArrayLike, p: ArrayLike) -> float:
    """Mean squared difference between probability and outcome."""
    yy, pp = _arrays(y, p)
    return float(np.mean((pp - yy) ** 2))


def pr_auc(y: ArrayLike, p: ArrayLike) -> float:
    """Area under the precision-recall curve (average precision); NaN without positives."""
    yy, pp = _arrays(y, p)
    if not np.any(yy > 0.5):
        return float("nan")
    return float(average_precision_score(yy, pp))


def calibration_slope_intercept(y: ArrayLike, p: ArrayLike) -> tuple[float, float]:
    """Fit cloglog(P(y=1)) = c0 + c1 * cloglog(p); return (c1 slope, c0 intercept).

    Maximum likelihood with the cloglog terms of ``riskmath``; ideal slope 1, intercept 0.
    """
    yy, pp = _arrays(y, p)
    x = riskmath.cloglog(pp)

    def nll(c: FloatArray) -> float:
        return -float(np.sum(riskmath.cloglog_loglik_terms(c[0] + c[1] * x, yy)))

    def grad(c: FloatArray) -> FloatArray:
        s = riskmath.cloglog_score_terms(c[0] + c[1] * x, yy)
        return np.asarray([-float(s.sum()), -float(s @ x)], dtype=np.float64)

    res = minimize(nll, np.asarray([0.0, 1.0]), jac=grad, method="BFGS")
    if not res.success and not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"calibration fit failed: {res.message}")
    return float(res.x[1]), float(res.x[0])


def oe_table(spells: pd.DataFrame, pred_col: str, by: str) -> pd.DataFrame:
    """Observed / expected per group (docs/02 §9): columns by, n_spells, observed, expected, oe."""
    rows: list[dict[str, object]] = []
    for key, g in spells.groupby(by, sort=True, observed=True):
        y = g["y"].to_numpy(dtype=np.float64)
        p = g[pred_col].to_numpy(dtype=np.float64)
        rows.append(
            {
                by: key,
                "n_spells": int(g.shape[0]),
                "observed": float(y.sum()),
                "expected": float(p.sum()),
                "oe": riskmath.observed_expected_ratio(y, p),
            }
        )
    return pd.DataFrame(rows, columns=[by, "n_spells", "observed", "expected", "oe"])


def oe_by_decile(spells: pd.DataFrame, pred_col: str, n_groups: int = 10) -> pd.DataFrame:
    """O/E per group of predicted risk (deciles by rank); column ``decile`` from 1 (lowest)."""
    rank = spells[pred_col].rank(method="first").to_numpy(dtype=np.float64)
    decile = np.ceil(rank / spells.shape[0] * n_groups).astype(np.int64)
    work = spells.assign(decile=np.clip(decile, 1, n_groups))
    return oe_table(work, pred_col, "decile")


def flag_top_share(score: ArrayLike, share: float) -> BoolArray:
    """Flags for the ``share`` of rows with the highest scores (ties keep row order)."""
    s = np.asarray(score, dtype=np.float64)
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be in [0, 1]")
    k = round(share * s.size)
    flags = np.zeros(s.size, dtype=np.bool_)
    if k:
        flags[np.argsort(-s, kind="stable")[:k]] = True
    return flags


@dataclass(frozen=True)
class AlertMetrics:
    """Spell-level alert metrics for one flag rule (docs/05 §2, docs/02 §7)."""

    flag_share: float
    alerts_per_team_match: float
    n_spells: int
    n_flagged: int
    n_injuries: int
    n_caught: int
    recall: float
    precision: float
    enrichment: float
    median_lead_min: float


def alert_metrics(spells: pd.DataFrame, flags: ArrayLike) -> AlertMetrics:
    """Metrics of flagged spells: an injury is caught when it starts in a flagged spell (the
    flag comes from the risk at the spell start, so it precedes the onset by up to 15 min).
    Alerts per team-match = flag share x 66 (``riskmath.alerts_per_match``)."""
    f = np.asarray(flags, dtype=np.bool_)
    y = spells["y"].to_numpy(dtype=np.int64) == 1
    if f.shape != y.shape:
        raise ValueError("one flag per spell required")
    n, n_flag, n_inj = int(y.size), int(f.sum()), int(y.sum())
    caught = f & y
    share = n_flag / n if n else 0.0
    recall = float(caught.sum()) / n_inj if n_inj else float("nan")
    lead = (spells["onset_t_min"].to_numpy(dtype=np.float64) - spells["t0_min"].to_numpy())[caught]
    return AlertMetrics(
        flag_share=share,
        alerts_per_team_match=riskmath.alerts_per_match(share) if n else 0.0,
        n_spells=n,
        n_flagged=n_flag,
        n_injuries=n_inj,
        n_caught=int(caught.sum()),
        recall=recall,
        precision=float(caught.sum()) / n_flag if n_flag else float("nan"),
        enrichment=recall / share if share > 0.0 and n_inj else float("nan"),
        median_lead_min=float(np.median(lead)) if lead.size else float("nan"),
    )
