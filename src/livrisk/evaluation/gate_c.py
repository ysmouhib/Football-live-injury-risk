"""Gate C runner, report and model card (docs/05_VALIDATION.md §2, §5, §6).

``run_gate_c`` fits the nested comparators M0, M1 and M2 (and the benchmarks and exploratory
comparators) inside each training fold, scores the pooled test folds at spell level and checks
the four pre-registered criteria. Training rows pass the training-data policy (docs/02 §6)
first; a ``prepare`` hook recomputes fold-dependent inputs (norms, pre-match recalibration)
from the training rows of each fold only. If any criterion fails, the report recommends
``engine.mode: monitor_only``.

There is one primary hypothesis (M2 vs M1 spell-level log loss, D-014). Exploratory comparators
are Holm-adjusted, reported apart and never adopted on the data that suggested them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from livrisk import riskmath
from livrisk.config import Config, config_hash
from livrisk.evaluation.benchmarks import BENCHMARKS, EXPLORATORY, NOT_RUNNABLE, Comparator
from livrisk.evaluation.bootstrap import (
    Interval,
    bootstrap_p_value,
    mean_interval,
    percentile_interval,
    ratio_draws,
)
from livrisk.evaluation.criteria import (
    ALERT_BUDGET_PER_TEAM_MATCH,
    CI_LEVEL,
    OE_RANGE,
    SUBGROUP_OE_RANGE,
)
from livrisk.evaluation.decision_curves import decision_curve
from livrisk.evaluation.metrics import (
    AlertMetrics,
    alert_metrics,
    brier,
    calibration_slope_intercept,
    flag_top_share,
    log_loss,
    log_loss_terms,
    oe_by_decile,
    oe_table,
    pr_auc,
    spell_table,
)
from livrisk.evaluation.models import MODELS, ModelFits, fit_models, predict_models
from livrisk.evaluation.sensitivity import (
    InputAttribution,
    aalen_johansen,
    input_attribution,
    ipcw_fit,
    substitution_intensity,
    tipping_point,
    tipping_points,
)
from livrisk.evaluation.splits import TrainingRows, make_splits, training_policy
from livrisk.model.live_hazard import FittedHazard

FoldPrep = Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]

MODEL_LABEL: dict[str, str] = {
    "m0": "M0 pre-match only",
    "m1": "M1 pre-match x time profile",
    "m2": "M2 M1 x live multiplier",
}


def standardize_in_fold(feature_names: Sequence[str]) -> FoldPrep:
    """A ``prepare`` hook that re-standardises each feature with the training fold's mean and
    SD, so a population norm is computed from training rows only."""
    names = list(feature_names)

    def prepare(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = train[names].to_numpy(dtype=np.float64)
        mean, sd = x.mean(axis=0), x.std(axis=0)
        sd = np.where(sd > 0.0, sd, 1.0)
        train = train.copy()
        test = test.copy()
        train[names] = (x - mean) / sd
        test[names] = (test[names].to_numpy(dtype=np.float64) - mean) / sd
        return train, test

    return prepare


@dataclass(frozen=True, eq=False)
class CrossFit:
    """Pooled test rows with one per-minute hazard column per model and comparator."""

    rows: pd.DataFrame
    folds: tuple[str, ...]
    fits: tuple[ModelFits, ...]
    training: TrainingRows
    comparators: tuple[str, ...]


def cross_fit(
    person_minutes: pd.DataFrame,
    feature_names: Sequence[str],
    cfg: Config,
    split: str = "time",
    *,
    prepare: FoldPrep | None = None,
    comparators: Mapping[str, Comparator] | None = None,
) -> CrossFit:
    """Fit M1 and M2 (and ``comparators``) on each training fold; score its test fold.

    Training rows are the fold's rows allowed by ``splits.training_policy``; test rows are
    every row of the test fold. ``prepare(train, test)`` runs per fold before fitting and sees
    only that fold's rows. Hazard columns: m0, m1, m2 and one per comparator; ``fold`` names
    the fold of each test row.
    """
    names = list(feature_names)
    policy = training_policy(person_minutes, cfg)
    parts: list[pd.DataFrame] = []
    fits: list[ModelFits] = []
    folds = make_splits(person_minutes, split)
    for fold in folds:
        train = person_minutes.loc[fold.train & policy.keep].copy()
        test = person_minutes.loc[fold.test].copy()
        if prepare is not None:
            train, test = prepare(train, test)
        fit = fit_models(train, names, cfg, f"gate-c {fold.name}")
        preds = predict_models(fit, test, names, cfg)
        for cname, comp in (comparators or {}).items():
            preds[cname] = comp(train, test, names, cfg)
        for col, values in preds.items():
            test[col] = values
        test["fold"] = fold.name
        parts.append(test)
        fits.append(fit)
    rows = pd.concat(parts, ignore_index=True)
    for col in ("match_id", "player_id", "club_id", "season"):
        if col in rows.columns:
            rows[col] = rows[col].astype(str)
    return CrossFit(
        rows=rows,
        folds=tuple(f.name for f in folds),
        fits=tuple(fits),
        training=policy,
        comparators=tuple(comparators or ()),
    )


@dataclass(frozen=True)
class PrimaryResult:
    """M2 vs M1 spell-level log loss (D-014): M1 minus M2, nats per spell; > 0 favours M2."""

    improvement: Interval
    passed: bool


def _clusters(spells: pd.DataFrame) -> np.ndarray:
    return spells["player_id"].astype(str).to_numpy()


def primary_test(spells: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> PrimaryResult:
    """Player-cluster bootstrap of the log-loss improvement of M2 over M1; passes when the 95 %
    interval excludes 0 on the side of M2."""
    y = spells["y"].to_numpy(dtype=np.float64)
    gain = log_loss_terms(y, spells["m1"]) - log_loss_terms(y, spells["m2"])
    interval, _ = mean_interval(gain, _clusters(spells), n_boot, seed, CI_LEVEL)
    return PrimaryResult(improvement=interval, passed=interval.lo > 0.0)


@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PreRegistration:
    """Fields of the pre-registration template (docs/05_VALIDATION.md §5)."""

    model_version: str
    config_hash: str
    data: str
    primary: str
    thresholds: str
    sensitivity: str
    exploratory: str
    training_data: str
    reported_regardless: str

    @classmethod
    def default(
        cls, cfg: Config, person_minutes: pd.DataFrame, model_version: str = "not stated"
    ) -> PreRegistration:
        """Template filled from the config and the data; free-text fields state the plan."""
        seasons = sorted(person_minutes["season"].astype(str).unique().tolist())
        clubs = person_minutes["club_id"].astype(str).nunique()
        return cls(
            model_version=model_version,
            config_hash=config_hash(cfg),
            data=(
                f"seasons {', '.join(seasons)}; {clubs} clubs; non-contact time-loss injuries "
                f">= 7 days; delta {cfg.engine.blanking_delta_s} s"
            ),
            primary="M2 vs M1, spell-level (15 min) log loss, player-cluster bootstrap 95 % CI",
            thresholds=(
                f"O/E in [{OE_RANGE[0]}, {OE_RANGE[1]}] per test season; alerts <= "
                f"{ALERT_BUDGET_PER_TEAM_MATCH} per team-match with caught share above flagged "
                f"share; subgroup O/E in [{SUBGROUP_OE_RANGE[0]}, {SUBGROUP_OE_RANGE[1]}]"
            ),
            sensitivity="IPCW for substitutions; tipping point; input attribution; Aalen-Johansen",
            exploratory=", ".join(cfg.evaluation.exploratory) + " (Holm-adjusted)",
            training_data=(
                f"sources {', '.join(cfg.evaluation.training_sources)}; live-mode rows censored "
                "at the first shown alert"
            ),
            reported_regardless="every criterion, metric, diagnostic and comparator below",
        )


@dataclass(frozen=True, eq=False)
class Diagnostics:
    """Diagnostics reported with every run (docs/05_VALIDATION.md §2)."""

    input_attribution: InputAttribution
    substitution_intensity: pd.DataFrame
    ipcw: pd.DataFrame
    tipping: pd.DataFrame
    tipping_points: dict[str, float | None]
    aalen_johansen: pd.DataFrame


@dataclass(frozen=True, eq=False)
class GateCReport:
    """Everything the Gate C run measured; ``to_markdown`` renders the report."""

    criteria: tuple[CriterionResult, ...]
    primary: PrimaryResult | None = None
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    oe: dict[str, pd.DataFrame] = field(default_factory=dict)
    alerts: AlertMetrics | None = None
    alerts_gap: Interval | None = None
    decision: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmarks: pd.DataFrame = field(default_factory=pd.DataFrame)
    exploratory: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: Diagnostics | None = None
    training: TrainingRows | None = None
    preregistration: PreRegistration | None = None
    folds: tuple[str, ...] = ()
    n_spells: int = 0
    n_events: int = 0
    final_fit: FittedHazard | None = None
    feature_names: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.criteria)

    @property
    def mode(self) -> str:
        """``engine.mode`` this report supports: risk only when every criterion holds."""
        return "risk" if self.passed else "monitor_only"

    def to_markdown(self) -> str:
        return _render_report(self)

    def model_card(self) -> str:
        return _render_model_card(self)


def _oe_criterion(table: pd.DataFrame, by: str, lo: float, hi: float, name: str) -> CriterionResult:
    values = table["oe"].to_numpy(dtype=np.float64)
    ok = bool(np.all((values >= lo) & (values <= hi)))
    detail = "; ".join(f"{k}: {v:.2f}" for k, v in zip(table[by], values, strict=True))
    return CriterionResult(name, ok, detail or "no groups")


def _improvement(
    spells: pd.DataFrame, base: str, other: str, n_boot: int, seed: int
) -> tuple[Interval, float]:
    """Log-loss improvement of ``other`` over ``base`` with its two-sided bootstrap p-value."""
    y = spells["y"].to_numpy(dtype=np.float64)
    gain = log_loss_terms(y, spells[base]) - log_loss_terms(y, spells[other])
    interval, draws = mean_interval(gain, _clusters(spells), n_boot, seed, CI_LEVEL)
    return interval, bootstrap_p_value(draws)


def _metrics_table(spells: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    y = spells["y"].to_numpy(dtype=np.float64)
    rows = []
    for col in cols:
        p = spells[col].to_numpy(dtype=np.float64)
        slope, intercept = calibration_slope_intercept(y, p)
        rows.append(
            {
                "model": col,
                "log_loss": log_loss(y, p),
                "brier": brier(y, p),
                "pr_auc": pr_auc(y, p),
                "oe": riskmath.observed_expected_ratio(y, p),
                "cal_slope": slope,
                "cal_intercept": intercept,
            }
        )
    return pd.DataFrame(rows)


def run_gate_c(
    person_minutes: pd.DataFrame,
    feature_names: Sequence[str],
    cfg: Config,
    split: str = "time",
    seed: int = 0,
    *,
    prepare: FoldPrep | None = None,
    preregistration: PreRegistration | None = None,
    n_boot: int = 1000,
    diagnostics: bool = True,
    benchmarks: bool = True,
) -> GateCReport:
    """Fit M0/M1/M2 (+ benchmarks) inside each training fold; evaluate on test folds.

    ``person_minutes`` has the schema of ``synth.generate_person_minutes`` (season and club_id
    for the splits; optional source, first_alert_t_min, team_id, position). Seeds: ``seed``
    drives every bootstrap and the tipping-point imputation.
    """
    names = list(feature_names)
    runnable: dict[str, Comparator] = {}
    status: dict[str, str] = {}
    for eid in cfg.evaluation.exploratory:
        if eid in EXPLORATORY:
            comp, needs = EXPLORATORY[eid]
            missing = [c for c in needs if c not in person_minutes.columns]
            if missing:
                status[eid] = f"not run: needs column {', '.join(missing)}"
            else:
                runnable[f"x_{eid}"] = comp
                status[eid] = "run"
        else:
            status[eid] = f"not run: {NOT_RUNNABLE.get(eid, 'no comparator for this id')}"
    comps: dict[str, Comparator] = dict(runnable)
    if benchmarks:
        comps.update({f"b_{k}": v for k, v in BENCHMARKS.items()})
    cross = cross_fit(person_minutes, names, cfg, split, prepare=prepare, comparators=comps)
    pred_cols = [*MODELS, *comps]
    spells = spell_table(cross.rows, pred_cols)

    primary = primary_test(spells, n_boot, seed)
    oe_tables = {
        "season": oe_table(spells, "m2", "season"),
        "club": oe_table(spells, "m2", "club_id"),
        "decile": oe_by_decile(spells, "m2"),
    }
    if "position" in spells.columns:
        oe_tables["position"] = oe_table(spells, "m2", "position")

    share = ALERT_BUDGET_PER_TEAM_MATCH / riskmath.PLAYER_SPELLS_PER_TEAM_MATCH
    score = spells["m2"].to_numpy(dtype=np.float64) / spells["m1"].to_numpy(dtype=np.float64)
    flags = flag_top_share(score, share)
    alerts = alert_metrics(spells, flags)
    y = spells["y"].to_numpy(dtype=np.float64)
    clusters = _clusters(spells)
    recall, recall_draws = ratio_draws(flags & (y > 0.5), y, clusters, n_boot, seed)
    flag_share, share_draws = ratio_draws(flags, np.ones_like(y), clusters, n_boot, seed)
    gap = percentile_interval(recall - flag_share, recall_draws - share_draws, CI_LEVEL)

    sub_tables = [(oe_tables["club"], "club_id")]
    if "position" in oe_tables:
        sub_tables.append((oe_tables["position"], "position"))
    sub_ok = [_oe_criterion(t, by, *SUBGROUP_OE_RANGE, f"subgroup {by}") for t, by in sub_tables]
    criteria = (
        CriterionResult(
            "1. M2 improves log loss over M1, CI excludes 0",
            primary.passed,
            f"improvement {primary.improvement.point:.6f} nats per spell, 95 % CI "
            f"{primary.improvement.lo:.6f} to {primary.improvement.hi:.6f}",
        ),
        _oe_criterion(
            oe_tables["season"], "season", *OE_RANGE, "2. O/E in [0.8, 1.25] in every test season"
        ),
        CriterionResult(
            f"3. At <= {ALERT_BUDGET_PER_TEAM_MATCH} alerts per team-match, caught share above "
            "flagged share, CI excludes equality",
            gap.lo > 0.0,
            f"caught {alerts.recall:.3f} vs flagged {alerts.flag_share:.4f}; difference 95 % CI "
            f"{gap.lo:.3f} to {gap.hi:.3f}; {alerts.alerts_per_team_match:.2f} alerts per "
            "team-match",
        ),
        CriterionResult(
            "4. No subgroup (position, club) with O/E outside 0.6-1.6",
            all(c.passed for c in sub_ok),
            " | ".join(f"{c.name}: {c.detail}" for c in sub_ok),
        ),
    )

    bench_rows = []
    for col in comps:
        if not col.startswith("b_"):
            continue
        interval, _ = _improvement(spells, "m1", col, n_boot, seed)
        bench_rows.append(
            {
                "benchmark": col[2:],
                "log_loss": log_loss(y, spells[col]),
                "improvement_vs_m1": interval.point,
                "lo": interval.lo,
                "hi": interval.hi,
            }
        )
    explore_rows = []
    for eid in cfg.evaluation.exploratory:
        col = f"x_{eid}"
        if col in comps:
            interval, p = _improvement(spells, "m2", col, n_boot, seed)
            explore_rows.append(
                {
                    "comparator": eid,
                    "status": "run",
                    "improvement_vs_m2": interval.point,
                    "lo": interval.lo,
                    "hi": interval.hi,
                    "p_raw": p,
                }
            )
        else:
            explore_rows.append({"comparator": eid, "status": status[eid]})
    explore = pd.DataFrame(
        explore_rows,
        columns=["comparator", "status", "improvement_vs_m2", "lo", "hi", "p_raw", "p_holm"],
    )
    ran = explore["status"] == "run"
    if bool(ran.any()):
        explore.loc[ran, "p_holm"] = riskmath.holm_adjust(
            explore.loc[ran, "p_raw"].to_numpy(dtype=np.float64)
        )

    # The final fit and the diagnostics see the same preparation as each fold's training rows.
    train_all = person_minutes.loc[cross.training.keep]
    if prepare is not None:
        train_all, _ = prepare(train_all.copy(), train_all.iloc[:0].copy())
    final = fit_models(train_all, names, cfg, "gate-c all training rows").m2
    diag = None
    if diagnostics:
        last = make_splits(person_minutes, split)[-1]
        ia_train = person_minutes.loc[last.train & cross.training.keep].copy()
        ia_test = person_minutes.loc[last.test].copy()
        if prepare is not None:
            ia_train, ia_test = prepare(ia_train, ia_test)
        tip = tipping_point(train_all, names, cfg, seed)
        diag = Diagnostics(
            input_attribution=input_attribution(ia_train, ia_test, cfg, n_boot, seed),
            substitution_intensity=substitution_intensity(train_all, names, cfg),
            ipcw=ipcw_fit(train_all, names, cfg),
            tipping=tip,
            tipping_points=tipping_points(tip),
            aalen_johansen=aalen_johansen(cross.rows, "m2"),
        )
    return GateCReport(
        criteria=criteria,
        primary=primary,
        metrics=_metrics_table(spells, pred_cols),
        oe=oe_tables,
        alerts=alerts,
        alerts_gap=gap,
        decision=decision_curve(spells, list(MODELS)),
        benchmarks=pd.DataFrame(bench_rows),
        exploratory=explore,
        diagnostics=diag,
        training=cross.training,
        preregistration=preregistration or PreRegistration.default(cfg, person_minutes),
        folds=cross.folds,
        n_spells=int(spells.shape[0]),
        n_events=int(y.sum()),
        final_fit=final,
        feature_names=tuple(names),
    )


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    if isinstance(value, float | np.floating):
        v = float(value)
        if math.isnan(v):
            return "-"
        return f"{v:.{digits}g}" if abs(v) < 1e-3 and v != 0.0 else f"{v:.{digits}f}"
    return str(value)


def _md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.shape[0] == 0:
        return "_none_\n"
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for rec in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in rec) + " |")
    return "\n".join(lines) + "\n"


def _render_report(r: GateCReport) -> str:
    out: list[str] = ["# Gate C report", ""]
    if r.passed:
        out.append(
            "**Verdict: all criteria hold.** The live layer may run with `engine.mode: risk`."
        )
    else:
        out.append(
            "**Verdict: at least one criterion fails.** Run with `engine.mode: monitor_only`: "
            "pre-match risk plus load against the player's own norm, with no live risk claims."
        )
    out += [
        "",
        f"Folds: {', '.join(r.folds)}. Test spells: {r.n_spells}; events: {r.n_events}.",
        "",
    ]
    if r.preregistration is not None:
        pre = r.preregistration
        out += ["## Pre-registration", ""]
        out.append(
            _md_table(
                pd.DataFrame(
                    {
                        "field": [
                            "Model version / commit",
                            "Config hash",
                            "Data",
                            "Primary comparison and metric",
                            "Gate C thresholds",
                            "Sensitivity analyses",
                            "Exploratory comparators",
                            "Training data and censoring of alerted players",
                            "Reported whatever the outcome",
                        ],
                        "value": [
                            pre.model_version,
                            pre.config_hash,
                            pre.data,
                            pre.primary,
                            pre.thresholds,
                            pre.sensitivity,
                            pre.exploratory,
                            pre.training_data,
                            pre.reported_regardless,
                        ],
                    }
                )
            )
        )
    out += ["## Gate C criteria", ""]
    out.append(
        _md_table(
            pd.DataFrame(
                {
                    "criterion": [c.name for c in r.criteria],
                    "result": ["PASS" if c.passed else "FAIL" for c in r.criteria],
                    "detail": [c.detail for c in r.criteria],
                }
            )
        )
    )
    out += ["## Secondary metrics (spell level, pooled test folds)", ""]
    out.append(_md_table(r.metrics))
    for name, table in r.oe.items():
        out += [f"### O/E of M2 by {name}", "", _md_table(table)]
    if r.alerts is not None:
        a = r.alerts
        out += [
            "## Alert metrics (flags on the M2 / M1 spell ratio)",
            "",
            f"- Flagged share {a.flag_share:.4f} ({a.alerts_per_team_match:.2f} per team-match), "
            f"{a.n_flagged} of {a.n_spells} spells.",
            f"- Injuries in flagged spells {a.n_caught} of {a.n_injuries} "
            f"(share {_fmt(a.recall)}, precision {_fmt(a.precision)}, enrichment "
            f"{_fmt(a.enrichment)}); median lead {_fmt(a.median_lead_min, 3)} min.",
            "",
        ]
    out += ["## Decision curves (net benefit per spell)", "", _md_table(r.decision, 3)]
    out += ["## Benchmarks (descriptive)", "", _md_table(r.benchmarks)]
    if r.diagnostics is not None:
        d = r.diagnostics
        ia = d.input_attribution
        out += [
            "## Diagnostics",
            "",
            "### Input attribution",
            "",
            f"- Slope on ln(Lambda): {ia.input_slope:.3f} (95 % CI {ia.input_slope_ci[0]:.3f} "
            f"to {ia.input_slope_ci[1]:.3f}); M1 O/E {ia.oe_m1:.3f} (95 % CI "
            f"{ia.oe_m1_ci[0]:.3f} to {ia.oe_m1_ci[1]:.3f}). Verdict: `{ia.verdict}`.",
            "",
            "### Substitution intensity (cloglog, per SD)",
            "",
            _md_table(d.substitution_intensity),
            "### IPCW-weighted live coefficients (sensitivity)",
            "",
            _md_table(d.ipcw),
            "### Tipping point (substituted minutes imputed at psi times the usual hazard)",
            "",
            _md_table(d.tipping),
            "Smallest psi erasing each effect: "
            + ", ".join(
                f"{k} {'not reached' if v is None else _fmt(v, 2)}"
                for k, v in d.tipping_points.items()
            )
            + ".",
            "",
            "### Aalen-Johansen cumulative incidence (pooled test rows)",
            "",
            _md_table(_aj_summary(d.aalen_johansen), 5),
        ]
    out += [
        "## Exploratory comparators (not confirmatory)",
        "",
        "Holm-adjusted across the comparators that ran; none is adopted on this data. "
        "Adoption needs a new pre-registration and a later season.",
        "",
        _md_table(r.exploratory),
    ]
    if r.training is not None:
        excluded = r.training.excluded_minutes
        out += [
            "## Training data",
            "",
            "Excluded player-minutes: "
            + (", ".join(f"{k} {v}" for k, v in excluded.items()) if excluded else "none")
            + f" (total {r.training.n_excluded}).",
            "",
        ]
    return "\n".join(out)


def _aj_summary(table: pd.DataFrame) -> pd.DataFrame:
    """The cumulative incidence at the end of each 15-minute block of regular time."""
    marks = [14.0, 29.0, 44.0, 59.0, 74.0, 89.0]
    keep = table.loc[table["t_min"].isin(marks)]
    return keep.reset_index(drop=True)


def _render_model_card(r: GateCReport) -> str:
    fit = r.final_fit
    out = ["# Model card: live hazard layer", ""]
    out += [
        "## Intended use",
        "",
        "Decision support for qualified bench staff: each player's risk for the rest of the "
        "match relative to his own normal, with data age and quality. It reports risk, not "
        "outcomes: it does not diagnose, forecasts no individual injury and does not recommend "
        "substitutions.",
        "",
        "## Forbidden uses",
        "",
        "- Automatic substitution or selection decisions.",
        "- Presenting a low value as an all-clear: most injuries happen to players rated ordinary.",
        "- Use outside the populations and data sources it was evaluated on.",
        "",
        "## Data",
        "",
    ]
    if r.preregistration is not None:
        out += [
            f"- {r.preregistration.data}.",
            f"- Training data: {r.preregistration.training_data}.",
        ]
    out += [
        f"- Evaluation folds: {', '.join(r.folds)}; {r.n_spells} test spells, {r.n_events} events.",
        "- Outcome: non-contact time-loss injury (>= 7 days) with onset in the match; contact "
        "injuries are a competing cause.",
        "",
        "## Features",
        "",
        ", ".join(r.feature_names) + " (player-relative z-scores, docs/02_MATH.md §10).",
        "",
    ]
    if fit is not None:
        sd = np.sqrt(np.clip(np.diag(fit.cov), 0.0, None))
        params = pd.DataFrame(
            {
                "parameter": list(fit.param_names),
                "estimate": fit.point,
                "lo95": fit.point - 1.959964 * sd,
                "hi95": fit.point + 1.959964 * sd,
            }
        )
        out += [
            "## Priors",
            "",
            f"alpha N(0, {fit.priors.alpha_sd}^2); theta N(0, {fit.priors.theta_sd}^2); "
            f"beta N(0, {fit.priors.beta_sd}^2).",
            "",
            f"## Fitted parameters ({fit.n_rows} person-minutes, {fit.n_events} events)",
            "",
            _md_table(params),
        ]
    out += ["## Metrics with intervals", ""]
    if r.primary is not None:
        p = r.primary.improvement
        out.append(
            f"- Primary: log-loss improvement of M2 over M1 {p.point:.6f} nats per spell "
            f"(95 % CI {p.lo:.6f} to {p.hi:.6f})."
        )
    out += ["", _md_table(r.metrics), "## Calibration by decile of predicted risk", ""]
    if "decile" in r.oe:
        out.append(_md_table(r.oe["decile"]))
    out += ["## Subgroups", ""]
    for name in ("club", "position"):
        if name in r.oe:
            out += [f"### By {name}", "", _md_table(r.oe[name])]
    out += [
        "## Known limits",
        "",
        "- Intervals include parameter and pre-match uncertainty, not norm uncertainty.",
        "- Substitution may remove high-risk players for reasons the data do not contain; the "
        "tipping-point analysis bounds that bias, it does not remove it.",
        "- Calibration holds for the data sources and populations above only.",
        f"- Gate C verdict for this version: engine mode `{r.mode}`.",
        "",
    ]
    return "\n".join(out)
