from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from livrisk import riskmath
from livrisk.app.wording import find_forbidden
from livrisk.config import Config, load_config
from livrisk.evaluation.bootstrap import cluster_bootstrap_delta, mean_interval, ratio_draws
from livrisk.evaluation.decision_curves import net_benefit
from livrisk.evaluation.gate_c import (
    CriterionResult,
    FoldPrep,
    GateCReport,
    cross_fit,
    primary_test,
    run_gate_c,
    standardize_in_fold,
)
from livrisk.evaluation.metrics import (
    brier,
    calibration_slope_intercept,
    log_loss,
    pr_auc,
    spell_index,
    spell_table,
)
from livrisk.evaluation.sensitivity import aalen_johansen, input_attribution
from livrisk.evaluation.splits import (
    TrainingRows,
    club_splits,
    time_splits,
    training_policy,
)
from livrisk.synth.generator import SynthTruth, generate_person_minutes
from tests.conftest import DEFAULT_CONFIG, DEV_SEEDS

pytestmark = [pytest.mark.milestone_m10]

SIGNAL = SynthTruth(p7_mean=0.10)
NULL = SynthTruth(p7_mean=0.10, beta=(0.0,) * 6)
#: True signals of SynthTruth and the coefficients that are exactly zero.
NONZERO = ("hsr_5m", "recovery_10m", "nearmax_cum", "disrupt_5m")
ZERO = ("spr_5m", "hir_drop_15m")


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def signal_minutes() -> pd.DataFrame:
    return generate_person_minutes(9000, seed=11, truth=SIGNAL)


@pytest.fixture(scope="module")
def report(cfg: Config, signal_minutes: pd.DataFrame) -> GateCReport:
    return run_gate_c(signal_minutes, SIGNAL.feature_names, cfg, split="time", seed=1)


def test_basic_scores() -> None:
    y = np.array([0, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.7, 0.1])
    assert brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    assert log_loss(y, p) == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert pr_auc(y, p) == pytest.approx(1.0)
    assert math.isnan(pr_auc(np.zeros(4), p))


def _minutes(rows: list[tuple[str, int, int, int, bool]], hazard: float) -> pd.DataFrame:
    """(player, period, t_min, y, substituted_after) rows of one match each."""
    frame = pd.DataFrame(rows, columns=["player_id", "period", "t_min", "y", "substituted_after"])
    frame["match_id"] = "m-" + frame["player_id"]
    frame["h"] = hazard
    frame["contact"] = False
    return frame


def test_spell_probability_covers_the_rest_of_an_onset_spell() -> None:
    injured = [("A", 1, t, int(t == 3), False) for t in range(4)]
    subbed = [("B", 2, t, 0, t == 52) for t in range(45, 53)]
    spells = spell_table(_minutes(injured + subbed, 0.01), "h")
    a = spells.loc[spells["player_id"] == "A"].iloc[0]
    b = spells.loc[spells["player_id"] == "B"].iloc[0]
    # The onset spell keeps its planned 15 minutes; a substitution censors exposure.
    assert a["y"] == 1 and a["n_min"] == 4 and a["onset_t_min"] == 3.0
    assert a["h"] == pytest.approx(riskmath.prob_from_cumhaz(15 * 0.01))
    assert b["y"] == 0 and b["h"] == pytest.approx(riskmath.prob_from_cumhaz(8 * 0.01))
    idx = spell_index(_minutes(injured + subbed, 0.01))
    assert idx.tolist() == [0] * 4 + [1] * 8


def test_calibration_slope_recovers_a_known_miscalibration() -> None:
    rng = np.random.default_rng(3)
    p_true = rng.uniform(0.002, 0.05, size=200_000)
    y = (rng.random(p_true.size) < p_true).astype(np.float64)
    assert calibration_slope_intercept(y, p_true)[0] == pytest.approx(1.0, abs=0.05)
    too_spread = riskmath.inv_cloglog(2.0 * riskmath.cloglog(p_true) + 3.0)
    assert calibration_slope_intercept(y, too_spread)[0] == pytest.approx(0.5, abs=0.05)


def test_bootstrap_and_decision_curve_helpers() -> None:
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(300), 5)
    values = rng.normal(0.2, 1.0, size=clusters.size)
    interval, draws = mean_interval(values, clusters, n_boot=400, seed=4)
    assert interval.lo < interval.point < interval.hi and draws.shape == (400,)
    _, a = ratio_draws(values, np.ones_like(values), clusters, 400, 4)
    _, b = ratio_draws(2.0 * values, np.ones_like(values), clusters, 400, 4)
    assert np.allclose(b, 2.0 * a)  # same seed, same resamples: paired statistics
    y = (rng.random(clusters.size) < 0.3).astype(np.float64)
    point, lo, hi = cluster_bootstrap_delta(brier, y, np.full(y.size, 0.3), y * 0.0 + 0.5, clusters)
    assert lo < point < hi
    nb = net_benefit(np.array([1, 0, 0, 0]), np.array([0.5, 0.5, 0.0, 0.0]), [0.25])
    assert nb[0] == pytest.approx(1 / 4 - 1 / 4 * 0.25 / 0.75)


def test_time_and_club_splits(signal_minutes: pd.DataFrame) -> None:
    folds = time_splits(signal_minutes)
    assert [f.name for f in folds] == ["season s2", "season s3"]
    seasons = signal_minutes["season"].astype(str).to_numpy()
    assert set(seasons[folds[0].train]) == {"s1"} and set(seasons[folds[1].train]) == {"s1", "s2"}
    clubs = club_splits(signal_minutes)
    assert len(clubs) == 10
    match = pd.factorize(signal_minutes["match_id"])[0]
    for fold in [*folds, *clubs]:
        assert not np.any(fold.train & fold.test)
        assert np.intersect1d(match[fold.train], match[fold.test]).size == 0


def test_signal_detected_and_null_controlled(cfg: Config, report: GateCReport) -> None:
    assert report.primary is not None and report.primary.passed
    assert report.primary.improvement.lo > 0.0
    assert report.criteria[0].passed
    claimed = 0
    seeds = DEV_SEEDS[:20]
    for seed in seeds:
        minutes = generate_person_minutes(4000, seed=seed, truth=NULL)
        cross = cross_fit(minutes, NULL.feature_names, cfg)
        result = primary_test(spell_table(cross.rows, ["m1", "m2"]), n_boot=500, seed=seed)
        claimed += int(result.passed)
    assert claimed <= 0.10 * len(seeds)


def test_oe_and_calibration_on_correctly_specified_data(report: GateCReport) -> None:
    assert {"season", "club", "decile"} <= set(report.oe)
    for table in report.oe.values():
        assert (table["oe"] > 0.0).all()
        assert table["observed"].sum() == report.n_events
    m2 = report.metrics.set_index("model").loc["m2"]
    assert m2["cal_slope"] == pytest.approx(1.0, abs=0.25)
    assert m2["oe"] == pytest.approx(1.0, abs=0.1)
    season = report.oe["season"]
    assert ((season["oe"] >= 0.8) & (season["oe"] <= 1.25)).all()


def _seasons(frame: pd.DataFrame) -> set[str]:
    return set(frame["season"].astype(str))


def test_fold_leak_trap(cfg: Config) -> None:
    minutes = generate_person_minutes(3000, seed=21, truth=SIGNAL)
    corrupted = minutes.copy()
    last = corrupted["season"].astype(str) == "s3"
    corrupted.loc[last, "hsr_5m"] = corrupted.loc[last, "hsr_5m"] + 100.0
    names = SIGNAL.feature_names
    seen: list[tuple[set[str], set[str]]] = []

    def recording(prepare: FoldPrep) -> FoldPrep:
        def wrapped(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            seen.append((_seasons(train), _seasons(test)))
            return prepare(train, test)

        return wrapped

    clean = cross_fit(minutes, names, cfg, prepare=recording(standardize_in_fold(names)))
    dirty = cross_fit(corrupted, names, cfg, prepare=standardize_in_fold(names))
    # The prepare step only ever saw the training seasons of its fold.
    assert seen == [({"s1"}, {"s2"}), ({"s1", "s2"}, {"s3"})]
    for a, b in zip(clean.fits, dirty.fits, strict=True):
        assert np.array_equal(a.m2.point, b.m2.point)

    def leaky(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        both = pd.concat([train, test])
        mean = both[list(names)].mean()
        sd = both[list(names)].std()
        return (
            train.assign(**{n: (train[n] - mean[n]) / sd[n] for n in names}),
            test.assign(**{n: (test[n] - mean[n]) / sd[n] for n in names}),
        )

    leak_clean = cross_fit(minutes, names, cfg, prepare=leaky)
    leak_dirty = cross_fit(corrupted, names, cfg, prepare=leaky)
    # The trap catches a step that mixes test rows into what training sees.
    assert not np.array_equal(leak_clean.fits[1].m2.point, leak_dirty.fits[1].m2.point)


PRE_REGISTRATION_FIELDS = (
    "Model version / commit",
    "Config hash",
    "Data",
    "Primary comparison and metric",
    "Gate C thresholds",
    "Sensitivity analyses",
    "Exploratory comparators",
    "Training data and censoring of alerted players",
    "Reported whatever the outcome",
)


def test_report_lists_all_gate_c_criteria(report: GateCReport) -> None:
    assert [c.name[:2] for c in report.criteria] == ["1.", "2.", "3.", "4."]
    md = report.to_markdown()
    assert "## Gate C criteria" in md and "## Pre-registration" in md
    for criterion in report.criteria:
        assert criterion.name in md
        assert ("PASS" if criterion.passed else "FAIL") in md
    for label in PRE_REGISTRATION_FIELDS:
        assert f"| {label} |" in md
    assert report.passed and report.mode == "risk" and "engine.mode: risk" in md
    assert report.benchmarks.shape[0] == 3 and np.isfinite(report.benchmarks["log_loss"]).all()
    card = report.model_card()
    for section in (
        "## Intended use",
        "## Forbidden uses",
        "## Data",
        "## Features",
        "## Priors",
        "## Fitted parameters",
        "## Metrics with intervals",
        "## Calibration by decile",
        "## Subgroups",
        "## Known limits",
    ):
        assert section in card
    assert find_forbidden(md) == [] and find_forbidden(card) == []

    failed = GateCReport(criteria=(CriterionResult("1. primary", False, "no gain"),))
    assert failed.mode == "monitor_only"
    assert "engine.mode: monitor_only" in failed.to_markdown()


def _misreport(minutes: pd.DataFrame, slope: float) -> pd.DataFrame:
    """p7 that is too spread: ln H_true = c + slope * ln H_reported (H = -ln(1 - p7))."""
    ln_h = np.log(-np.log1p(-minutes["p7"].to_numpy(dtype=np.float64)))
    centre = float(ln_h.mean())
    reported = np.exp(centre + (ln_h - centre) / slope)
    return minutes.assign(p7=-np.expm1(-reported))


def test_input_attribution_detects_miscalibrated_prematch_input(cfg: Config) -> None:
    minutes = generate_person_minutes(7000, seed=31, truth=SIGNAL)
    train = minutes["season"].astype(str) != "s3"
    good = input_attribution(minutes.loc[train], minutes.loc[~train], cfg, n_boot=400)
    assert good.verdict == "input_ok"
    assert good.input_slope_ci[0] <= 1.0 <= good.input_slope_ci[1]
    bad_minutes = _misreport(minutes, 0.6)
    bad = input_attribution(bad_minutes.loc[train], bad_minutes.loc[~train], cfg, n_boot=400)
    assert bad.verdict == "input_miscalibrated"
    assert bad.input_slope_ci[1] < 1.0
    assert bad.input_slope == pytest.approx(0.6, abs=0.2)


def test_tipping_point_is_monotone_and_reported(report: GateCReport) -> None:
    diag = report.diagnostics
    assert diag is not None
    tip = diag.tipping
    assert sorted(tip["psi"].unique().tolist()) == [1.0, 1.5, 2.0, 3.0, 5.0]
    for signal in NONZERO:
        coef = tip.loc[tip["signal"] == signal].sort_values("psi")["coef"].abs().to_numpy()
        assert np.all(np.diff(coef) <= 1e-9), signal
        assert coef[-1] < coef[0]
    for signal in ZERO:
        assert diag.tipping_points[signal] == 1.0
    assert set(diag.tipping_points) == set(SIGNAL.feature_names)
    md = report.to_markdown()
    assert "Smallest psi erasing each effect" in md
    # Substitution in the generator depends on hsr_5m with log hazard ratio 0.5 only.
    sub = diag.substitution_intensity.set_index("signal")["coef"]
    assert sub["hsr_5m"] == pytest.approx(SIGNAL.sub_log_hr, abs=0.1)
    assert (sub.drop("hsr_5m").abs() < 0.1).all()
    assert diag.ipcw.shape[0] == len(SIGNAL.feature_names)


def test_aalen_johansen_matches_hand_count_and_model(report: GateCReport) -> None:
    rows = [("A", 1, 0, 0, False), ("A", 1, 1, 1, False), ("B", 1, 0, 0, False)]
    rows += [("B", 1, 1, 0, False), ("B", 1, 2, 0, False)]
    aj = aalen_johansen(_minutes(rows, 0.01))
    assert aj["n_at_risk"].tolist() == [2, 2, 1]
    assert aj["cif_noncontact"].tolist() == pytest.approx([0.0, 0.5, 0.5])
    assert report.diagnostics is not None
    pooled = report.diagnostics.aalen_johansen
    assert pooled["cif_noncontact"].is_monotonic_increasing
    end = pooled.iloc[-1]
    assert end["cif_noncontact_model"] == pytest.approx(end["cif_noncontact"], rel=0.15)


def test_exploratory_comparisons_are_holm_adjusted(cfg: Config, report: GateCReport) -> None:
    table = report.exploratory
    assert table["comparator"].tolist() == list(cfg.evaluation.exploratory)
    status = table.set_index("comparator")["status"]
    assert status["ewma_features"] == "run"
    assert status["team_relative"].startswith("not run")
    md = report.to_markdown()
    assert md.index("## Exploratory comparators (not confirmatory)") > md.index(
        "## Gate C criteria"
    )
    assert "exploratory" not in report.criteria[0].detail.lower()

    minutes = generate_person_minutes(4000, seed=41, truth=SIGNAL)
    minutes["team_id"] = minutes["club_id"].astype(str)
    small = run_gate_c(
        minutes, SIGNAL.feature_names, cfg, seed=2, n_boot=400, diagnostics=False, benchmarks=False
    )
    ran = small.exploratory.loc[small.exploratory["status"] == "run"]
    assert ran["comparator"].tolist() == ["team_relative", "ewma_features"]
    expected = riskmath.holm_adjust(ran["p_raw"].to_numpy(dtype=np.float64))
    assert np.allclose(ran["p_holm"].to_numpy(dtype=np.float64), expected)


def test_training_sources_exclude_alerted_minutes(cfg: Config) -> None:
    minutes = generate_person_minutes(600, seed=51, truth=SIGNAL)
    match = minutes["match_id"].astype(str)
    codes = pd.factorize(match)[0] % 4
    minutes["source"] = np.asarray(["historical", "shadow", "live", "replay"])[codes]
    minutes["first_alert_t_min"] = np.where(codes == 2, 30.0, np.nan)
    live_no_alert = (codes == 2) & (pd.factorize(match)[0] % 8 == 6)
    minutes.loc[live_no_alert, "first_alert_t_min"] = np.nan
    t = minutes["t_min"].to_numpy()
    policy = training_policy(minutes, cfg)
    keep = policy.keep
    assert keep[codes <= 1].all()
    assert not keep[codes == 3].any()
    live = codes == 2
    expect_live = np.isnan(minutes["first_alert_t_min"].to_numpy()) | (t < 30.0)
    assert np.array_equal(keep[live], expect_live[live])
    assert policy.excluded_minutes == {
        "live": int((live & ~expect_live).sum()),
        "replay": int((codes == 3).sum()),
    }
    cross = cross_fit(minutes, SIGNAL.feature_names, cfg)
    first_train = time_splits(minutes)[0].train
    assert cross.fits[0].m2.n_rows == int((first_train & keep).sum())

    no_live = cfg.model_copy(
        update={"competition": cfg.competition.model_copy(update={"live_data_allowed": False})}
    )
    assert not training_policy(minutes, no_live).keep[live].any()
    shown = GateCReport(
        criteria=(), training=TrainingRows(keep=keep, excluded_minutes=policy.excluded_minutes)
    ).to_markdown()
    assert f"live {policy.excluded_minutes['live']}" in shown
    assert f"total {policy.n_excluded}" in shown
