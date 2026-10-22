from __future__ import annotations

import numpy as np
import pytest

from livrisk.evaluation.metrics import brier, log_loss

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m10]


def test_basic_scores() -> None:
    y = np.array([0, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.7, 0.1])
    assert brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    assert log_loss(y, p) == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_signal_detected_and_null_controlled() -> None:
    # With SynthTruth signal: run_gate_c finds M2 > M1 (CI excludes 0).
    # With beta = 0: M2 > M1 claimed in <= 10 % of seeds.
    pytest.fail("implement")


def test_fold_leak_trap() -> None:
    # Norms/pre-match inputs must be recomputed inside each training fold.
    pytest.fail("implement")


def test_report_lists_all_gate_c_criteria() -> None:
    pytest.fail("implement")


def test_input_attribution_detects_miscalibrated_prematch_input() -> None:
    # TODO: generate data whose p7 is too spread (true slope 0.6 on ln Lambda); input_attribution
    # returns input_slope CI excluding 1 and verdict 'input_miscalibrated'; with correct p7 it
    # returns 'input_ok'.
    pytest.fail("implement: input attribution")


def test_tipping_point_is_monotone_and_reported() -> None:
    # TODO: with informative censoring in the generator, beta intervals move toward 0 as the
    # multiplier grows; the report names the smallest multiplier that erases each effect.
    pytest.fail("implement: tipping point")


def test_exploratory_comparisons_are_holm_adjusted() -> None:
    # TODO: the Gate C report lists exploratory comparators from evaluation.exploratory with
    # riskmath.holm_adjust p-values, separate from the primary test, marked 'not confirmatory'.
    pytest.fail("implement: multiplicity")


def test_training_sources_exclude_alerted_minutes() -> None:
    # TODO: rows from live-mode matches are censored at each player's first shown alert; only
    # historical and shadow rows enter unchanged; the excluded minutes are counted in the report.
    pytest.fail("implement: feedback-loop policy")
