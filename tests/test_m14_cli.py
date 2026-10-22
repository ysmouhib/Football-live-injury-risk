from __future__ import annotations

import pytest

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m14]


def test_replay_cli_end_to_end() -> None:
    # `livrisk replay --synthetic --seed 1` runs and serves the app (use a short replay in tests)
    pytest.fail("implement")


def test_fit_and_evaluate_reproduce_report() -> None:
    pytest.fail("implement")
