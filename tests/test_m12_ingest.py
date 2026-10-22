from __future__ import annotations

import pytest

from tests.conftest import OPEN_DATA, skip_without

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m12]


def test_tabular_roundtrip() -> None:
    # Export a synthetic match to CSV with vendor-style column names; read back via column_map.
    pytest.fail("implement")


@pytest.mark.needs_open_data
@skip_without(OPEN_DATA / "skillcorner")
def test_skillcorner_match_end_to_end() -> None:
    pytest.fail("implement")


def test_clock_drift_flags_degraded() -> None:
    pytest.fail("implement")
