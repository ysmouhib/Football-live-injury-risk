from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from livrisk import riskmath
from livrisk.contracts import PrematchBand, PreMatchRisk
from livrisk.model import prematch

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m5]


def test_match_cumhaz_uses_riskmath(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[float, float]] = []

    def fake(p7: float, match_share: float = 0.6) -> float:
        calls.append((p7, match_share))
        return 0.123

    monkeypatch.setattr(riskmath, "weekly_to_match_cumhaz", fake)
    r = PreMatchRisk(player_id="p", as_of_date=date(2027, 1, 1), p7=0.02, model_version="v")
    assert prematch.match_cumhaz(r, 0.6) == 0.123
    assert calls == [(0.02, 0.6)]


def test_rejects_lookahead_dates(tmp_path: Path) -> None:
    p = tmp_path / "pm.csv"
    p.write_text(
        "player_id,as_of_date,p7,model_version,congested,days_since_last_match\n"
        "p1,2027-01-02,0.02,v1,false,4\n"
    )
    with pytest.raises(ValueError):
        prematch.load_prematch_csv(p, match_date=date(2027, 1, 2))


def test_bands() -> None:
    assert prematch.prematch_band(0.005, 0.012, 0.020) == PrematchBand.LOW
    assert prematch.prematch_band(0.015, 0.012, 0.020) == PrematchBand.WATCH
    assert prematch.prematch_band(0.030, 0.012, 0.020) == PrematchBand.HIGH
