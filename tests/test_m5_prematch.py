from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from livrisk import riskmath
from livrisk.config import load_config
from livrisk.contracts import PrematchBand, PreMatchRisk
from livrisk.model import prematch

pytestmark = [pytest.mark.milestone_m5]


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


def test_rejects_duplicate_players(tmp_path: Path) -> None:
    p = tmp_path / "pm.csv"
    p.write_text(
        "player_id,as_of_date,p7,model_version\np1,2027-01-01,0.02,v1\np1,2027-01-01,0.03,v1\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        prematch.load_prematch_csv(p, match_date=date(2027, 1, 2))


def test_loads_interval_and_optional_fields(tmp_path: Path) -> None:
    p = tmp_path / "pm.csv"
    p.write_text(
        "player_id,as_of_date,p7,p7_lo,p7_hi,interval_level,model_version,congested,"
        "days_since_last_match,days_since_return\n"
        "p1,2027-01-01,0.02,0.01,0.04,0.9,v1,true,4,21\n"
        "p2,2027-01-01,0.03,,,,v1,false,3,\n"
    )
    out = prematch.load_prematch_csv(p, match_date=date(2027, 1, 2))
    assert set(out) == {"p1", "p2"}
    assert out["p1"].p7_lo == 0.01 and out["p1"].p7_hi == 0.04
    assert out["p1"].congested is True and out["p1"].days_since_return == 21
    assert out["p2"].p7_lo is None and out["p2"].p7_hi is None


def test_bands() -> None:
    assert prematch.prematch_band(0.005, 0.012, 0.020) == PrematchBand.LOW
    assert prematch.prematch_band(0.015, 0.012, 0.020) == PrematchBand.WATCH
    assert prematch.prematch_band(0.030, 0.012, 0.020) == PrematchBand.HIGH


def test_bands_follow_config_cutoffs(config_path: Path) -> None:
    cfg = load_config(config_path)
    watch, high = cfg.prematch.band_cutoffs.watch, cfg.prematch.band_cutoffs.high
    edge = PreMatchRisk(player_id="p", as_of_date=date(2027, 1, 1), p7=0.02, model_version="v")
    cumhaz = prematch.match_cumhaz(edge, cfg.prematch.match_share_prior)
    r90 = prematch.usual_risk(cumhaz)
    assert prematch.prematch_band(r90, watch, high) in set(PrematchBand)
    assert prematch.prematch_band(0.5 * watch, watch, high) == PrematchBand.LOW
    assert prematch.prematch_band(2.0 * high, watch, high) == PrematchBand.HIGH


def test_usual_risk_scales_with_expected_minutes() -> None:
    cumhaz = riskmath.weekly_to_match_cumhaz(0.03, 0.6)
    full = prematch.usual_risk(cumhaz, 90.0)
    half = prematch.usual_risk(cumhaz, 45.0)
    assert full == pytest.approx(riskmath.prob_from_cumhaz(cumhaz))
    assert half == pytest.approx(riskmath.prob_from_cumhaz(cumhaz / 2.0))
    with pytest.raises(ValueError):
        prematch.usual_risk(cumhaz, 0.0)
