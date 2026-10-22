from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from livrisk import contracts as c


def _sample(**kw: object) -> c.TrackingSample:
    base: dict[str, object] = dict(
        match_id="m1",
        player_id="p1",
        team_id="t1",
        period=1,
        t_s=10.0,
        x_m=1.0,
        y_m=2.0,
        source=c.Source.SYNTHETIC,
    )
    base.update(kw)
    return c.TrackingSample(**base)


def test_tracking_sample_valid() -> None:
    s = _sample()
    assert s.identity_confidence == 1.0


def test_tracking_sample_needs_position_or_speed() -> None:
    with pytest.raises(ValidationError):
        _sample(x_m=None, y_m=None)
    assert _sample(x_m=None, y_m=None, speed_mps=3.0).speed_mps == 3.0


def test_tracking_sample_period_clock() -> None:
    with pytest.raises(ValidationError):
        _sample(period=2, t_s=100.0)  # period 2 starts at 2700 s


def test_substitution_needs_both_players() -> None:
    with pytest.raises(ValidationError):
        c.MatchEvent(
            match_id="m1", period=2, t_s=3000.0, type=c.EventType.SUBSTITUTION, player_id="p1"
        )


def test_lineup_unique_and_max_starters() -> None:
    players = tuple(
        c.LineupPlayer(
            player_id=f"p{i}", shirt_number=i, position_group=c.PositionGroup.MF, starting=True
        )
        for i in range(1, 13)
    )
    with pytest.raises(ValidationError):
        c.Lineup(
            match_id="m1",
            team_id="t1",
            kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
            players=players,
        )


def test_prematch_risk_bounds() -> None:
    with pytest.raises(ValidationError):
        c.PreMatchRisk(player_id="p1", as_of_date=date(2026, 1, 1), p7=0.0, model_version="v1")


def test_feature_snapshot_rejects_lookahead() -> None:
    with pytest.raises(ValidationError):
        c.FeatureSnapshot(
            match_id="m1",
            player_id="p1",
            period=1,
            as_of_t_s=600.0,
            cutoff_t_s=601.0,
            features={},
            z={},
            coverage={},
            identity_confidence=1.0,
            quality=c.Quality.GOOD,
        )


def _risk(**kw: object) -> c.RiskSnapshot:
    base: dict[str, object] = dict(
        match_id="m1",
        player_id="p1",
        period=2,
        as_of_t_s=4000.0,
        data_age_s=8.0,
        quality=c.Quality.GOOD,
        mode=c.Mode.RISK,
        p15=0.003,
        p15_lo=0.002,
        p15_hi=0.005,
        r_rest=0.01,
        r_rest_lo=0.007,
        r_rest_hi=0.015,
        r_usual=0.006,
        rr=1.6,
        prematch_band=c.PrematchBand.WATCH,
        model_version="v0",
    )
    base.update(kw)
    return c.RiskSnapshot(**base)


def test_risk_snapshot_interval_order() -> None:
    _risk()
    with pytest.raises(ValidationError):
        _risk(p15=0.01)


def test_monitor_only_requires_rr_one() -> None:
    with pytest.raises(ValidationError):
        _risk(mode=c.Mode.MONITOR_ONLY)
    _risk(mode=c.Mode.MONITOR_ONLY, rr=1.0)


def test_models_are_frozen() -> None:
    s = _sample()
    with pytest.raises(ValidationError):
        s.t_s = 20.0  # type: ignore[misc]


def test_prematch_interval_both_or_none_and_ordered() -> None:
    ok = c.PreMatchRisk(
        player_id="p1",
        as_of_date=date(2026, 1, 1),
        p7=0.02,
        p7_lo=0.01,
        p7_hi=0.04,
        model_version="v1",
    )
    assert ok.interval_level == 0.90
    with pytest.raises(ValidationError):
        c.PreMatchRisk(
            player_id="p1", as_of_date=date(2026, 1, 1), p7=0.02, p7_lo=0.01, model_version="v1"
        )
    with pytest.raises(ValidationError):
        c.PreMatchRisk(
            player_id="p1",
            as_of_date=date(2026, 1, 1),
            p7=0.05,
            p7_lo=0.01,
            p7_hi=0.04,
            model_version="v1",
        )


def test_feature_snapshot_quality_reasons_default_empty() -> None:
    f = c.FeatureSnapshot(
        match_id="m1",
        player_id="p1",
        period=1,
        as_of_t_s=600.0,
        cutoff_t_s=540.0,
        features={},
        z={},
        coverage={},
        identity_confidence=1.0,
        quality=c.Quality.DEGRADED,
        quality_reasons=("norm_reset",),
    )
    assert f.quality_reasons == ("norm_reset",)
