from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from livrisk.app import wording
from livrisk.app.payloads import (
    COACH_ALERT_KEYS,
    COACH_KEYS,
    NUMBER_FIELDS,
    RowInput,
    collect_strings,
    live_row,
    live_table,
)
from livrisk.app.server import create_app
from livrisk.app.state import ReplayState
from livrisk.config import Config, config_hash
from livrisk.contracts import Mode, PrematchBand, Quality, RiskSnapshot
from livrisk.storage.logstore import LogStore
from tests.replay_setup import ReplayBundle

pytestmark = [pytest.mark.milestone_m11]

NS = 1_000_000_000
#: Row and payload keys of the docs/06_UI_SPEC.md §8 example.
PAYLOAD_KEYS = {
    "type",
    "match_id",
    "period",
    "clock",
    "data_age_s",
    "alerts_used",
    "alerts_budget",
    "mode",
    "rows",
    "alerts",
}
ROW_KEYS = {
    "player_id",
    "label",
    "prematch_band",
    "rr",
    "r_rest",
    "r_rest_lo",
    "r_rest_hi",
    "r_usual",
    "p15",
    "reason_code",
    "reason_text",
    "quality",
    "data_age_s",
}
ALERT_KEYS = {"alert_id", "player_id", "level", "reason_code"}


def _state(bundle: ReplayBundle, store: LogStore | None = None) -> ReplayState:
    return ReplayState(
        bundle.match, bundle.result, bundle.fitted, bundle.norms, bundle.cfg, store=store
    )


def _first_alert_wall_s(bundle: ReplayBundle, state: ReplayState) -> tuple[float, str]:
    """Seconds after kick-off when the first alert of the shown team reached the screen."""
    kickoff_ns = int(pd.Timestamp(bundle.match.lineups[0].kickoff_utc).value)
    alerts = bundle.result.alerts
    team = sorted(lu.team_id for lu in bundle.match.lineups)[0]
    opened = alerts.loc[(alerts["team_id"] == team) & (alerts["change"] == "opened")]
    assert opened.shape[0] > 0, "the shared replay should open at least one alert"
    first = opened.iloc[0]
    return (int(first["shown_wall_ns"]) - kickoff_ns) / NS, str(first["alert_id"])


@pytest.fixture()
def store(tmp_path: Path, replay_bundle: ReplayBundle) -> LogStore:
    cfg = replay_bundle.cfg
    return LogStore(tmp_path, replay_bundle.match.match_id, config_hash=config_hash(cfg))


@pytest.fixture()
def client(
    replay_bundle: ReplayBundle, store: LogStore
) -> Iterator[tuple[TestClient, ReplayState]]:
    state = _state(replay_bundle, store)
    with TestClient(create_app(state, replay_bundle.cfg, push_interval_s=0.01)) as c:
        yield c, state


def test_live_payload_schema(
    replay_bundle: ReplayBundle, client: tuple[TestClient, ReplayState]
) -> None:
    c, state = client
    wall, _ = _first_alert_wall_s(replay_bundle, state)
    state.seek(wall + 5.0)
    payload = c.get("/api/live", params={"role": "analyst"}).json()
    assert set(payload) >= PAYLOAD_KEYS and payload["type"] == "live_table"
    assert payload["alerts_budget"] == replay_bundle.cfg.alerts.budget_per_match
    assert 1 <= payload["alerts_used"] <= payload["alerts_budget"]
    assert len(payload["rows"]) >= 10
    for row in payload["rows"]:
        assert set(row) >= ROW_KEYS
        assert isinstance(row["reason_text"], str) and len(row["reason_text"]) <= 36
        assert row["quality"] in {q.value for q in Quality}
    shown = [r["r_rest"] for r in payload["rows"] if r["r_rest"] is not None]
    assert shown == sorted(shown, reverse=True)
    assert payload["alerts"] and all(set(a) >= ALERT_KEYS for a in payload["alerts"])
    assert payload["alerts"][0]["disclaimer"] == wording.ALERT_DISCLAIMER
    with c.websocket_connect("/ws/live?role=analyst") as ws:
        assert ws.receive_json() == payload
    assert c.get("/").status_code == 200
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/api/wording").json()["ui"]["col_now"] == wording.UI_TEXT["col_now"]


def _numbers_anywhere(value: Any) -> list[float]:
    if isinstance(value, float):
        return [value]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers_anywhere(v)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers_anywhere(v)]
    return []


def test_coach_role_receives_only_allowed_fields(
    replay_bundle: ReplayBundle, client: tuple[TestClient, ReplayState]
) -> None:
    c, state = client
    wall, _ = _first_alert_wall_s(replay_bundle, state)
    state.seek(wall + 5.0)
    coach = c.get("/api/live", params={"role": "coach"}).json()
    assert set(coach) <= COACH_KEYS and coach["type"] == "coach_summary"
    assert coach["alerts"]
    for alert in coach["alerts"]:
        assert set(alert) == COACH_ALERT_KEYS
        assert alert["label"].startswith("#")
    assert _numbers_anywhere(coach) == []
    with c.websocket_connect("/ws/live?role=coach") as ws:
        assert set(ws.receive_json()) <= COACH_KEYS
    assert c.get("/api/live", params={"role": "agent-x"}).status_code == 403


def _risk(rr: float = 1.6, quality: Quality = Quality.GOOD) -> RiskSnapshot:
    return RiskSnapshot(
        match_id="m1",
        player_id="P1",
        period=2,
        as_of_t_s=4000.0,
        data_age_s=5.0,
        quality=quality,
        mode=Mode.RISK,
        p15=0.004,
        p15_lo=0.003,
        p15_hi=0.006,
        r_rest=0.012,
        r_rest_lo=0.008,
        r_rest_hi=0.02,
        r_usual=0.008,
        rr=rr,
        prematch_band=PrematchBand.WATCH,
        top_reason_code="RECOVERY_SHORT",
        model_version="v",
    )


def test_stale_values_hidden_server_side(
    replay_bundle: ReplayBundle, client: tuple[TestClient, ReplayState]
) -> None:
    cfg: Config = replay_bundle.cfg
    fresh = live_row(RowInput(_risk(), "#4 Full back", 10.0), cfg)
    grey = live_row(RowInput(_risk(), "#4 Full back", 100.0), cfg)
    stale = live_row(RowInput(_risk(), "#4 Full back", 200.0), cfg)
    assert fresh["state"] == "live" and fresh["rr"] == 1.6 and fresh["age_text"] == ""
    assert grey["state"] == "grey" and grey["rr"] == 1.6 and grey["age_text"] == "100 s old"
    assert stale["state"] == "hidden" and all(stale[f] is None for f in NUMBER_FIELDS)
    assert stale["reason_code"] == "DATA_LOST" and stale["quality"] == "stale"
    small = _risk().model_copy(update={"r_rest_lo": 0.0001, "r_rest": 0.0002})
    tiny = live_row(RowInput(small, "#4 Full back", 5.0), cfg)
    assert tiny["r_rest_text"] == wording.UI_TEXT["percent_below"].format(value="0.1")
    table = live_table(
        match_id="m1",
        period=2,
        t_s=4000.0,
        mode="risk",
        rows=[RowInput(_risk(), "#4 Full back", 200.0)],
        open_alerts=[],
        labels={},
        budget_used=0,
        cfg=cfg,
    )
    assert table["banner"] == wording.UI_TEXT["banner_stale"].format(clock="63:20")

    # Right after the second-half kick-off the newest data is from before the break.
    c, state = client
    state.seek(state._p2_start_s + 15.0)
    payload = c.get("/api/live", params={"role": "analyst"}).json()
    # The data stopped at the end of the first half, not 15 minutes of match time earlier.
    end = int(state._p1_end_s)
    expected = wording.UI_TEXT["banner_stale"].format(clock=f"{end // 60}:{end % 60:02d}")
    assert payload["type"] == "live_table" and payload["banner"] == expected
    for row in payload["rows"]:
        assert row["data_age_s"] > cfg.quality.stale_hide_s
        assert row["state"] == "hidden" and all(row[f] is None for f in NUMBER_FIELDS)
    # The header clock follows the second half, not the last first-half update.
    assert payload["period"] == 2 and payload["clock"] == "45:15"


def test_budget_marker_on_rows_without_a_card(replay_bundle: ReplayBundle) -> None:
    # docs/06 §3: a player who meets the card rule after the budget is spent gets a marker.
    cfg: Config = replay_bundle.cfg
    marked = live_row(RowInput(_risk(), "#4 Full back", 10.0, muted=True), cfg)
    assert marked["muted"] and marked["muted_text"] == wording.UI_TEXT["row_muted"]
    carded = live_row(RowInput(_risk(), "#4 Full back", 10.0, alert_level="check", muted=True), cfg)
    plain = live_row(RowInput(_risk(), "#4 Full back", 10.0), cfg)
    assert not carded["muted"] and not plain["muted"] and plain["muted_text"] == ""
    shown = replay_bundle.result.displayed
    assert "muted" in shown.columns and shown["muted"].dtype == bool
    with_card = shown["alert_level"].notna()
    assert not bool((shown["muted"] & with_card).any())


def test_payloads_from_replay_have_no_forbidden_wording(replay_bundle: ReplayBundle) -> None:
    state = _state(replay_bundle)
    seen: set[str] = set()
    found: list[tuple[float, str, str]] = []
    for wall in np.arange(-120.0, state._end_s + 240.0, 30.0):
        state.seek(float(wall))
        for role in replay_bundle.cfg.app.roles:
            payload = state.live_table(role)
            seen.add(str(payload["type"]))
            for text in collect_strings(payload):
                found.extend((float(wall), role, hit) for hit in wording.find_forbidden(text))
    assert seen == {"pre_match", "live_table", "half_time", "post_match", "coach_summary"}
    assert found == []
    assert all(wording.find_forbidden(s) == [] for s in wording.all_strings())


def test_decisions_logged(
    replay_bundle: ReplayBundle, client: tuple[TestClient, ReplayState], store: LogStore
) -> None:
    c, state = client
    wall, alert_id = _first_alert_wall_s(replay_bundle, state)
    state.seek(wall + 5.0)
    body = {"alert_id": alert_id, "actor_role": "physio", "button": "button_physio"}
    res = c.post("/api/decisions", params={"role": "physio"}, json=body)
    assert res.status_code == 201 and res.json()["saved"] == wording.UI_TEXT["decision_saved"]
    logged = store.read("decisions")
    assert logged.shape[0] == 1
    row = logged.iloc[0]
    assert row["alert_id"] == alert_id and row["action"] == "physio_check"
    assert row["actor_role"] == "physio" and row["config_hash"] == config_hash(replay_bundle.cfg)
    assert math.isclose(float(row["t_s"]), float(row["t_s"]))
    bad = c.post("/api/decisions", json={**body, "alert_id": "T1-A99"})
    assert bad.status_code == 400
    assert c.post("/api/decisions", json={**body, "button": "launch"}).status_code == 400
    assert c.post("/api/decisions", json={**body, "extra": 1}).status_code == 422
    state.seek(state._end_s + 120.0)
    post = c.get("/api/live").json()
    assert post["type"] == "post_match"
    entry = next(a for a in post["alerts"] if a["alert_id"] == alert_id)
    assert entry["decisions"] == [wording.UI_TEXT["button_physio"]]


def test_pre_match_and_half_time_screens(replay_bundle: ReplayBundle) -> None:
    state = _state(replay_bundle)
    state.seek(-60.0)
    pre = state.live_table("analyst")
    # Goalkeepers are tracked but left out of risk outputs by default (docs/03 §4).
    assert pre["type"] == "pre_match" and len(pre["rows"]) == 18
    assert [r["starting"] for r in pre["rows"]][:10] == [True] * 10
    assert not any(r["label"].endswith(wording.POSITION_TEXT["GK"]) for r in pre["rows"])
    assert {r["prematch_band_text"] for r in pre["rows"]} <= set(wording.BAND_TEXT.values())
    state.seek(state._p1_end_s + 300.0)
    half = state.live_table("doctor")
    assert half["type"] == "half_time" and len(half["rows"]) >= 10
    for row in half["rows"]:
        assert row["r_45"] > row["r_20"] > 0.0
        assert row["load_lines"]
        assert all(line.endswith("SD vs his usual") for line in row["load_lines"])


def test_lan_exposure_needs_a_token(replay_bundle: ReplayBundle) -> None:
    cfg = replay_bundle.cfg
    lan = cfg.model_copy(update={"app": cfg.app.model_copy(update={"host": "0.0.0.0"})})
    state = _state(replay_bundle)
    with pytest.raises(ValueError):
        create_app(state, lan)
    with TestClient(create_app(state, lan, token="s3cret")) as c:
        assert c.get("/api/live").status_code == 401
        assert c.get("/api/live", params={"token": "s3cret"}).status_code == 200
