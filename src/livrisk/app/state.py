"""Live state of the bench app for a replayed match (replay mode, docs/04_ARCHITECTURE.md §7).

``ReplayState`` serves one team's bench view of a ``ReplayResult`` at a cursor on the simulated
wall clock (seconds since kick-off). The cursor is frozen with ``seek`` or follows real time at a
chosen speed with ``play``. The phase follows the cursor: pre-match before kick-off, live during
play, half-time during the break, post-match after the final whistle. What the live screen shows
at the cursor is the latest push that had reached the screen, aged by the time since it arrived;
staff decisions are validated and written to the log store.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from livrisk import riskmath
from livrisk.app.payloads import (
    RowInput,
    clock_text,
    coach_idle,
    coach_summary,
    half_time_table,
    live_table,
    player_label,
    post_match_report,
    pre_match_table,
)
from livrisk.app.wording import DECISION_BUTTONS
from livrisk.config import Config
from livrisk.contracts import (
    Action,
    ActorRole,
    Alert,
    DecisionRecord,
    PositionGroup,
    PrematchBand,
    RiskSnapshot,
)
from livrisk.features.norms import NormTable
from livrisk.model.live_hazard import FittedHazard
from livrisk.model.prematch import match_cumhaz, prematch_band, usual_risk
from livrisk.replay.simulator import MatchRecord, ReplayResult, break_before_s, period_layout
from livrisk.storage.logstore import LogStore

_NS = 1_000_000_000
#: Seconds of the half-time screen's second-half projections (docs/06 §1: 45 vs 20 more minutes).
_HALF_TIME_HORIZONS_MIN: tuple[float, float] = (45.0, 20.0)
_REASON_FEATURE: dict[str, str] = {"HSR_HIGH": "hsr_5m", "SPR_HIGH": "spr_5m"}


def _risk_snapshot(rec: Mapping[Any, Any]) -> RiskSnapshot:
    data = {name: rec[name] for name in RiskSnapshot.model_fields}
    if data["top_reason_code"] is None or (
        isinstance(data["top_reason_code"], float) and math.isnan(data["top_reason_code"])
    ):
        data["top_reason_code"] = None
    return RiskSnapshot.model_validate(data)


def _alert(rec: Mapping[Any, Any]) -> Alert:
    data = {name: rec[name] for name in Alert.model_fields}
    cleared = data["cleared_t_s"]
    if cleared is None or (isinstance(cleared, float) and math.isnan(cleared)):
        data["cleared_t_s"] = None
    return Alert.model_validate(data)


class ReplayState:
    """``server.LiveState`` for one team of a replayed match."""

    def __init__(
        self,
        match: MatchRecord,
        result: ReplayResult,
        fitted: FittedHazard,
        norms: NormTable,
        cfg: Config,
        *,
        team_id: str | None = None,
        store: LogStore | None = None,
    ) -> None:
        self._cfg = cfg
        self._match_id = match.match_id
        self._fitted = fitted
        self._norms = norms
        self._store = store
        teams = sorted(lu.team_id for lu in match.lineups)
        self._team = team_id or teams[0]
        lineup = next(lu for lu in match.lineups if lu.team_id == self._team)
        self._players = list(lineup.players)
        self._labels = {
            p.player_id: player_label(p.shirt_number, p.position_group.value) for p in self._players
        }
        self._positions: dict[str, PositionGroup] = {
            p.player_id: p.position_group for p in self._players
        }
        self._prematch = {r.player_id: r for r in match.prematch if r.player_id in self._labels}

        starts, end_u = period_layout(match)
        last_period = 2 if end_u > starts[2] else 1
        self._p1_end_s = starts[2]
        self._p2_start_s = starts[2] + break_before_s(2)
        self._end_s = end_u + break_before_s(last_period)
        kickoff_ns = int(pd.Timestamp(lineup.kickoff_utc).value)

        shown = result.displayed
        shown = shown.loc[shown["team_id"] == self._team].copy()
        shown["wall_s"] = (shown["shown_wall_ns"].to_numpy(dtype=np.int64) - kickoff_ns) / _NS
        self._shown = shown
        self._push_s = np.unique(shown["wall_s"].to_numpy(dtype=np.float64))
        self._delay_s = (
            float(shown["wall_s"].iloc[0] - self._tick_wall(float(shown["play_s"].iloc[0])))
            if shown.shape[0]
            else 0.0
        )

        risks = result.risks.loc[result.risks["team_id"] == self._team]
        self._risk: dict[tuple[str, int, float], RiskSnapshot] = {
            (str(r["player_id"]), int(r["period"]), float(r["as_of_t_s"])): _risk_snapshot(r)
            for r in risks.to_dict("records")
        }
        enriched = result.enriched
        enriched = enriched.loc[enriched["player_id"].isin(list(self._labels))]
        self._enriched = {
            (str(r["player_id"]), int(r["match_period"]), float(r["match_t_s"])): r
            for r in enriched.to_dict("records")
        }
        changes = result.alerts.loc[result.alerts["team_id"] == self._team].copy()
        changes["wall_s"] = (changes["shown_wall_ns"].to_numpy(dtype=np.int64) - kickoff_ns) / _NS
        self._changes = [
            (float(r["wall_s"]), str(r["change"]), _alert(r)) for r in changes.to_dict("records")
        ]
        self._decisions: list[DecisionRecord] = []
        self._cursor_s = -60.0
        self._follow: tuple[float, float, float] | None = None

    # ---------------------------------------------------------------- clock and phase

    def _tick_wall(self, play_s: float) -> float:
        return play_s + (break_before_s(2) if play_s > self._p1_end_s + 1e-9 else 0.0)

    def seek(self, wall_s: float) -> None:
        """Freeze the cursor at ``wall_s`` seconds after kick-off on the simulated wall clock."""
        self._follow = None
        self._cursor_s = float(wall_s)

    def play(self, speed: float = 1.0, from_wall_s: float | None = None) -> None:
        """Let the cursor follow real time at ``speed`` times, from ``from_wall_s``."""
        if not speed > 0.0:
            raise ValueError("speed must be > 0")
        start = self._cursor_s if from_wall_s is None else float(from_wall_s)
        self._follow = (start, time.monotonic(), float(speed))

    @property
    def wall_s(self) -> float:
        """Cursor, seconds after kick-off on the simulated wall clock."""
        if self._follow is None:
            return self._cursor_s
        start, t0, speed = self._follow
        return start + (time.monotonic() - t0) * speed

    @property
    def phase(self) -> str:
        w = self.wall_s
        if w < 0.0:
            return "pre_match"
        if self._p1_end_s + self._delay_s < w < self._p2_start_s:
            return "half_time"
        if w > self._end_s + self._delay_s:
            return "post_match"
        return "live"

    # ---------------------------------------------------------------- views

    def _open_alerts(self, w: float) -> tuple[list[Alert], int]:
        open_: dict[str, Alert] = {}
        used = 0
        for wall, change, alert in self._changes:
            if wall > w:
                break
            if change == "cleared":
                open_.pop(alert.alert_id, None)
            else:
                open_[alert.alert_id] = alert
                used += change == "opened"
        return list(open_.values()), used

    def _usual_factor(self, risk: RiskSnapshot) -> float | None:
        """His value over his usual one for the feature named by the reason code."""
        feature = _REASON_FEATURE.get(risk.top_reason_code or "")
        rec = self._enriched.get((risk.player_id, risk.period, risk.as_of_t_s))
        if feature is None or rec is None:
            return None
        raw = rec.get(f"features.{feature}")
        if raw is None or not math.isfinite(float(raw)):
            return None
        band = riskmath.band_index(risk.as_of_t_s / 60.0, risk.period)
        try:
            entry = self._norms.get(risk.player_id, self._positions[risk.player_id], band, feature)
        except KeyError:
            return None
        if feature in self._cfg.norms.log1p_transform:
            usual = math.expm1(entry.mean)
        elif feature in self._cfg.norms.log_transform:
            usual = math.exp(entry.mean)
        else:
            usual = entry.mean
        return float(raw) / usual if usual > 0.0 else None

    def _push_at(self, w: float) -> tuple[float, pd.DataFrame] | None:
        """Arrival time and rows of the latest push on the screen at ``w``."""
        k = int(np.searchsorted(self._push_s, w, side="right")) - 1
        if k < 0:
            return None
        pushed = float(self._push_s[k])
        return pushed, self._shown.loc[self._shown["wall_s"] == pushed]

    def _clock_at(self, w: float) -> tuple[int, float]:
        """Match clock (period, t_s) on the screen at ``w``."""
        found = self._push_at(w)
        if found is None:
            return 1, 0.0
        pushed, push = found
        first = push.iloc[0]
        return int(first["match_period"]), float(first["match_t_s"]) + max(0.0, w - pushed)

    def _match_t_at(self, wall_s: float) -> float:
        """Match clock at a wall time; the half-time break maps to the end of the first half."""
        if wall_s < self._p1_end_s:
            return max(wall_s, 0.0)
        if wall_s < self._p2_start_s:
            return self._p1_end_s
        return 2700.0 + (wall_s - self._p2_start_s)

    def _live(self) -> dict[str, Any]:
        w = self.wall_s
        found = self._push_at(w)
        if found is None:
            return pre_match_table(self._match_id, self._squad_rows())
        pushed, push = found
        alerts, used = self._open_alerts(w)
        level = {a.player_id: a.level.value for a in alerts}
        rows: list[RowInput] = []
        for rec in push.to_dict("records"):
            key = (str(rec["player_id"]), int(rec["risk_period"]), float(rec["risk_as_of_t_s"]))
            risk = self._risk[key]
            enriched = self._enriched.get(key)
            reasons: tuple[str, ...] = ()
            if enriched is not None and isinstance(enriched.get("quality_reasons"), str):
                reasons = tuple(json.loads(str(enriched["quality_reasons"])))
            rows.append(
                RowInput(
                    risk=risk,
                    label=self._labels[risk.player_id],
                    age_s=float(rec["data_age_s"]) + (w - pushed),
                    usual_factor=self._usual_factor(risk),
                    alert_level=level.get(risk.player_id),
                    quality_reasons=reasons,
                )
            )
        period, t_s = self._clock_at(w)
        hide = float(self._cfg.quality.stale_hide_s)
        hidden_ages = [r.age_s for r in rows if r.age_s > hide]
        stale_since = self._match_t_at(w - max(hidden_ages)) if hidden_ages else None
        return live_table(
            match_id=self._match_id,
            period=period,
            t_s=t_s,
            mode=self._cfg.engine.mode,
            rows=rows,
            open_alerts=alerts,
            labels=self._labels,
            budget_used=used,
            cfg=self._cfg,
            stale_since_t_s=stale_since,
        )

    def _squad_rows(self) -> list[tuple[str, str, bool, str, float]]:
        a0 = float(self._cfg.prematch.match_share_prior)
        cut = self._cfg.prematch.band_cutoffs
        rows = []
        for p in self._players:
            risk = self._prematch.get(p.player_id)
            keeper = p.position_group == PositionGroup.GK
            if risk is None or (keeper and not self._cfg.engine.include_goalkeepers):
                continue
            r90 = usual_risk(match_cumhaz(risk, a0))
            band: PrematchBand = prematch_band(r90, float(cut.watch), float(cut.high))
            rows.append((p.player_id, self._labels[p.player_id], p.starting, band.value, r90))
        return rows

    def _half_time(self) -> dict[str, Any]:
        a0 = float(self._cfg.prematch.match_share_prior)
        g = self._fitted.g_levels()
        cumhaz_scale = math.exp(self._fitted.alpha)
        s1 = float(self._cfg.engine.default_stoppage_1h_min)
        s2 = float(self._cfg.engine.default_stoppage_2h_min)
        # His latest first-half snapshot with a usable comparison (stoppage minutes may lack
        # norms, which leaves every z missing there).
        last: dict[str, Mapping[Any, Any]] = {}
        z_cols = [f"z.{f}" for f in self._fitted.feature_names]
        for (pid, period, t_s), rec in sorted(self._enriched.items(), key=lambda kv: kv[0][2]):
            usable = any(c in rec and not pd.isna(rec[c]) for c in z_cols)
            if period == 1 and t_s <= self._p1_end_s and usable:
                last[pid] = rec
        rows = []
        for pid, snap in last.items():
            if self._positions.get(pid) == PositionGroup.GK or pid not in self._prematch:
                continue
            lam = match_cumhaz(self._prematch[pid], a0) * cumhaz_scale
            r45, r20 = (
                riskmath.projected_probability(lam, g, 1.0, 45.0, 2, h, s1, s2)
                for h in _HALF_TIME_HORIZONS_MIN
            )
            z = {
                f: (None if pd.isna(snap.get(f"z.{f}")) else float(snap[f"z.{f}"]))
                for f in self._fitted.feature_names
                if f"z.{f}" in snap
            }
            rows.append((pid, self._labels[pid], z, r45, r20))
        return half_time_table(self._match_id, rows)

    def _post_match(self) -> dict[str, Any]:
        # The last change of an alert carries its final level and, if cleared, the clearing.
        final: dict[str, Alert] = {}
        for _, _, alert in self._changes:
            final[alert.alert_id] = alert
        alerts = [
            (
                a,
                self._labels.get(a.player_id, a.player_id),
                None if a.cleared_t_s is None else clock_text(a.cleared_t_s),
            )
            for a in final.values()
        ]
        count: dict[str, int] = {}
        for a in final.values():
            count[a.player_id] = count.get(a.player_id, 0) + 1
        max_rr: dict[str, float] = {}
        for (pid, _, _), r in self._risk.items():
            max_rr[pid] = max(max_rr.get(pid, 0.0), r.rr)
        players = [
            (self._labels[pid], count.get(pid, 0), max_rr.get(pid))
            for pid in self._labels
            if pid in max_rr
        ]
        return post_match_report(self._match_id, alerts, self._decisions, players)

    def live_table(self, role: str) -> dict[str, Any]:
        """The payload of the current screen for ``role`` (coach: summary only)."""
        phase = self.phase
        if phase == "pre_match":
            payload = pre_match_table(self._match_id, self._squad_rows())
        elif phase == "half_time":
            payload = self._half_time()
        elif phase == "post_match":
            payload = self._post_match()
        else:
            payload = self._live()
        if role == ActorRole.COACH.value:
            return (
                coach_summary(payload)
                if payload["type"] == "live_table"
                else coach_idle(self._match_id)
            )
        return payload

    # ---------------------------------------------------------------- decisions

    @property
    def decisions(self) -> tuple[DecisionRecord, ...]:
        return tuple(self._decisions)

    def record_decision(self, payload: dict[str, Any]) -> DecisionRecord:
        """Validate a staff decision on an alert shown so far and append it to the log store.

        ``payload``: alert_id, actor_role, and either ``button`` (a key of
        ``wording.DECISION_BUTTONS``) or ``action`` with an optional ``note``. The decision is
        stamped with the match clock at the cursor. Raises ``ValueError`` on an unknown alert,
        role, button or action.
        """
        w = self.wall_s
        shown_ids = {a.alert_id for wall, _, a in self._changes if wall <= w}
        alert_id = str(payload.get("alert_id", ""))
        if alert_id not in shown_ids:
            raise ValueError(f"no alert {alert_id!r} has been shown to this team")
        button = payload.get("button")
        if button is not None:
            if button not in DECISION_BUTTONS:
                raise ValueError(f"unknown button {button!r}")
            action, note = DECISION_BUTTONS[str(button)]
        else:
            action, note = str(payload.get("action", "")), str(payload.get("note", ""))
        _, t_s = self._clock_at(w)
        record = DecisionRecord(
            alert_id=alert_id,
            t_s=t_s,
            actor_role=ActorRole(str(payload.get("actor_role", ""))),
            action=Action(action),
            note=note,
        )
        self._decisions.append(record)
        if self._store is not None:
            self._store.append(
                "decisions", [record], team_id=[self._team], shown_wall_s=[round(w, 3)]
            )
            self._store.flush()
        return record
