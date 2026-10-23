"""Screen payloads of the bench app (docs/06_UI_SPEC.md §1-§5, §8).

Pure functions from engine outputs to JSON-ready dicts. Every user-facing string comes from
``wording.py``. The stale rules of docs/06 §4 are applied here, on the server, from the age a
value has when the payload is built: above ``quality.stale_grey_s`` a row is greyed with its age,
above ``quality.stale_hide_s`` its numbers are removed and the header carries the stale banner.
The ``coach`` view carries only names, alert level and the one-line reason.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from livrisk.app.wording import (
    ACTION_TEXT,
    ALERT_DISCLAIMER,
    BAND_TEXT,
    DECISION_BUTTONS,
    FEATURE_TEXT,
    LEVEL_TEXT,
    PERIOD_TEXT,
    POSITION_TEXT,
    QUALITY_REASON_TEXT,
    QUALITY_TEXT,
    REASON_TEXT,
    REASON_TEXT_NO_FACTOR,
    UI_TEXT,
)
from livrisk.config import Config
from livrisk.contracts import Alert, DecisionRecord, Mode, Quality, RiskSnapshot
from livrisk.replay.simulator import GREY, HIDDEN, display_state

#: Top-level keys and alert keys a coach payload may carry (docs/04 §8, docs/06 §7).
COACH_KEYS: frozenset[str] = frozenset(
    {"type", "match_id", "period", "period_text", "clock", "alerts"}
)
COACH_ALERT_KEYS: frozenset[str] = frozenset({"label", "level", "level_text", "reason_text"})
#: Numeric risk fields of a live row; removed when the row is hidden.
NUMBER_FIELDS: tuple[str, ...] = ("rr", "p15", "r_rest", "r_rest_lo", "r_rest_hi", "r_usual")
#: Smallest probability shown as a number (0.05 %, which rounds to 0.1 %).
_SMALLEST_SHOWN: float = 0.0005
#: RR at or beyond these bounds shows an arrow (docs/06 §2: "x1.9 up", "x0.8 down").
_RR_UP, _RR_DOWN = 1.05, 0.95


def player_label(shirt_number: int, position: str) -> str:
    """ "#14 Full back": shirt number and position group, never a name inside the engine."""
    return f"#{shirt_number} {POSITION_TEXT.get(position, position)}"


def clock_text(t_s: float) -> str:
    """Match clock as minutes:seconds ("67:12") from continuous match seconds."""
    total = max(0, int(t_s))
    return f"{total // 60}:{total % 60:02d}"


def percent_text(p: float | None) -> str:
    """ "2.1 %"; values that would round to 0.0 % show as "< 0.1 %", never as zero risk."""
    if p is None or not math.isfinite(p):
        return UI_TEXT["blank"]
    if p < _SMALLEST_SHOWN:
        return UI_TEXT["percent_below"].format(value="0.1")
    return UI_TEXT["percent"].format(value=f"{100.0 * p:.1f}")


def rr_text(rr: float | None) -> str:
    if rr is None:
        return UI_TEXT["blank"]
    shown = f"{rr:.1f}"
    if rr >= _RR_UP:
        return UI_TEXT["rr_up"].format(rr=shown)
    if rr <= _RR_DOWN:
        return UI_TEXT["rr_down"].format(rr=shown)
    return UI_TEXT["rr_level"].format(rr=shown)


def reason_text(code: str | None, factor: float | None, age_s: float) -> str:
    """Wording of a reason code (docs/06 §5); ``factor`` is his value over his usual one."""
    if code is None:
        code = "WITHIN_NORMAL"
    if code == "DATA_LOST":
        return REASON_TEXT[code].format(m=max(1, round(age_s / 60.0)))
    if "{k}" in REASON_TEXT.get(code, ""):
        if factor is None or not math.isfinite(factor) or factor <= 0.0:
            return REASON_TEXT_NO_FACTOR.get(code, REASON_TEXT["WITHIN_NORMAL"])
        return REASON_TEXT[code].format(k=f"{factor:.1f}")
    return REASON_TEXT.get(code, REASON_TEXT["WITHIN_NORMAL"])


@dataclass(frozen=True)
class RowInput:
    """One on-pitch player's latest risk row with its current age (seconds)."""

    risk: RiskSnapshot
    label: str
    age_s: float
    usual_factor: float | None = None
    alert_level: str | None = None
    quality_reasons: tuple[str, ...] = ()
    #: Qualifies for a card but the team's budget is spent (docs/06 §3: a marker, no card).
    muted: bool = False


def live_row(row: RowInput, cfg: Config) -> dict[str, Any]:
    """A live-table row (docs/06 §2, §8) with the stale rules applied to ``row.age_s``."""
    r = row.risk
    q = cfg.quality
    state = display_state(row.age_s, float(q.stale_grey_s), float(q.stale_hide_s))
    hidden = state == HIDDEN
    quality = Quality.STALE.value if hidden else r.quality.value
    no_rr = (
        hidden or r.mode == Mode.MONITOR_ONLY or r.quality in (Quality.PREMATCH_ONLY, Quality.STALE)
    )
    code = "DATA_LOST" if hidden else r.top_reason_code
    out: dict[str, Any] = {
        "player_id": r.player_id,
        "label": row.label,
        "prematch_band": r.prematch_band.value,
        "prematch_band_text": BAND_TEXT[r.prematch_band.value],
        "rr": None if no_rr else r.rr,
        "r_rest": None if hidden else r.r_rest,
        "r_rest_lo": None if hidden else r.r_rest_lo,
        "r_rest_hi": None if hidden else r.r_rest_hi,
        "r_usual": None if hidden else r.r_usual,
        "p15": None if hidden else r.p15,
        "reason_code": code,
        "reason_text": reason_text(code, row.usual_factor, row.age_s),
        "quality": quality,
        "quality_text": QUALITY_TEXT[quality],
        "quality_reasons": [
            QUALITY_REASON_TEXT[c] for c in row.quality_reasons if c in QUALITY_REASON_TEXT
        ],
        "data_age_s": round(row.age_s, 1),
        "state": state,
        "alert_level": row.alert_level,
        "muted": row.muted and row.alert_level is None,
    }
    out["rr_text"] = rr_text(out["rr"])
    out["r_rest_text"] = percent_text(out["r_rest"])
    out["r_usual_text"] = percent_text(out["r_usual"])
    out["p15_text"] = percent_text(out["p15"])
    out["interval_text"] = (
        UI_TEXT["blank"] if hidden else f"{percent_text(r.r_rest_lo)} - {percent_text(r.r_rest_hi)}"
    )
    out["muted_text"] = UI_TEXT["row_muted"] if out["muted"] else ""
    out["age_text"] = UI_TEXT["row_age"].format(s=round(row.age_s)) if state == GREY else ""
    return out


@dataclass(frozen=True)
class AlertInput:
    """An open alert with its player's label and current live row (None when off the table)."""

    alert: Alert
    label: str
    row: dict[str, Any] | None


def alert_card(a: AlertInput) -> dict[str, Any]:
    """Alert card (docs/06 §2): level, player, agreed action, one line of numbers, disclaimer."""
    level = a.alert.level.value
    row = a.row
    numbers = UI_TEXT["blank"]
    if row is not None and row["r_rest"] is not None:
        numbers = UI_TEXT["alert_numbers"].format(
            rest=row["r_rest_text"], usual=row["r_usual_text"]
        )
    return {
        "alert_id": a.alert.alert_id,
        "player_id": a.alert.player_id,
        "label": a.label,
        "level": level,
        "level_text": LEVEL_TEXT[level],
        "reason_code": a.alert.reason_code,
        "reason_text": row["reason_text"]
        if row is not None
        else reason_text(a.alert.reason_code, None, 0.0),
        "action_text": ACTION_TEXT[level],
        "numbers_text": numbers,
        "disclaimer": ALERT_DISCLAIMER,
        "buttons": [{"key": k, "text": UI_TEXT[k]} for k in DECISION_BUTTONS],
    }


def live_table(
    *,
    match_id: str,
    period: int,
    t_s: float,
    mode: str,
    rows: Sequence[RowInput],
    open_alerts: Sequence[Alert],
    labels: dict[str, str],
    budget_used: int,
    cfg: Config,
    stale_since_t_s: float | None = None,
) -> dict[str, Any]:
    """Live table payload (docs/06 §8): rows sorted by r_rest descending, hidden rows last.

    ``stale_since_t_s`` is the match clock when the oldest hidden row's data stopped; the caller
    knows the breaks between periods. Without it, it is ``t_s`` minus that row's age.
    """
    shaped = [live_row(r, cfg) for r in rows]
    shaped.sort(key=lambda r: (r["r_rest"] is None, -(r["r_rest"] or 0.0), r["player_id"]))
    by_player = {r["player_id"]: r for r in shaped}
    ages = [r.age_s for r in rows]
    header_age = min(ages) if ages else None
    hidden = [
        r
        for r in rows
        if display_state(r.age_s, float(cfg.quality.stale_grey_s), float(cfg.quality.stale_hide_s))
        == HIDDEN
    ]
    banner = None
    if hidden:
        since = stale_since_t_s
        if since is None:
            since = t_s - max(r.age_s for r in hidden)
        # Nearest second: at 10 Hz the feed stops mid-second, and flooring would place the
        # banner one second before the last live one.
        banner = UI_TEXT["banner_stale"].format(clock=clock_text(float(round(since))))
    budget = int(cfg.alerts.budget_per_match)
    return {
        "type": "live_table",
        "match_id": match_id,
        "period": period,
        "period_text": PERIOD_TEXT.get(period, ""),
        "clock": clock_text(t_s),
        "data_age_s": None if header_age is None else round(header_age, 1),
        "data_age_text": ""
        if header_age is None
        else UI_TEXT["header_data_age"].format(s=round(header_age)),
        "alerts_used": budget_used,
        "alerts_budget": budget,
        "alerts_text": UI_TEXT["header_alerts"].format(used=budget_used, budget=budget),
        "mode": mode,
        "mode_note": UI_TEXT["monitor_only"] if mode == Mode.MONITOR_ONLY.value else "",
        "banner": banner,
        "rows": shaped,
        "alerts": [
            alert_card(
                AlertInput(a, labels.get(a.player_id, a.player_id), by_player.get(a.player_id))
            )
            for a in open_alerts
        ],
    }


def coach_summary(live: dict[str, Any]) -> dict[str, Any]:
    """The coach's view of a live table: names, alert level and the one-line reason only."""
    return {
        "type": "coach_summary",
        "match_id": live["match_id"],
        "period": live["period"],
        "period_text": live["period_text"],
        "clock": live["clock"],
        "alerts": [{k: a[k] for k in COACH_ALERT_KEYS} for a in live["alerts"]],
    }


def coach_idle(match_id: str) -> dict[str, Any]:
    """The coach's view outside play: no alerts."""
    return {
        "type": "coach_summary",
        "match_id": match_id,
        "period": 0,
        "period_text": "",
        "clock": "",
        "alerts": [],
    }


def pre_match_table(
    match_id: str, rows: Sequence[tuple[str, str, bool, str, float]]
) -> dict[str, Any]:
    """Squad list before kick-off: (player_id, label, starting, band, r_usual over 90 min)."""
    shaped = [
        {
            "player_id": pid,
            "label": label,
            "starting": starting,
            "starting_text": UI_TEXT["starting_yes" if starting else "starting_no"],
            "prematch_band": band,
            "prematch_band_text": BAND_TEXT[band],
            "r_usual_90": r90,
            "r_usual_90_text": percent_text(r90),
        }
        for pid, label, starting, band, r90 in sorted(rows, key=lambda r: (not r[2], -r[4], r[0]))
    ]
    return {"type": "pre_match", "match_id": match_id, "rows": shaped}


def load_lines(z: dict[str, float | None]) -> list[str]:
    """ "His value vs his usual" lines from player-relative z values (SD units)."""
    return [
        UI_TEXT["load_line"].format(
            feature=FEATURE_TEXT.get(name, name), z=f"{round(value, 1) + 0.0:+.1f}"
        )
        for name, value in z.items()
        if value is not None and math.isfinite(value)
    ]


def half_time_table(
    match_id: str, rows: Sequence[tuple[str, str, dict[str, float | None], float, float]]
) -> dict[str, Any]:
    """Between halves: (player_id, label, first-half z by feature, usual risk over 45 and 20
    more minutes of the second half)."""
    shaped = [
        {
            "player_id": pid,
            "label": label,
            "load_lines": load_lines(z),
            "r_45": r45,
            "r_45_text": percent_text(r45),
            "r_20": r20,
            "r_20_text": percent_text(r20),
        }
        for pid, label, z, r45, r20 in sorted(rows, key=lambda r: (-r[3], r[0]))
    ]
    return {"type": "half_time", "match_id": match_id, "rows": shaped}


def decision_text(record: DecisionRecord) -> str:
    """The button a decision came from, else its action code."""
    for key, (action, note) in DECISION_BUTTONS.items():
        if record.action.value == action and record.note == note:
            return UI_TEXT[key]
    return record.action.value


def post_match_report(
    match_id: str,
    alerts: Sequence[tuple[Alert, str, str | None]],
    decisions: Sequence[DecisionRecord],
    players: Sequence[tuple[str, int, float | None]],
) -> dict[str, Any]:
    """After the match: alerts as (final alert, label, cleared clock), the decisions taken on
    them, and per player (label, alerts, highest RR shown)."""
    by_alert: dict[str, list[str]] = {}
    for d in decisions:
        by_alert.setdefault(d.alert_id, []).append(decision_text(d))
    return {
        "type": "post_match",
        "match_id": match_id,
        "alerts": [
            {
                "alert_id": a.alert_id,
                "label": label,
                "level_text": LEVEL_TEXT[a.level.value],
                "reason_text": reason_text(a.reason_code, None, 0.0),
                "opened": clock_text(a.opened_t_s),
                "cleared": cleared or UI_TEXT["blank"],
                "decisions": by_alert.get(a.alert_id, []),
            }
            for a, label, cleared in alerts
        ],
        "players": [
            {"label": label, "alerts": n, "max_rr_text": rr_text(max_rr)}
            for label, n, max_rr in players
        ],
        "none_open": UI_TEXT["none_open"] if not alerts else "",
    }


def collect_strings(payload: object) -> list[str]:
    """Every string anywhere in a payload (for the forbidden-wording check)."""
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        return [s for v in payload.values() for s in collect_strings(v)]
    if isinstance(payload, list | tuple):
        return [s for v in payload for s in collect_strings(v)]
    return []
