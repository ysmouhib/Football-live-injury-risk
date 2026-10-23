"""All user-facing strings (docs/06_UI_SPEC.md §5-§6). Nothing user-facing is written elsewhere.

``tests/test_wording.py`` checks every string here against ``FORBIDDEN_PHRASES``; M11 extends the
check to payloads produced during a replayed match.
"""

from __future__ import annotations

import re

#: Reason code -> short text (<= 36 characters after formatting). Placeholders: {k}, {m}.
REASON_TEXT: dict[str, str] = {
    "HSR_HIGH": "High-speed metres {k}× his usual",
    "SPR_HIGH": "Sprint metres {k}× his usual",
    "RECOVERY_SHORT": "Short recovery between sprints",
    "HIR_DROP": "Running drop larger than usual",
    "NEARMAX_MANY": "Many near-top-speed sprints",
    "DISRUPT": "Just after a card, goal or stoppage",
    "TIME_PROFILE": "Higher-risk spell of the match",
    "WITHIN_NORMAL": "Within his normal range",
    "LOAD_BELOW": "Sprint load below his usual",
    "DATA_LOST": "Live data lost {m} min ago",
}

#: Agreed actions shown on alert cards.
ACTION_TEXT: dict[str, str] = {
    "check": "Physio observes at the next stoppage",
    "discuss": "Staff discuss options",
    "possible_injury": "Medical staff look now",
}

LEVEL_TEXT: dict[str, str] = {
    "check": "Check",
    "discuss": "Discuss",
    "possible_injury": "Possible injury event",
}

QUALITY_TEXT: dict[str, str] = {
    "good": "Good",
    "degraded": "Fair",
    "prematch_only": "Lost",
    "stale": "Stale",
}

BAND_TEXT: dict[str, str] = {"low": "Low", "watch": "Watch", "high": "High"}

#: Quality reason code -> short text shown beside the quality chip (<= 40 characters).
QUALITY_REASON_TEXT: dict[str, str] = {
    "low_coverage": "Part of the window has no data",
    "identity_low": "Player identity uncertain",
    "norm_reset": "Compared with position norm",
    "few_matches": "Few past matches for his norm",
    "clock_drift": "Clock out of sync, resyncing",
}

#: Help texts (info buttons).
HELP_TEXT: dict[str, str] = {
    "interval": (
        "The range shows model uncertainty, and pre-match uncertainty when the weekly model "
        "supplies it. Norm uncertainty is not included."
    ),
    "r_rest": "Risk for the rest of the match if he stays on, compared with his usual state.",
    "norm_reset": (
        "Back from an absence: his own norm is rebuilt over three matches, so he is compared "
        "mainly with his position norm."
    ),
}

#: Reason texts when the "his usual" factor cannot be formed (no usable norm for the value).
REASON_TEXT_NO_FACTOR: dict[str, str] = {
    "HSR_HIGH": "High-speed metres above his usual",
    "SPR_HIGH": "Sprint metres above his usual",
}

#: Position group -> label used after the shirt number ("#14 Full back").
POSITION_TEXT: dict[str, str] = {
    "GK": "Goalkeeper",
    "CB": "Centre back",
    "FB": "Full back",
    "MF": "Midfielder",
    "W": "Winger",
    "FW": "Forward",
}

#: Live feature -> plain name for "his value vs his usual" lines.
FEATURE_TEXT: dict[str, str] = {
    "hsr_5m": "High-speed metres, last 5 min",
    "spr_5m": "Sprint metres, last 5 min",
    "recovery_10m": "Recovery between efforts",
    "hir_drop_15m": "Running vs his first 15 min",
    "nearmax_cum": "Near-top-speed efforts",
    "disrupt_5m": "Cards, goals, stoppages",
}

PERIOD_TEXT: dict[int, str] = {
    1: "first half",
    2: "second half",
    3: "extra time, first half",
    4: "extra time, second half",
}

#: Screen headings, column titles, buttons and status lines of the bench app.
UI_TEXT: dict[str, str] = {
    "title": "Live load and risk",
    "screen_pre_match": "Pre-match",
    "screen_live": "Live",
    "screen_half_time": "Half-time",
    "screen_post_match": "Post-match",
    "col_player": "Player",
    "col_prematch": "Pre-match",
    "col_now": "Now vs his normal",
    "col_rest": "Rest of match",
    "col_why": "Why",
    "col_data": "Data",
    "col_usual_90": "Usual risk, 90 min",
    "col_starting": "Starting",
    "col_load": "First half vs his usual",
    "col_second_45": "Second half, 45 min",
    "col_second_20": "Second half, 20 min",
    "col_alerts": "Alerts",
    "col_max_rr": "Highest vs his normal",
    "col_decision": "Staff decision",
    "col_level": "Level",
    "col_time": "Opened - cleared",
    "header_data_age": "Live data {s} s old",
    "header_alerts": "Alerts used: {used} of {budget}",
    "row_age": "{s} s old",
    "banner_stale": "Data stale since {clock}",
    "monitor_only": "Load against his own norm only; live risk figures are off in this mode.",
    "alert_numbers": "Rest-of-match risk {rest} against {usual} usual.",
    "load_line": "{feature}: {z} SD vs his usual",
    "rr_up": "×{rr} ▲",
    "rr_down": "×{rr} ▼",
    "rr_level": "×{rr}",
    "percent": "{value} %",
    "percent_below": "< {value} %",
    "blank": "-",
    "detail_p15": "Next 15 min",
    "detail_usual": "Rest of match, usual",
    "detail_interval": "Range",
    "button_ack": "Acknowledge",
    "button_physio": "Physio checked",
    "button_discussed": "Discussed",
    "button_dismiss": "Dismiss",
    "connection_live": "Connected",
    "connection_lost": "Connection lost, reconnecting",
    "starting_yes": "Yes",
    "starting_no": "Bench",
    "none_open": "No open alerts",
    "decision_saved": "Decision saved",
}

#: Alert-card buttons -> (DecisionRecord action, note).
DECISION_BUTTONS: dict[str, tuple[str, str]] = {
    "button_ack": ("none", "acknowledged"),
    "button_physio": ("physio_check", ""),
    "button_discussed": ("discussed", ""),
    "button_dismiss": ("none", "dismissed"),
}

#: Fixed sentence on every alert card.
ALERT_DISCLAIMER: str = "Not a prediction of injury; staff decide."

#: Phrases that must never appear in any user-facing output (case-insensitive, whole words).
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "will be injured",
    "is going to get injured",
    "injury predicted",
    "predicts injury",
    "safe",
    "no risk",
    "substitute now",
    "take him off",
    "must come off",
    "danger",
    "red zone",
    "guaranteed",
)

_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in FORBIDDEN_PHRASES) + r")\b", re.IGNORECASE
)


def find_forbidden(text: str) -> list[str]:
    """Return forbidden phrases found in ``text`` (empty list if clean)."""
    return [m.group(0) for m in _FORBIDDEN_RE.finditer(text)]


def all_strings() -> list[str]:
    """Every user-facing string defined in this module."""
    out: list[str] = [ALERT_DISCLAIMER]
    for table in (
        REASON_TEXT,
        REASON_TEXT_NO_FACTOR,
        ACTION_TEXT,
        LEVEL_TEXT,
        QUALITY_TEXT,
        BAND_TEXT,
        QUALITY_REASON_TEXT,
        HELP_TEXT,
        POSITION_TEXT,
        FEATURE_TEXT,
        PERIOD_TEXT,
        UI_TEXT,
    ):
        out.extend(table.values())
    out.extend(note for _, note in DECISION_BUTTONS.values() if note)
    return out
