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
        ACTION_TEXT,
        LEVEL_TEXT,
        QUALITY_TEXT,
        BAND_TEXT,
        QUALITY_REASON_TEXT,
        HELP_TEXT,
    ):
        out.extend(table.values())
    return out
