"""Pre-match bridge (docs/02_MATH.md §1-§2; docs/03 §5).

Consumes the weekly model's output; never re-implements formulas (use ``riskmath``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from livrisk import riskmath
from livrisk.contracts import PrematchBand, PreMatchRisk


def load_prematch_csv(path: str | Path, match_date: date) -> dict[str, PreMatchRisk]:
    """One row per player. Rejects ``as_of_date >= match_date`` and duplicate players.

    Columns are exactly the ``PreMatchRisk`` fields (docs/03 §5); optional interval fields may be
    empty. ``match_date`` is the match date the file is used for (look-ahead guard).
    """
    frame = pd.read_csv(path)
    out: dict[str, PreMatchRisk] = {}
    for row in frame.to_dict(orient="records"):
        record = {k: v for k, v in row.items() if not pd.isna(v)}
        risk = PreMatchRisk.model_validate(record)
        if risk.as_of_date >= match_date:
            raise ValueError(
                f"as_of_date {risk.as_of_date} must be before the match date {match_date} "
                f"(player {risk.player_id})"
            )
        if risk.player_id in out:
            raise ValueError(f"duplicate pre-match row for player {risk.player_id}")
        out[risk.player_id] = risk
    return out


def match_cumhaz(risk: PreMatchRisk, match_share: float) -> float:
    """Lambda for 90 regular minutes via ``riskmath.weekly_to_match_cumhaz``."""
    return riskmath.weekly_to_match_cumhaz(risk.p7, match_share)


def usual_risk(cumhaz: float, expected_minutes: float = riskmath.REGULAR_MINUTES) -> float:
    """P(non-contact injury) over ``expected_minutes`` at the usual state (g averages 1)."""
    if not 0.0 < expected_minutes <= 120.0:
        raise ValueError(f"expected_minutes must be in (0, 120], got {expected_minutes}")
    return riskmath.prob_from_cumhaz(cumhaz * expected_minutes / riskmath.REGULAR_MINUTES)


def prematch_band(r_usual_90: float, watch: float, high: float) -> PrematchBand:
    """Display band from the 90-minute usual risk and the configured cut-offs."""
    if not 0.0 < watch < high:
        raise ValueError("band cut-offs must satisfy 0 < watch < high")
    if r_usual_90 < watch:
        return PrematchBand.LOW
    if r_usual_90 < high:
        return PrematchBand.WATCH
    return PrematchBand.HIGH
