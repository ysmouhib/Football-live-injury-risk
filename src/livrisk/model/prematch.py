"""Pre-match bridge (docs/02_MATH.md §1-§2; docs/03 §5).

Consumes the weekly model's output; never re-implements formulas (use ``riskmath``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from livrisk.contracts import PrematchBand, PreMatchRisk


def load_prematch_csv(path: str | Path, match_date: date) -> dict[str, PreMatchRisk]:
    """One row per player. Rejects ``as_of_date >= match_date`` and duplicate players."""
    raise NotImplementedError("planned for M5")


def match_cumhaz(risk: PreMatchRisk, match_share: float) -> float:
    """Lambda for 90 regular minutes via ``riskmath.weekly_to_match_cumhaz``."""
    raise NotImplementedError("planned for M5")


def prematch_band(r_usual_90: float, watch: float, high: float) -> PrematchBand:
    """Display band from the 90-minute usual risk and the configured cut-offs."""
    raise NotImplementedError("planned for M5")
