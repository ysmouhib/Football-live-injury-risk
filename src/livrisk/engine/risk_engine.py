"""Online risk engine (docs/02_MATH.md §3, §5).

Every ``engine.risk_tick_s`` seconds: feature snapshots + fitted parameters + pre-match risks ->
``RiskSnapshot`` per player on the pitch, with intervals, quality and data age.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from livrisk.config import Config
from livrisk.contracts import FeatureSnapshot, PreMatchRisk, RiskSnapshot
from livrisk.model.live_hazard import FittedHazard


@dataclass(frozen=True)
class MatchState:
    """Match clock and announced stoppage (None until announced)."""

    match_id: str
    period: int
    t_s: float
    stoppage_1h_min: float | None = None
    stoppage_2h_min: float | None = None


class RiskEngine:
    """Combines pre-match baseline, time profile and live multiplier (docs/02_MATH.md §5)."""

    def __init__(
        self, fitted: FittedHazard, prematch: dict[str, PreMatchRisk], cfg: Config
    ) -> None:
        raise NotImplementedError("planned for M7")

    def update(
        self,
        snapshots: Sequence[FeatureSnapshot],
        state: MatchState,
        data_age_s: dict[str, float],
    ) -> list[RiskSnapshot]:
        """Deterministic: posterior draws seeded from (match_id, player_id, tick)."""
        raise NotImplementedError("planned for M7")
