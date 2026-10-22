"""Alert state machine (docs/06_UI_SPEC.md §3).

Deterministic: persistence, hysteresis, budget, priority by r_rest, Check -> Discuss upgrade
without extra budget, clearing on substitution, no Check/Discuss alerts in monitor_only mode.
"""

from __future__ import annotations

from collections.abc import Sequence

from livrisk.config import Config
from livrisk.contracts import Alert, DecisionRecord, RiskSnapshot


class AlertLogic:
    def __init__(self, cfg: Config, match_id: str) -> None:
        raise NotImplementedError("planned for M8")

    def update(self, risks: Sequence[RiskSnapshot], on_pitch: set[str]) -> list[Alert]:
        """Return alerts opened, upgraded or cleared in this update (in that order)."""
        raise NotImplementedError("planned for M8")

    def open_alerts(self) -> list[Alert]:
        raise NotImplementedError("planned for M8")

    @property
    def budget_used(self) -> int:
        raise NotImplementedError("planned for M8")

    def record_decision(self, record: DecisionRecord) -> None:
        raise NotImplementedError("planned for M8")
