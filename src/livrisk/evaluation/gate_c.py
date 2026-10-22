"""Gate C runner and report (docs/05_VALIDATION.md §2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from livrisk.config import Config


@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateCReport:
    criteria: tuple[CriterionResult, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.criteria)

    def to_markdown(self) -> str:
        raise NotImplementedError("planned for M10")


def run_gate_c(
    person_minutes: pd.DataFrame,
    feature_names: Sequence[str],
    cfg: Config,
    split: str = "time",
    seed: int = 0,
) -> GateCReport:
    """Fit M0/M1/M2 (+ benchmarks) inside each training fold; evaluate on test folds."""
    raise NotImplementedError("planned for M10")
