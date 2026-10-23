"""Pre-registered Gate C thresholds (docs/05_VALIDATION.md §2). Changing one needs a new
pre-registration, not a config edit, so they are constants here rather than tunables."""

from __future__ import annotations

#: Criterion 2: O/E of M2 in every test season (and of M1 in the input attribution).
OE_RANGE: tuple[float, float] = (0.8, 1.25)
#: Criterion 4: O/E of M2 in every subgroup (position, club).
SUBGROUP_OE_RANGE: tuple[float, float] = (0.6, 1.6)
#: Criterion 3: alert budget, alerts per team-match (flag share = budget / 66 spells).
ALERT_BUDGET_PER_TEAM_MATCH: float = 1.5
#: Level of the bootstrap intervals behind every criterion.
CI_LEVEL: float = 0.95
#: Level of the beta intervals in the tipping-point analysis (docs/02_MATH.md §4).
TIPPING_LEVEL: float = 0.90
