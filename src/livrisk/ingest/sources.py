"""Concrete adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from livrisk.contracts import Lineup, MatchEvent, Source


def read_tracking_table(
    path: str | Path,
    column_map: dict[str, str],
    source: Source,
    match_id: str,
) -> pd.DataFrame:
    """CSV/Parquet export (e.g. wearable vendor) -> TrackingSample columns via ``column_map``."""
    raise NotImplementedError("planned for M12")


def load_skillcorner_match(
    match_dir: str | Path,
) -> tuple[pd.DataFrame, list[MatchEvent], tuple[Lineup, ...]]:
    """SkillCorner open-data match (10 fps): ``is_detected`` -> ``detected``; source=broadcast."""
    raise NotImplementedError("planned for M12")


def read_events_table(
    path: str | Path, column_map: dict[str, str], match_id: str
) -> list[MatchEvent]:
    """Official-feed events export -> MatchEvent list."""
    raise NotImplementedError("planned for M12")
