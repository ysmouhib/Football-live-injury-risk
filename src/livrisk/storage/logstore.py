"""Append-only log store (Parquet + DuckDB views). Used from M9 on (docs/03 §9)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

KINDS: tuple[str, ...] = (
    "tracking",
    "events",
    "features",
    "risk",
    "alerts",
    "decisions",
    "display",
)


class LogStore:
    """Writes ``logs/{date}/{match_id}/{kind}.parquet``; never rewrites existing rows."""

    def __init__(self, root: str | Path, match_id: str) -> None:
        raise NotImplementedError("planned for M9")

    def append(self, kind: str, rows: Sequence[BaseModel]) -> None:
        raise NotImplementedError("planned for M9")

    def flush(self) -> None:
        raise NotImplementedError("planned for M9")

    def read(self, kind: str) -> pd.DataFrame:
        raise NotImplementedError("planned for M9")
