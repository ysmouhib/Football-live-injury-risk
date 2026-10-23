"""Append-only log store (Parquet + DuckDB views; docs/03_DATA_CONTRACTS.md §9).

Layout: ``{root}/{match_date}/{match_id}/{kind}/part-{n:05d}.parquet``. Every ``flush`` writes
one new part file per kind with pending rows. Part files are created exclusively and never
opened for writing again, so nothing is rewritten and a crash loses at most the rows not yet
flushed. Every row carries the ``config_hash`` of the run (``config.config_hash``).

Rows are stored flat: a mapping field becomes one column per key (``features.hsr_5m``,
``z.hsr_5m``, ...) and a sequence field one JSON text column, so part files written at different
times share column types. ``read`` concatenates the parts of a kind in order; ``sql`` runs a
DuckDB query over one view per kind (quote dotted column names: ``"z.hsr_5m"``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType

import duckdb
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


def _flat_record(model: BaseModel) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in model.model_dump(mode="json").items():
        if isinstance(value, Mapping):
            for sub, inner in value.items():
                out[f"{key}.{sub}"] = inner
        elif isinstance(value, list | tuple):
            out[key] = json.dumps(list(value))
        else:
            out[key] = value
    return out


def records_frame(rows: Sequence[BaseModel], **columns: Sequence[object]) -> pd.DataFrame:
    """Contract rows as a flat frame (see the module docstring); ``columns`` adds per-row
    columns (one value per row)."""
    frame = pd.DataFrame([_flat_record(r) for r in rows])
    for name, values in columns.items():
        if len(values) != len(rows):
            raise ValueError(f"column {name!r} has {len(values)} values for {len(rows)} rows")
        frame[name] = pd.Series(list(values), index=frame.index)
    return frame


class LogStore:
    """Writes ``{root}/{match_date}/{match_id}/{kind}/part-NNNNN.parquet``; never rewrites.

    ``match_date`` defaults to today's UTC date. Reopening the store of a match continues the
    part numbering after the files already present.
    """

    def __init__(
        self,
        root: str | Path,
        match_id: str,
        *,
        config_hash: str,
        match_date: date | None = None,
    ) -> None:
        day = match_date or datetime.now(UTC).date()
        self._dir = Path(root) / day.isoformat() / match_id
        self._config_hash = config_hash
        self._pending: dict[str, list[pd.DataFrame]] = {kind: [] for kind in KINDS}

    @property
    def directory(self) -> Path:
        """Folder of this match's logs."""
        return self._dir

    @property
    def config_hash(self) -> str:
        """The config hash stamped on every row this store writes."""
        return self._config_hash

    def _check_kind(self, kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown log kind {kind!r}; expected one of {KINDS}")

    def append(self, kind: str, rows: Sequence[BaseModel], **columns: Sequence[object]) -> None:
        """Queue contract rows (with optional extra columns, see ``records_frame``)."""
        self._check_kind(kind)
        if rows:
            self._pending[kind].append(records_frame(rows, **columns))

    def append_frame(self, kind: str, frame: pd.DataFrame) -> None:
        """Queue a table that is already flat (e.g. tracking samples with contract columns)."""
        self._check_kind(kind)
        if frame.shape[0]:
            self._pending[kind].append(frame.copy())

    def _parts(self, kind: str) -> list[Path]:
        folder = self._dir / kind
        return sorted(folder.glob("part-*.parquet")) if folder.is_dir() else []

    def flush(self) -> None:
        """Write one new part file per kind with pending rows."""
        for kind in KINDS:
            frames = self._pending[kind]
            if not frames:
                continue
            frame = pd.concat(frames, ignore_index=True)
            frame["config_hash"] = self._config_hash
            folder = self._dir / kind
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"part-{len(self._parts(kind)):05d}.parquet"
            with path.open("xb") as fh:
                frame.to_parquet(fh, index=False)
            self._pending[kind] = []

    def read(self, kind: str) -> pd.DataFrame:
        """Every flushed row of ``kind`` in writing order (empty frame when there is none)."""
        self._check_kind(kind)
        parts = [pd.read_parquet(p) for p in self._parts(kind)]
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def sql(self, query: str) -> pd.DataFrame:
        """Run a DuckDB query; each kind with flushed rows is a view of that name."""
        con = duckdb.connect()
        try:
            for kind in KINDS:
                frame = self.read(kind)
                if frame.shape[0]:
                    con.register(kind, frame)
            result: pd.DataFrame = con.execute(query).df()
            return result
        finally:
            con.close()

    def __enter__(self) -> LogStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.flush()
