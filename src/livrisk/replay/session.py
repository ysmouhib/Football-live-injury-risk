"""Replay sessions: serve a finished replay from its log store (docs/04_ARCHITECTURE.md §7).

A replay run with a log store also writes, at the top level of the match's log folder:

- ``fitted.json``: the fitted hazard (``model.live_hazard.FittedHazard`` JSON),
- ``context.json``: lineups, pre-match risks, peak speeds, the config hash of the run and the
  last replayed match-clock reading,
- ``history.parquet``: the norm history the replay's norms were built from (optional).

``load_session`` rebuilds the replay result frames from the store kinds (``display``, ``risk``,
``alerts`` and the risk-tick rows of ``features``) and the match record from the manifest, so
the bench app can be served without re-running the pipeline. The stored tracking rows are not
read back: serving needs the produced outputs, not the inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from livrisk.config import Config
from livrisk.contracts import PERIOD_START_S, Lineup, MatchEvent, PreMatchRisk
from livrisk.features.norms import NormTable, build_norms
from livrisk.model.live_hazard import FittedHazard
from livrisk.replay.simulator import MatchRecord, ReplayResult, period_layout
from livrisk.storage.logstore import LogStore

FITTED_FILE = "fitted.json"
CONTEXT_FILE = "context.json"
HISTORY_FILE = "history.parquet"


@dataclass(frozen=True, eq=False)
class StoredMatch:
    """``MatchRecord`` of a finished replay, rebuilt from the session manifest and stored events.

    ``samples`` holds a single synthetic row at the last replayed clock reading: it is used only
    for the period layout when no period_end event was replayed (a truncated replay).
    """

    match_id: str
    lineups: tuple[Lineup, ...]
    events: tuple[MatchEvent, ...]
    prematch: tuple[PreMatchRisk, ...]
    vmax_mps: dict[str, float]
    end_period: int
    end_t_s: float

    @property
    def samples(self) -> pd.DataFrame:
        return pd.DataFrame({"period": [self.end_period], "t_s": [self.end_t_s]})


@dataclass(frozen=True, eq=False)
class ReplaySession:
    """Everything serving a replay needs: match record, result frames, fit, norms and store."""

    cfg: Config
    match: MatchRecord
    result: ReplayResult
    fitted: FittedHazard
    norms: NormTable
    store: LogStore


def _end_clock(match: MatchRecord, until_s: float | None) -> tuple[int, float]:
    """Match-clock reading (period, t_s) of the last replayed play second."""
    starts, match_end_u = period_layout(match)
    end_u = min(match_end_u, until_s) if until_s is not None else match_end_u
    period = max(p for p in (1, 2, 3, 4) if starts[p] <= end_u)
    return period, PERIOD_START_S[period] + (end_u - starts[period])


def write_session(
    store: LogStore,
    match: MatchRecord,
    fitted: FittedHazard,
    history: pd.DataFrame | None,
    *,
    until_s: float | None = None,
) -> Path:
    """Write the session manifest of a finished replay next to its log parts; return the folder.

    ``until_s`` is the replay's truncation on the play clock (None: the whole match); the last
    replayed clock reading is derived from it and the match's period layout. ``history`` is the
    norm history the replay's norms were built from (see ``synth.history.feature_history``).
    """
    folder = store.directory
    folder.mkdir(parents=True, exist_ok=True)
    fitted.save(folder / FITTED_FILE)
    end_period, end_t_s = _end_clock(match, until_s)
    payload = {
        "version": 1,
        "match_id": match.match_id,
        "config_hash": store.config_hash,
        "end": {"period": end_period, "t_s": end_t_s},
        "lineups": [lu.model_dump(mode="json") for lu in match.lineups],
        "prematch": [r.model_dump(mode="json") for r in match.prematch],
        "vmax_mps": match.vmax_mps,
    }
    (folder / CONTEXT_FILE).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    if history is not None:
        history.to_parquet(folder / HISTORY_FILE, index=False)
    return folder


def _session_dir(log_root: str | Path, match_id: str, match_date: date | None) -> Path:
    root = Path(log_root)
    if match_date is not None:
        folder = root / match_date.isoformat() / match_id
        if not folder.is_dir():
            raise FileNotFoundError(f"no log partition at {folder}")
        return folder
    found = sorted(p for p in root.glob(f"*/{match_id}") if p.is_dir())
    if not found:
        raise FileNotFoundError(f"no log partition for match {match_id!r} under {root}")
    if len(found) > 1:
        raise ValueError(
            f"several log partitions for match {match_id!r}; pass the match date explicitly"
        )
    return found[0]


def _clean(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and math.isnan(value) else value


def _stored_events(store: LogStore) -> tuple[MatchEvent, ...]:
    """The events of the store as contracts (NaN back to None on the optional fields)."""
    frame = store.read("events")
    events = []
    for rec in frame.to_dict("records"):
        data = {name: _clean(rec.get(name)) for name in MatchEvent.model_fields}
        events.append(MatchEvent.model_validate(data))
    return tuple(events)


def _result_from_store(store: LogStore) -> ReplayResult:
    """The replay result frames read back from the store kinds."""
    displayed = store.read("display")
    risks = store.read("risk")
    if displayed.shape[0] == 0 or risks.shape[0] == 0:
        raise ValueError(
            "the stored replay has no display or risk rows; it did not reach the first risk "
            "update, so there is nothing to serve"
        )
    alerts = store.read("alerts")
    for col in ("team_id", "change", "shown_wall_ns"):
        if col not in alerts.columns:
            alerts[col] = pd.Series(dtype=object)
    features = store.read("features")
    if "stage" in features.columns:
        enriched = features.loc[features["stage"] == "risk_tick"].reset_index(drop=True)
    else:
        enriched = features
    return ReplayResult(
        features=features,
        risks=risks,
        alerts=alerts,
        displayed=displayed,
        delivered=pd.DataFrame(),
        delivered_events=store.read("events"),
        n_lost=0,
        n_queue_dropped=0,
        enriched=enriched,
    )


def load_session(
    log_root: str | Path, match_id: str, cfg: Config, *, match_date: date | None = None
) -> ReplaySession:
    """Load the session of a finished replay from its log partition.

    ``match_date`` selects the date partition when several exist; without it there must be
    exactly one. Norms are rebuilt from the stored history with the current config; a session
    without a history file gets an empty norm table, which only drops the "vs his usual" factor
    of a reason line. Decisions recorded while serving are appended to the same store, stamped
    with the config hash of the replay run.
    """
    folder = _session_dir(log_root, match_id, match_date)
    raw = json.loads((folder / CONTEXT_FILE).read_text(encoding="utf-8"))
    store = LogStore(
        Path(log_root),
        match_id,
        config_hash=str(raw["config_hash"]),
        match_date=date.fromisoformat(folder.parent.name),
    )
    match = StoredMatch(
        match_id=str(raw["match_id"]),
        lineups=tuple(Lineup.model_validate(lu) for lu in raw["lineups"]),
        events=_stored_events(store),
        prematch=tuple(PreMatchRisk.model_validate(r) for r in raw["prematch"]),
        vmax_mps={str(k): float(v) for k, v in raw["vmax_mps"].items()},
        end_period=int(raw["end"]["period"]),
        end_t_s=float(raw["end"]["t_s"]),
    )
    history_path = folder / HISTORY_FILE
    if history_path.is_file():
        positions = {p.player_id: p.position_group for lu in match.lineups for p in lu.players}
        norms = build_norms(
            pd.read_parquet(history_path), positions, match.lineups[0].kickoff_utc.date(), cfg
        )
    else:
        norms = NormTable({}, {})
    return ReplaySession(
        cfg=cfg,
        match=match,
        result=_result_from_store(store),
        fitted=FittedHazard.load(folder / FITTED_FILE),
        norms=norms,
        store=store,
    )
