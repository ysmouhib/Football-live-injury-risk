"""One replayed synthetic match shared by the replay and bench-app tests, built once per run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from livrisk.config import Config, config_hash, load_config
from livrisk.contracts import PERIOD_START_S
from livrisk.features.norms import NormTable, build_norms
from livrisk.features.stream import compute_features_offline
from livrisk.model.live_hazard import FittedHazard, Priors, build_design, fit_map
from livrisk.replay.simulator import ReplayResult, ReplaySettings, replay_match
from livrisk.storage.logstore import LogStore
from livrisk.synth.generator import (
    SynthMatch,
    SynthTruth,
    generate_match,
    generate_person_minutes,
    match_tables,
)

HISTORY_SEED = 201
MATCH_SEED = 203
#: The acceptance scenario of docs/05_VALIDATION.md §3.
SETTINGS = ReplaySettings(latency_s=30.0, dropout_rate=0.05, burst_s=30.0, seed=1)
#: (band, period, band start minute) of the six regular bands.
REGULAR_BANDS = ((0, 1, 0.0), (1, 1, 15.0), (2, 1, 30.0), (4, 2, 45.0), (5, 2, 60.0), (6, 2, 75.0))


def norm_history(match: SynthMatch, cfg: Config) -> pd.DataFrame:
    """Raw feature values of one past match at the ends of three 5-minute windows per band."""
    events = match_tables(match)["events"]
    p1_end = float(
        events.loc[(events["type"] == "period_end") & (events["period"] == 1), "t_s"].iloc[0]
    )
    starts = {1: 0.0, 2: p1_end}
    day = match.lineups[0].kickoff_utc.date()
    rows: list[tuple[str, object, int, str, float]] = []
    for pid, frame in match.samples.groupby("player_id"):
        for band, period, b0 in REGULAR_BANDS:
            for w in (5.0, 10.0, 15.0):
                end_u = starts[period] + ((b0 + w) * 60.0 - PERIOD_START_S[period])
                snap = compute_features_offline(
                    frame,
                    events,
                    str(pid),
                    end_u + cfg.engine.blanking_delta_s,
                    match.vmax_mps.get(str(pid)),
                    cfg,
                )
                for feat, value in snap.features.items():
                    if value is not None and snap.coverage[feat] >= cfg.quality.min_coverage:
                        rows.append((str(pid), day, band, feat, value))
    return pd.DataFrame(rows, columns=["player_id", "match_date", "band", "feature", "value"])


@dataclass(frozen=True, eq=False)
class ReplayBundle:
    """Inputs and output of the shared replay; ``elapsed_s`` is the replay's wall time."""

    cfg: Config
    fitted: FittedHazard
    norms: NormTable
    match: SynthMatch
    result: ReplayResult
    elapsed_s: float
    store: LogStore


def build_bundle(config_path: Path, log_root: Path) -> ReplayBundle:
    cfg = load_config(config_path)
    truth = SynthTruth(p7_mean=0.10)
    df = generate_person_minutes(3000, seed=3, truth=truth, informative_censoring=False)
    fitted = fit_map(
        build_design(df, truth.feature_names, truth.match_share),
        Priors.from_config(cfg),
        config_hash=config_hash(cfg),
        version="replay-test",
    )
    match = generate_match(MATCH_SEED, cfg=cfg)
    history = norm_history(generate_match(HISTORY_SEED, cfg=cfg), cfg)
    positions = {p.player_id: p.position_group for lu in match.lineups for p in lu.players}
    norms = build_norms(history, positions, match.lineups[0].kickoff_utc.date(), cfg)
    store = LogStore(
        log_root,
        match.match_id,
        config_hash=config_hash(cfg),
        match_date=match.lineups[0].kickoff_utc.date(),
    )
    t0 = time.perf_counter()
    result = replay_match(match, fitted, cfg, SETTINGS, norms=norms, store=store)
    return ReplayBundle(cfg, fitted, norms, match, result, time.perf_counter() - t0, store)
