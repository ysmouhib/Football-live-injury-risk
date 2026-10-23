"""Command line of the tool (docs/04_ARCHITECTURE.md §7; docs/RUNBOOK.md).

Subcommands:

- ``replay``: run a synthetic match through the replay pipeline (source, features, engine,
  alerts) with a log store, write the session manifest and serve the bench app against the
  replayed state.
- ``serve``: serve the bench app of a finished replay from its log store.
- ``fit``: fit the live hazard model on a person-minutes table and write the FittedHazard JSON.
- ``evaluate``: reproduce the synthetic Gate C report (docs/05_VALIDATION.md §2) to a file.
- ``retention``: delete log partitions older than the retention window.

Every random process is seeded explicitly: a command run twice with the same arguments produces
identical output. The tool reports risk, relative risk and data quality; it never predicts
injuries and never recommends substitutions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from livrisk.app.server import create_app
from livrisk.app.state import ReplayState
from livrisk.config import Config, config_hash, load_config
from livrisk.evaluation.gate_c import run_gate_c
from livrisk.features.norms import build_norms
from livrisk.model.live_hazard import Priors, build_design, fit_map
from livrisk.replay.session import ReplaySession, load_session, write_session
from livrisk.replay.simulator import ReplaySettings, replay_match
from livrisk.storage.logstore import LogStore
from livrisk.storage.retention import enforce_retention
from livrisk.synth.generator import SynthTruth, generate_match, generate_person_minutes
from livrisk.synth.history import feature_history

logger = logging.getLogger(__name__)

#: Defaults of the synthetic replay wiring (the scenario of the replay tests).
HISTORY_SEED = 201
FIT_SEED = 3
FIT_MATCHES = 3000
FIT_VERSION = "synthetic-demo"

#: The synthetic Gate C scenario of the M10 acceptance tests (tests/test_m10_evaluation.py).
EVALUATE_MATCHES = 9000
EVALUATE_DATA_SEED = 11
EVALUATE_SEED = 1
EVALUATE_N_BOOT = 1000


@dataclass(frozen=True, eq=False)
class ServedReplay:
    """A bench app serving a replay session, with the state behind it."""

    state: ReplayState
    app: FastAPI


def build_replay_session(
    cfg: Config,
    *,
    seed: int,
    settings: ReplaySettings,
    log_root: str | Path,
    history_seed: int = HISTORY_SEED,
    fit_seed: int = FIT_SEED,
    fit_matches: int = FIT_MATCHES,
    sample_hz: float | None = None,
) -> ReplaySession:
    """Replay a synthetic match end to end and write the session for ``serve``.

    The fit uses ``fit_matches`` person-minutes from the fast synthetic path (seed
    ``fit_seed``, non-informative censoring); the norms come from one history match (seed
    ``history_seed``) through ``synth.history.feature_history``. The match (seed ``seed``) runs
    through ``replay.simulator.replay_match`` under ``settings`` (times in play seconds) and is
    written to a log store under ``log_root`` together with the session manifest
    (``replay.session.write_session``). Deterministic given the seeds and the config.
    """
    truth = SynthTruth(p7_mean=0.10)
    minutes = generate_person_minutes(
        fit_matches, seed=fit_seed, truth=truth, informative_censoring=False
    )
    fitted = fit_map(
        build_design(minutes, truth.feature_names, truth.match_share),
        Priors.from_config(cfg),
        config_hash=config_hash(cfg),
        version=FIT_VERSION,
    )
    match = generate_match(seed, cfg=cfg, sample_hz=sample_hz)
    history = feature_history(generate_match(history_seed, cfg=cfg, sample_hz=sample_hz), cfg)
    positions = {p.player_id: p.position_group for lu in match.lineups for p in lu.players}
    norms = build_norms(history, positions, match.lineups[0].kickoff_utc.date(), cfg)
    store = LogStore(
        log_root,
        match.match_id,
        config_hash=config_hash(cfg),
        match_date=match.lineups[0].kickoff_utc.date(),
    )
    result = replay_match(match, fitted, cfg, settings, norms=norms, store=store)
    write_session(store, match, fitted, history, until_s=settings.until_s)
    return ReplaySession(
        cfg=cfg, match=match, result=result, fitted=fitted, norms=norms, store=store
    )


def serve_session(
    session: ReplaySession,
    *,
    team_id: str | None = None,
    speed: float = 1.0,
    token: str | None = None,
) -> ServedReplay:
    """The bench app over a replay session; the cursor follows real time at ``speed``.

    ``speed`` is in multiples of real time (2.0 replays twice as fast). Staff decisions posted
    while serving are appended to the session's log store.
    """
    state = ReplayState(
        session.match,
        session.result,
        session.fitted,
        session.norms,
        session.cfg,
        team_id=team_id,
        store=session.store,
    )
    state.play(speed)
    return ServedReplay(state=state, app=create_app(state, session.cfg, token=token))


def _run_server(app: FastAPI, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _features(argument: str | None, default: list[str]) -> list[str]:
    if argument is None:
        return default
    return [name.strip() for name in argument.split(",") if name.strip()]


def _read_minutes(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"unsupported person-minutes format {suffix!r} (use .csv or .parquet)")


def cmd_replay(args: argparse.Namespace) -> int:
    """Run ``livrisk replay``: synthetic match through the live path, then serve."""
    cfg: Config = load_config(args.config)
    settings = ReplaySettings(
        latency_s=args.latency_s,
        dropout_rate=args.dropout_rate,
        burst_s=args.burst_s,
        until_s=args.until_s,
        seed=args.seed,
    )
    session = build_replay_session(
        cfg,
        seed=args.seed,
        settings=settings,
        log_root=args.log_dir or cfg.storage.log_dir,
        history_seed=args.history_seed,
        fit_seed=args.fit_seed,
        fit_matches=args.fit_matches,
        sample_hz=args.sample_hz,
    )
    print(
        f"fit: {session.fitted.n_rows} person-minutes, {session.fitted.n_events} events "
        f"(version {session.fitted.version})"
    )
    print(
        f"replay: match {session.match.match_id}; {session.result.risks.shape[0]} risk rows, "
        f"{session.result.alerts.shape[0]} alert rows; logs at {session.store.directory}"
    )
    if args.serve:
        host = args.host or cfg.app.host
        port = int(args.port or cfg.app.port)
        served = serve_session(session, team_id=args.team_id, speed=args.speed, token=args.token)
        print(
            f"serving the bench app on http://{host}:{port} "
            f"(replay mode, speed {args.speed:g}x); Ctrl+C stops"
        )
        _run_server(served.app, host, port)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run ``livrisk serve``: the bench app of a finished replay from its log store."""
    cfg: Config = load_config(args.config)
    session = load_session(
        args.log_dir or cfg.storage.log_dir, args.match_id, cfg, match_date=args.match_date
    )
    host = args.host or cfg.app.host
    port = int(args.port or cfg.app.port)
    served = serve_session(session, team_id=args.team_id, speed=args.speed, token=args.token)
    print(
        f"serving match {args.match_id} on http://{host}:{port} "
        f"(replay mode, speed {args.speed:g}x); Ctrl+C stops"
    )
    _run_server(served.app, host, port)
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """Run ``livrisk fit``: MAP fit of the live hazard model on a person-minutes table."""
    cfg: Config = load_config(args.config)
    if args.synthetic:
        if args.seed is None:
            raise ValueError("fit --synthetic needs an explicit --seed")
        truth = SynthTruth()
        minutes = generate_person_minutes(args.n_player_matches, seed=args.seed, truth=truth)
        feature_names = _features(args.features, list(truth.feature_names))
        match_share = truth.match_share
    else:
        minutes = _read_minutes(Path(args.input))
        feature_names = _features(args.features, list(cfg.features.main_features))
        match_share = float(cfg.prematch.match_share_prior)
    fitted = fit_map(
        build_design(minutes, feature_names, match_share),
        Priors.from_config(cfg),
        config_hash=config_hash(cfg),
        version=args.version,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fitted.save(out)
    print(
        f"fit: {fitted.n_rows} person-minutes, {fitted.n_events} events, "
        f"config {fitted.config_hash[:12]} -> {out}"
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run ``livrisk evaluate``: the synthetic Gate C report, byte-identical for fixed seeds."""
    cfg: Config = load_config(args.config)
    truth = SynthTruth(p7_mean=0.10)
    minutes = generate_person_minutes(args.n_player_matches, seed=args.data_seed, truth=truth)
    report = run_gate_c(
        minutes, truth.feature_names, cfg, split="time", seed=args.seed, n_boot=args.n_boot
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_markdown() + "\n", encoding="utf-8")
    print(
        f"evaluate: {report.n_spells} test spells, {report.n_events} events, "
        f"engine mode {report.mode} -> {out}"
    )
    if args.card is not None:
        card = Path(args.card)
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(report.model_card() + "\n", encoding="utf-8")
        print(f"evaluate: model card -> {card}")
    return 0


def cmd_retention(args: argparse.Namespace) -> int:
    """Run ``livrisk retention``: delete log partitions older than the retention window."""
    cfg: Config = load_config(args.config)
    days = int(args.days if args.days is not None else cfg.storage.retention_days)
    root = args.log_dir or cfg.storage.log_dir
    removed = enforce_retention(root, days)
    print(f"retention: removed {len(removed)} partition(s) older than {days} days from {root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``livrisk`` command."""
    parser = argparse.ArgumentParser(
        prog="livrisk",
        description=(
            "Local decision support: live, player-relative injury risk for football bench "
            "staff. It reports risk and data quality; it never predicts an injury."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            type=Path,
            default=None,
            help="config YAML (default: LIVRISK_CONFIG or configs/default.yaml)",
        )

    def serving(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", default=None, help="bind address (default: app.host)")
        p.add_argument("--port", type=int, default=None, help="TCP port (default: app.port)")
        p.add_argument(
            "--token", default=None, help="access token (required off loopback, docs/04 §8)"
        )
        p.add_argument(
            "--team-id", default=None, help="bench shown by the app (default: first team)"
        )
        p.add_argument(
            "--speed",
            type=float,
            default=1.0,
            help="cursor speed of the served replay, multiples of real time (default: 1)",
        )

    p_replay = sub.add_parser(
        "replay", help="replay a synthetic match end to end and serve the bench app"
    )
    p_replay.add_argument(
        "--synthetic",
        action="store_true",
        required=True,
        help="replay a synthetic match (the only source so far)",
    )
    p_replay.add_argument("--seed", type=int, required=True, help="seed of the match")
    p_replay.add_argument(
        "--until-s",
        type=float,
        default=None,
        help="stop the replay after this many play seconds (default: the whole match)",
    )
    p_replay.add_argument(
        "--serve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="serve the bench app after the replay (default; --no-serve only writes the store)",
    )
    p_replay.add_argument(
        "--latency-s", type=float, default=5.0, help="source-to-ingest delay, seconds"
    )
    p_replay.add_argument(
        "--dropout-rate", type=float, default=0.0, help="share of samples lost at the source"
    )
    p_replay.add_argument(
        "--burst-s",
        type=float,
        default=0.0,
        help="one contiguous loss of this length per player and period, seconds",
    )
    p_replay.add_argument(
        "--history-seed", type=int, default=HISTORY_SEED, help="seed of the norm history match"
    )
    p_replay.add_argument("--fit-seed", type=int, default=FIT_SEED, help="seed of the fit data")
    p_replay.add_argument(
        "--fit-matches",
        type=int,
        default=FIT_MATCHES,
        help="player-matches of the fit data",
    )
    p_replay.add_argument(
        "--sample-hz",
        type=float,
        default=None,
        help="tracking rate of the synthetic matches (default: synth.tick_hz)",
    )
    p_replay.add_argument(
        "--log-dir", default=None, help="log store root (default: storage.log_dir)"
    )
    common(p_replay)
    serving(p_replay)
    p_replay.set_defaults(func=cmd_replay)

    p_serve = sub.add_parser("serve", help="serve the bench app of a finished replay")
    p_serve.add_argument("--match-id", required=True, help="match to serve")
    p_serve.add_argument(
        "--match-date",
        type=date.fromisoformat,
        default=None,
        help="match log partition date, YYYY-MM-DD (needed when several exist)",
    )
    p_serve.add_argument(
        "--log-dir", default=None, help="log store root (default: storage.log_dir)"
    )
    common(p_serve)
    serving(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_fit = sub.add_parser("fit", help="fit the live hazard model on a person-minutes table")
    source = p_fit.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", default=None, help="person-minutes table (.csv or .parquet)")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="generate the table from the synthetic fast path (needs --seed)",
    )
    p_fit.add_argument("--output", required=True, help="where to write the FittedHazard JSON")
    p_fit.add_argument(
        "--seed", type=int, default=None, help="seed of the synthetic table (--synthetic only)"
    )
    p_fit.add_argument(
        "--n-player-matches",
        type=int,
        default=3000,
        help="size of the synthetic table (default: 3000)",
    )
    p_fit.add_argument(
        "--features",
        default=None,
        help="comma-separated feature names (default: features.main_features)",
    )
    p_fit.add_argument("--version", default="dev", help="model version tag stored in the fit")
    common(p_fit)
    p_fit.set_defaults(func=cmd_fit)

    p_evaluate = sub.add_parser("evaluate", help="reproduce the synthetic Gate C report to a file")
    p_evaluate.add_argument("--output", required=True, help="where to write the markdown report")
    p_evaluate.add_argument("--card", default=None, help="optional path for the model card")
    p_evaluate.add_argument(
        "--n-player-matches",
        type=int,
        default=EVALUATE_MATCHES,
        help=f"size of the synthetic table (default: {EVALUATE_MATCHES}, the M10 scenario)",
    )
    p_evaluate.add_argument(
        "--data-seed",
        type=int,
        default=EVALUATE_DATA_SEED,
        help=f"seed of the synthetic table (default: {EVALUATE_DATA_SEED}, the M10 scenario)",
    )
    p_evaluate.add_argument(
        "--seed",
        type=int,
        default=EVALUATE_SEED,
        help=f"seed of the bootstraps and imputations (default: {EVALUATE_SEED})",
    )
    p_evaluate.add_argument(
        "--n-boot",
        type=int,
        default=EVALUATE_N_BOOT,
        help=f"bootstrap draws per interval (default: {EVALUATE_N_BOOT})",
    )
    common(p_evaluate)
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_retention = sub.add_parser(
        "retention", help="delete log partitions older than the retention window"
    )
    p_retention.add_argument(
        "--log-dir", default=None, help="log store root (default: storage.log_dir)"
    )
    p_retention.add_argument(
        "--days",
        type=int,
        default=None,
        help="retention window in days (default: storage.retention_days)",
    )
    common(p_retention)
    p_retention.set_defaults(func=cmd_retention)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``livrisk`` command; returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        code: int = args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
