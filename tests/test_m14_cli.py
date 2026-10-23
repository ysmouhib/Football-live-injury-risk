from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from livrisk import cli
from livrisk.app import wording
from livrisk.cli import (
    EVALUATE_DATA_SEED,
    EVALUATE_MATCHES,
    EVALUATE_N_BOOT,
    EVALUATE_SEED,
    build_parser,
    main,
    serve_session,
)
from livrisk.config import config_hash, load_config
from livrisk.evaluation.gate_c import run_gate_c
from livrisk.model.live_hazard import FittedHazard, Priors, build_design, fit_map
from livrisk.replay.session import load_session
from livrisk.storage.retention import enforce_retention
from livrisk.synth.generator import SynthTruth, generate_person_minutes
from tests.conftest import DEFAULT_CONFIG

pytestmark = [pytest.mark.milestone_m14]

CONFIG = str(DEFAULT_CONFIG)
ROOT = Path(__file__).resolve().parents[1]
#: A short replay is enough: the first risk update lands at 60 play seconds.
REPLAY_ARGS = [
    "replay",
    "--synthetic",
    "--seed",
    "1",
    "--until-s",
    "300",
    "--sample-hz",
    "2",
    "--no-serve",
    "--config",
    CONFIG,
]


@pytest.fixture(scope="module")
def replay_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One short synthetic replay through the CLI, shared by the replay/serve tests."""
    log_dir = tmp_path_factory.mktemp("cli_logs")
    assert main([*REPLAY_ARGS, "--log-dir", str(log_dir)]) == 0
    return log_dir


def test_replay_cli_end_to_end(replay_logs: Path) -> None:
    folders = sorted(replay_logs.glob("*/synth-1"))
    assert len(folders) == 1
    folder = folders[0]
    assert (folder / "fitted.json").is_file()
    assert (folder / "context.json").is_file()
    assert (folder / "history.parquet").is_file()
    assert list((folder / "risk").glob("part-*.parquet"))
    assert list((folder / "display").glob("part-*.parquet"))

    cfg = load_config(DEFAULT_CONFIG)
    session = load_session(replay_logs, "synth-1", cfg)
    assert session.result.risks.shape[0] > 0
    served = serve_session(session)
    served.state.seek(120.0)
    with TestClient(served.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        payload = client.get("/api/live", params={"role": "analyst"}).json()
        assert payload["type"] == "live_table"
        assert payload["match_id"] == "synth-1" and payload["rows"]
        assert client.get("/api/live", params={"role": "coach"}).json()["type"] == "coach_summary"
    served.state.seek(-60.0)
    with TestClient(served.app) as client:
        assert client.get("/api/live", params={"role": "analyst"}).json()["type"] == "pre_match"


def test_replay_cli_is_deterministic(replay_logs: Path, tmp_path: Path) -> None:
    other = tmp_path / "logs2"
    assert main([*REPLAY_ARGS, "--log-dir", str(other)]) == 0
    cfg = load_config(DEFAULT_CONFIG)
    first = load_session(replay_logs, "synth-1", cfg).result
    second = load_session(other, "synth-1", cfg).result
    pd.testing.assert_frame_equal(
        first.risks.reset_index(drop=True), second.risks.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        first.displayed.reset_index(drop=True), second.displayed.reset_index(drop=True)
    )


def test_serve_command_serves_from_the_store(
    replay_logs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served_apps: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "_run_server",
        lambda app, host, port: served_apps.update(app=app, host=host, port=port),
    )
    code = main(
        ["serve", "--match-id", "synth-1", "--log-dir", str(replay_logs), "--config", CONFIG]
    )
    assert code == 0
    assert served_apps["port"] == load_config(DEFAULT_CONFIG).app.port
    app = served_apps["app"]
    assert isinstance(app, FastAPI)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
    assert main(["serve", "--match-id", "nope", "--log-dir", str(replay_logs)]) == 1


def test_fit_and_evaluate_reproduce_report(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG)
    # fit: two runs are byte-identical and equal to the direct library fit.
    truth = SynthTruth()
    fit_args = [
        "fit",
        "--synthetic",
        "--n-player-matches",
        "600",
        "--seed",
        "7",
        "--version",
        "m14-test",
        "--config",
        CONFIG,
    ]
    fit1, fit2 = tmp_path / "fit1.json", tmp_path / "fit2.json"
    assert main([*fit_args, "--output", str(fit1)]) == 0
    assert main([*fit_args, "--output", str(fit2)]) == 0
    assert fit1.read_bytes() == fit2.read_bytes()
    minutes = generate_person_minutes(600, seed=7, truth=truth)
    direct = fit_map(
        build_design(minutes, truth.feature_names, truth.match_share),
        Priors.from_config(cfg),
        config_hash=config_hash(cfg),
        version="m14-test",
    )
    loaded = FittedHazard.load(fit1)
    assert np.array_equal(loaded.point, direct.point)
    assert np.array_equal(loaded.cov, direct.cov)
    assert loaded.config_hash == config_hash(cfg) and loaded.version == "m14-test"

    # evaluate: the written report equals run_gate_c's markdown byte for byte.
    signal = SynthTruth(p7_mean=0.10)
    eval_minutes = generate_person_minutes(1200, seed=11, truth=signal)
    gate = run_gate_c(eval_minutes, signal.feature_names, cfg, split="time", seed=1, n_boot=100)
    expected = gate.to_markdown() + "\n"
    eval_args = [
        "evaluate",
        "--n-player-matches",
        "1200",
        "--data-seed",
        "11",
        "--seed",
        "1",
        "--n-boot",
        "100",
        "--config",
        CONFIG,
    ]
    report1, report2, card = tmp_path / "r1.md", tmp_path / "r2.md", tmp_path / "card.md"
    assert main([*eval_args, "--output", str(report1), "--card", str(card)]) == 0
    assert main([*eval_args, "--output", str(report2)]) == 0
    assert report1.read_bytes() == report2.read_bytes()
    assert report1.read_text(encoding="utf-8") == expected
    assert "## Gate C criteria" in expected and "engine.mode:" in expected
    assert "## Intended use" in card.read_text(encoding="utf-8")
    # The bare command's defaults are the M10 synthetic scenario.
    defaults = build_parser().parse_args(["evaluate", "--output", "x.md"])
    assert (
        (
            defaults.n_player_matches,
            defaults.data_seed,
            defaults.seed,
            defaults.n_boot,
        )
        == (EVALUATE_MATCHES, EVALUATE_DATA_SEED, EVALUATE_SEED, EVALUATE_N_BOOT)
        == (
            9000,
            11,
            1,
            1000,
        )
    )


def test_retention_removes_only_old_date_partitions(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    for name in ("2026-09-01", "2026-10-20"):
        part = root / name / "m1" / "risk"
        part.mkdir(parents=True)
        (part / "part-00000.parquet").write_bytes(b"placeholder")
    notes = root / "notes"
    notes.mkdir()
    (notes / "keep.txt").write_text("keep", encoding="utf-8")
    removed = enforce_retention(root, 30, today=date(2026, 10, 23))
    assert removed == [root / "2026-09-01"]
    assert (root / "2026-10-20").is_dir() and (notes / "keep.txt").is_file()

    # The CLI takes the window from storage.retention_days when --days is omitted; a partition
    # 45 days old is removed, one 5 days old is kept, whatever today is.
    today = date.today()
    old = root / (today - timedelta(days=45)).isoformat()
    recent = root / (today - timedelta(days=5)).isoformat()
    old.mkdir()
    recent.mkdir()
    assert main(["retention", "--log-dir", str(root), "--config", CONFIG]) == 0
    assert not old.exists() and recent.is_dir()
    with pytest.raises(ValueError):
        enforce_retention(root, 0)


def test_runbook_covers_every_failure_mode() -> None:
    text = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    # The failure modes of docs/04_ARCHITECTURE.md section 6, one subsection each.
    for heading in (
        "Wearable dropout / not worn",
        "Identity swap (video)",
        "Player off camera",
        "Clock drift",
        "Substitution not entered",
        "Network loss to tablet",
        "Out-of-range inputs",
        "Crash",
    ):
        assert heading in text, heading
    assert "shadow" in text.lower()
    assert wording.find_forbidden(text) == []


def test_cli_help_has_no_forbidden_wording() -> None:
    parser = build_parser()
    texts = [parser.format_help()]
    subs = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for subparser in subs.choices.values():
        texts.append(subparser.format_help())
    for text in texts:
        assert wording.find_forbidden(text) == []
