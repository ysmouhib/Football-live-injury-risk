from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from livrisk.config import config_hash, load_config

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m1]


def test_default_config_loads_typed(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert cfg.engine.blanking_delta_s == 60  # type: ignore[attr-defined]
    assert cfg.alerts.budget_per_match == 2  # type: ignore[attr-defined]


def test_unknown_key_rejected(tmp_path: Path, config_path: Path) -> None:
    data = yaml.safe_load(config_path.read_text())
    data["engine"]["surprise"] = 1
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_config(p)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("alerts", "budget_per_match", -1),
        ("quality", "min_coverage", 1.5),
        ("engine", "risk_tick_s", 0),
    ],
)
def test_out_of_range_rejected(
    tmp_path: Path, config_path: Path, section: str, key: str, value: float
) -> None:
    data = yaml.safe_load(config_path.read_text())
    data[section][key] = value
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_config(p)


def test_config_hash_stable_and_sensitive(tmp_path: Path, config_path: Path) -> None:
    h1 = config_hash(load_config(config_path))
    assert h1 == config_hash(load_config(config_path))
    data = yaml.safe_load(config_path.read_text())
    data["alerts"]["budget_per_match"] = 3
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    assert config_hash(load_config(p)) != h1
