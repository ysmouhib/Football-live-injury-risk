from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from livrisk.config import Config, config_hash, load_config

pytestmark = [pytest.mark.milestone_m1]


def test_default_config_loads_typed(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert cfg.engine.blanking_delta_s == 60
    assert cfg.alerts.budget_per_match == 2
    assert cfg.features.hsr_threshold_mps == 5.5
    assert cfg.norms.recency_half_life_matches == 10


def test_every_field_documents_units(config_path: Path) -> None:
    # Acceptance: typed values with units documented in field descriptions.
    load_config(config_path)
    missing: list[str] = []

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            path = f"{prefix}{name}"
            ann = field.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                walk(ann, f"{path}.")
            elif not field.description:
                missing.append(path)

    walk(Config, "")
    assert missing == []


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
        ("engine", "blanking_delta_s", -10),
        ("features", "smoothing_window_s", -1.0),
        ("quality", "min_identity", -0.1),
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


def test_reliability_above_one_rejected(tmp_path: Path, config_path: Path) -> None:
    data = yaml.safe_load(config_path.read_text())
    data["model"]["reliability"]["hsr_5m"] = 1.2
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_config(p)


def test_negative_window_rejected(tmp_path: Path, config_path: Path) -> None:
    data = yaml.safe_load(config_path.read_text())
    data["features"]["windows_min"] = [1, -5, 10]
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


def test_env_var_selects_config(
    tmp_path: Path, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = yaml.safe_load(config_path.read_text())
    data["alerts"]["budget_per_match"] = 7
    p = tmp_path / "alt.yaml"
    p.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("LIVRISK_CONFIG", str(p))
    assert load_config().alerts.budget_per_match == 7
