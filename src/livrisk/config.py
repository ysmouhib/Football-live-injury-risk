"""Configuration loading.

Replace the placeholder ``Config`` with nested, frozen pydantic models that mirror
``configs/default.yaml`` exactly (``extra="forbid"``), with units in field descriptions.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Placeholder. M1: replace with typed nested models (engine, prematch, features, norms,
    quality, model, alerts, competition, app, storage, seeds)."""

    model_config = ConfigDict(frozen=True, extra="allow")


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate a YAML config. Raises on unknown keys or out-of-range values."""
    raise NotImplementedError("planned for M1")


def config_hash(cfg: Config) -> str:
    """Stable SHA-256 of the canonical JSON of ``cfg`` (sorted keys). Logged with every output."""
    raise NotImplementedError("planned for M1")
