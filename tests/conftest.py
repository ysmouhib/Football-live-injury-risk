from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

#: Seeds reserved for a final holdout check of the synthetic pipeline. Never tune on these.
RESERVED_HOLDOUT_SEEDS: tuple[int, ...] = tuple(range(900_000, 900_050))
#: Seeds for development and acceptance tests.
DEV_SEEDS: tuple[int, ...] = tuple(range(1, 51))

OPEN_DATA = ROOT / "data" / "open"


@pytest.fixture
def config_path() -> Path:
    return DEFAULT_CONFIG


def skip_without(path: Path) -> pytest.MarkDecorator:
    """Skip a test when an open dataset is not downloaded locally."""
    return pytest.mark.skipif(not path.exists(), reason=f"open data not present: {path}")
