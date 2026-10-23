from __future__ import annotations

from pathlib import Path

import pytest

from tests.replay_setup import ReplayBundle, build_bundle

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


@pytest.fixture(scope="session")
def replay_bundle(tmp_path_factory: pytest.TempPathFactory) -> ReplayBundle:
    """A full synthetic match replayed once per session (the M9 acceptance scenario)."""
    return build_bundle(DEFAULT_CONFIG, tmp_path_factory.mktemp("logs"))


def skip_without(path: Path) -> pytest.MarkDecorator:
    """Skip a test when an open dataset is not downloaded locally."""
    return pytest.mark.skipif(not path.exists(), reason=f"open data not present: {path}")
