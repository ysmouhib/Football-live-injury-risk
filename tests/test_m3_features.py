from __future__ import annotations

import numpy as np
import pytest

from livrisk.features.stream import detect_efforts, smooth_speed

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m3]


def _trace(segments: list[tuple[float, float]], hz: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant speed trace from (duration_s, speed_mps) segments."""
    t: list[float] = []
    v: list[float] = []
    now = 0.0
    for dur, speed in segments:
        n = round(dur * hz)
        t.extend(now + i / hz for i in range(n))
        v.extend([speed] * n)
        now += dur
    return np.asarray(t), np.asarray(v)


def test_effort_detection_merges_short_gaps_and_ignores_blips() -> None:
    t, v = _trace([(5, 3.0), (2.0, 7.5), (0.3, 6.0), (1.5, 7.5), (5, 3.0), (0.6, 7.5), (5, 3.0)])
    efforts = detect_efforts(t, v, threshold_mps=7.0, min_duration_s=1.0, merge_gap_s=0.5)
    assert len(efforts) == 1  # 2.0 s + 0.3 s gap + 1.5 s merged; the 0.6 s blip ignored
    assert efforts[0].start_s == pytest.approx(5.0, abs=0.11)
    assert efforts[0].end_s == pytest.approx(8.8, abs=0.11)


def test_smooth_speed_constant_velocity() -> None:
    hz = 25.0
    t = np.arange(0, 20, 1 / hz)
    x = 6.0 * t
    y = np.zeros_like(t)
    s = smooth_speed(t, x, y, window_s=1.0, polyorder=2)
    assert np.allclose(s[int(hz) : -int(hz)], 6.0, atol=1e-6)


def test_online_equals_offline_on_synthetic_match() -> None:
    # TODO: generate a synthetic match (M2 full path), feed it through StreamFeatureEngine at
    # 10 s ticks, and compare every snapshot with compute_features_offline to 1e-9.
    pytest.fail("implement: online == offline property test")


def test_no_read_beyond_cutoff() -> None:
    # TODO: instrument the ring buffer so that any access to samples with t_s > as_of - delta
    # raises; run a full match and assert no exception.
    pytest.fail("implement: instrumented buffer test")
