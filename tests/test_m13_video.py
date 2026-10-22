from __future__ import annotations

import pytest

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m13]


def test_identity_accuracy_with_misreads_and_swaps() -> None:
    # Synthetic tracks from an M2 match: 2 % jersey misreads, injected track ID swaps.
    # Expect >= 95 % of player-seconds correctly assigned; every swap flagged within 5 s.
    pytest.fail("implement")


def test_low_confidence_lowers_identity_confidence() -> None:
    pytest.fail("implement")
