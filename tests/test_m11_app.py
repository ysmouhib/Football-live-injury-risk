from __future__ import annotations

import pytest

pytestmark = [pytest.mark.pending, pytest.mark.milestone_m11]


def test_live_payload_schema() -> None:
    # httpx/TestClient: payload matches docs/06_UI_SPEC.md §8
    pytest.fail("implement")


def test_coach_role_receives_only_allowed_fields() -> None:
    pytest.fail("implement")


def test_stale_values_hidden_server_side() -> None:
    pytest.fail("implement")


def test_payloads_from_replay_have_no_forbidden_wording() -> None:
    # Use livrisk.app.wording.find_forbidden on every string in every pushed payload
    pytest.fail("implement")


def test_decisions_logged() -> None:
    pytest.fail("implement")
