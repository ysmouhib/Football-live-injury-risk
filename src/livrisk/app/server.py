"""Bench app server (docs/06_UI_SPEC.md).

FastAPI app with a WebSocket ``/ws/live?role=...`` pushing the live table payload of
docs/06_UI_SPEC.md §8, REST ``POST /decisions`` writing DecisionRecords, and static files from
``app/static``. Binds to 127.0.0.1 by default. All strings come from ``wording.py``.
"""

from __future__ import annotations

from typing import Any, Protocol

from fastapi import FastAPI

from livrisk.config import Config


class LiveState(Protocol):
    """What the server needs from the running engine (live, shadow or replay)."""

    def live_table(self, role: str) -> dict[str, Any]: ...

    def record_decision(self, payload: dict[str, Any]) -> None: ...


def create_app(state: LiveState, cfg: Config) -> FastAPI:
    raise NotImplementedError("planned for M11")
