"""Bench app server (docs/06_UI_SPEC.md).

FastAPI app with a WebSocket ``/ws/live?role=...`` pushing the live table payload of
docs/06_UI_SPEC.md §8, REST ``POST /api/decisions`` writing DecisionRecords, and static files
from ``app/static``. Binds to 127.0.0.1 by default; any other address needs a token, checked on
every request. All strings come from ``wording.py``. Every connection is logged with its role
(the access log of docs/07).
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from livrisk.app import wording
from livrisk.config import Config

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


class LiveState(Protocol):
    """What the server needs from the running engine (live, shadow or replay)."""

    def live_table(self, role: str) -> dict[str, Any]: ...

    def record_decision(self, payload: dict[str, Any]) -> object: ...


class DecisionIn(BaseModel):
    """A staff decision posted from an alert card."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    actor_role: str
    button: str | None = None
    action: str | None = None
    note: str = ""


def create_app(
    state: LiveState,
    cfg: Config,
    *,
    token: str | None = None,
    push_interval_s: float = 1.0,
) -> FastAPI:
    """The bench app for ``state``. ``token`` is required unless ``app.host`` is a loopback
    address; ``push_interval_s`` is the WebSocket push cadence (the payload carries its own data
    age, so a slow push never shows a stale value as live)."""
    if cfg.app.host not in _LOOPBACK and not token:
        raise ValueError(f"serving on {cfg.app.host} needs a token (LAN exposure, docs/04 §8)")
    roles = set(cfg.app.roles)
    app = FastAPI(title=wording.UI_TEXT["title"], docs_url=None, redoc_url=None, openapi_url=None)

    def check(role: str, given: str | None) -> None:
        if token is not None and not hmac.compare_digest(given or "", token):
            raise HTTPException(status_code=401, detail="token required")
        if role not in roles:
            raise HTTPException(status_code=403, detail=f"unknown role {role!r}")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/wording")
    def texts() -> dict[str, Any]:
        return {
            "ui": wording.UI_TEXT,
            "help": wording.HELP_TEXT,
            "buttons": [{"key": k, "text": wording.UI_TEXT[k]} for k in wording.DECISION_BUTTONS],
        }

    @app.get("/api/live")
    def live(role: str = Query("analyst"), token_q: str | None = Query(None, alias="token")) -> Any:
        check(role, token_q)
        return state.live_table(role)

    @app.post("/api/decisions", status_code=201)
    def decide(
        body: DecisionIn,
        role: str = Query("analyst"),
        token_q: str | None = Query(None, alias="token"),
    ) -> dict[str, Any]:
        check(role, token_q)
        if body.actor_role not in roles:
            raise HTTPException(status_code=400, detail=f"unknown actor role {body.actor_role!r}")
        try:
            record = state.record_decision(body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info("decision on %s by %s", body.alert_id, body.actor_role)
        saved: dict[str, Any] = {"saved": wording.UI_TEXT["decision_saved"]}
        if isinstance(record, BaseModel):
            saved["record"] = record.model_dump(mode="json")
        return saved

    @app.websocket("/ws/live")
    async def ws_live(
        websocket: WebSocket,
        role: str = Query("analyst"),
        token_q: str | None = Query(None, alias="token"),
    ) -> None:
        try:
            check(role, token_q)
        except HTTPException as exc:
            await websocket.close(code=4000 + exc.status_code)
            return
        await websocket.accept()
        logger.info("bench connection opened, role %s", role)
        try:
            while True:
                await websocket.send_json(state.live_table(role))
                await asyncio.sleep(push_interval_s)
        except WebSocketDisconnect:
            logger.info("bench connection closed, role %s", role)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
