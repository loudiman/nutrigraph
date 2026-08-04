"""FastAPI. One route that runs one Turn and streams its events."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .config import CHECKPOINT_SCHEMA, Settings
from .db import PostgresDatabase
from .deps import Deps
from .graph import build_graph
from .models import TurnEventEnvelope, TurnRequest
from .turn import run_turn

NDJSON = "application/x-ndjson"

log = logging.getLogger("nutrigraph.agent")


async def open_checkpointer(database_url: str) -> tuple[AsyncPostgresSaver, AsyncConnectionPool]:
    """The checkpointer's tables live in the `langgraph` schema, which the
    library owns. No migration file references it, so the service creates it."""
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        await conn.execute(f"create schema if not exists {CHECKPOINT_SCHEMA}")
    pool = AsyncConnectionPool(
        database_url,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": f"-c search_path={CHECKPOINT_SCHEMA},public",
        },
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver, pool


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = AsyncConnectionPool(settings.database_url, open=False)
        await pool.open()
        saver, saver_pool = await open_checkpointer(settings.database_url)
        app.state.deps = Deps(db=PostgresDatabase(pool))
        app.state.graph = build_graph(saver)
        try:
            yield
        finally:
            await pool.close()
            await saver_pool.close()

    app = FastAPI(title="NutriGraph agent", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    def require_internal_caller(
        request: Request, x_dev_auth: Annotated[str | None, Header()] = None
    ) -> None:
        """In production the caller is verified by Cloud Run before application
        code runs, so there is nothing to check here. Locally the agent binds to
        loopback and accepts this plain development header instead."""
        if not settings.dev_auth:
            return
        if x_dev_auth != settings.dev_token:
            raise HTTPException(status_code=401, detail="internal caller not recognised")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/turn",
        responses={
            200: {"model": TurnEventEnvelope, "content": {NDJSON: {}}},
            401: {"description": "internal caller not recognised"},
        },
        dependencies=[Depends(require_internal_caller)],
    )
    async def turn(
        request: Request,
        body: TurnRequest,
        x_turn_id: Annotated[UUID, Header(description="The one identifier that ties a Turn together")],
    ) -> StreamingResponse:
        async def lines() -> AsyncIterator[bytes]:
            async for event in run_turn(
                request.app.state.graph,
                request.app.state.deps,
                user_id=body.user_id,
                turn_id=x_turn_id,
                message=body.message,
            ):
                yield TurnEventEnvelope(event).model_dump_json().encode() + b"\n"

        return StreamingResponse(lines(), media_type=NDJSON)

    return app
