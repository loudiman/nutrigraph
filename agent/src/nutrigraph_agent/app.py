"""FastAPI. One route that runs one Turn and streams its events."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .config import Settings
from .db import PostgresDatabase
from .deps import Deps
from .graph import build_graph
from .models import TurnEventEnvelope, TurnRequest
from .providers import Models, langchain_embedding_factory, langchain_factory
from .turn import run_turn

NDJSON = "application/x-ndjson"

log = logging.getLogger("nutrigraph.agent")


async def open_checkpointer(database_url: str) -> tuple[AsyncPostgresSaver, AsyncConnectionPool]:
    """The checkpointer creates and owns its own tables. No migration file
    names one, and nothing here names one either."""
    pool = AsyncConnectionPool(
        database_url,
        open=False,
        check=AsyncConnectionPool.check_connection,
        kwargs={
            # Neon's pooled endpoint hands one server connection to many
            # clients, so a prepared statement outlives the transaction that
            # made it and is then looked for on a connection that never saw it.
            "prepare_threshold": 0,
            "autocommit": True,
            "row_factory": dict_row,
        },
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver, pool


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # An instance can sit idle for longer than Neon keeps a connection, and
        # the pool then hands out a closed socket. `check` costs one round trip
        # per checkout and turns that into a reconnect.
        pool = AsyncConnectionPool(
            settings.database_url,
            open=False,
            check=AsyncConnectionPool.check_connection,
            kwargs={"prepare_threshold": 0},
        )
        await pool.open()
        saver, saver_pool = await open_checkpointer(settings.database_url)
        app.state.deps = Deps(
            db=PostgresDatabase(pool),
            models=Models(
                factory=langchain_factory(settings.model_provider),
                schema_model=settings.schema_model,
                prose_model=settings.prose_model,
                embedding_factory=langchain_embedding_factory(settings.model_provider),
                embedding_model=settings.embedding_model,
            ),
        )
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
