"""One Turn, end to end. This is the agent turn seam: a `user_id` and a message
in, a validated `CoachReply` and the emitted node events out."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from .deps import Deps
from .graph import TURN_CONTEXT_KEY, TurnContext, UnknownUser
from .models import AnswerEvent, ErrorEvent, NodeEvent, TurnEvent

log = logging.getLogger("nutrigraph.agent.turn")

FALLBACK_ERROR = "The Coach could not finish that. Nothing was saved. Please try again."


async def run_turn(
    graph: Any, deps: Deps, *, user_id: str, turn_id: UUID, message: str
) -> AsyncIterator[TurnEvent]:
    """Yield the Turn's events in order: the node events while the Turn runs,
    then the answer as one event at the end — or a typed error event, after
    which the caller closes the stream."""
    ctx = TurnContext(deps=deps, turn_id=turn_id, raw_message=message)
    config = {
        # One Thread for each User, and it never restarts. A Session ends; the
        # Thread continues, so the thread identifier is the user identifier.
        "configurable": {"thread_id": user_id, TURN_CONTEXT_KEY: ctx},
    }
    inputs = {
        "user_id": user_id,
        "messages": [{"role": "user", "text": message}],
        "pending_clarification": None,
    }
    try:
        async for update in graph.astream(inputs, config, stream_mode="updates"):
            for node in update:
                log.info("node finished", extra={"turn_id": str(turn_id), "node": node})
                yield NodeEvent(turn_id=turn_id, node=node)

        if ctx.reply is None:  # pragma: no cover - a graph that reached no composer
            raise RuntimeError("the graph produced no CoachReply")

        # Store the raw message and the trace, then release the answer. A later
        # slice runs the guardrail text scan on this line; the contract holds.
        await deps.db.store_message(
            user_id=user_id, turn_id=turn_id, role="user", raw_text=message
        )
        await deps.db.store_message(
            user_id=user_id, turn_id=turn_id, role="coach", raw_text=ctx.reply.text
        )
        log.info("turn finished", extra={"turn_id": str(turn_id)})
        yield AnswerEvent(turn_id=turn_id, reply=ctx.reply)
    except Exception as exc:
        code = "unknown_user" if isinstance(exc, UnknownUser) else "turn_failed"
        log.exception("turn failed", extra={"turn_id": str(turn_id), "code": code})
        yield ErrorEvent(turn_id=turn_id, code=code, message=FALLBACK_ERROR)
