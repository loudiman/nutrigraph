"""One Turn, end to end. This is the agent turn seam: a `user_id` and a message
in, a validated `CoachReply` and the emitted node events out."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from .deps import Deps
from .graph import TURN_CONTEXT_KEY, TurnContext, UnknownUser
from .guardrail import safe_reply, scan_reply
from .models import AnswerEvent, ErrorEvent, NodeEvent, TurnEvent
from .providers import ProviderUnavailable

log = logging.getLogger("nutrigraph.agent.turn")

FALLBACK_ERROR = "The Coach could not finish that. Nothing was saved. Please try again."

# The failures that have a name of their own on the error event. Everything else
# is `turn_failed`, and the User reads the same fixed message either way.
CODES = {UnknownUser: "unknown_user", ProviderUnavailable: "provider_unavailable"}


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
        # LangSmith is switched on by LANGSMITH_TRACING and LANGSMITH_API_KEY
        # and no code change. Each node becomes a nested run under this one, and
        # the trace carries the same turn identifier as the `message` and
        # `interaction_event` rows.
        "run_name": f"turn {turn_id}",
        "metadata": {"turn_id": str(turn_id), "user_id": user_id},
        "tags": [f"turn:{turn_id}"],
    }
    # `pending_clarification` is deliberately not seeded here. It is a plain
    # overwrite key, so passing it would wipe the pending Clarification at the
    # start of every Turn, and only the nodes that answer one may clear it — a
    # Refusal turn must leave it standing.
    inputs = {"user_id": user_id, "messages": [{"role": "user", "text": message}]}
    try:
        async for update in graph.astream(inputs, config, stream_mode="updates"):
            for node in update:
                log.info("node finished", extra={"turn_id": str(turn_id), "node": node})
                yield NodeEvent(turn_id=turn_id, node=node)

        if ctx.reply is None:  # pragma: no cover - a graph that reached no composer
            raise RuntimeError("the graph produced no CoachReply")

        # The guardrail's last gate. The answer was held back, so the scan runs
        # before the answer event is sent and an unapproved sentence never
        # reaches the screen. What fails the scan is replaced whole: the Turn
        # ends with the fixed safe message, not with a partial answer. A Refusal
        # is a template in code, never a model's words, so it is not scanned.
        reply = ctx.reply
        claim = None if ctx.refused else scan_reply(reply.text)
        if claim is not None:
            log.warning("the text scan blocked the answer",
                        extra={"turn_id": str(turn_id), "claim": claim})
            reply = safe_reply()

        # Store the raw message and the trace, then release the answer.
        await deps.db.store_message(
            user_id=user_id, turn_id=turn_id, role="user", raw_text=message
        )
        await deps.db.store_message(
            user_id=user_id, turn_id=turn_id, role="coach", raw_text=reply.text
        )
        # The private table that maps a placeholder back to what the User wrote.
        if ctx.models is not None:
            await deps.db.store_redaction_map(turn_id=turn_id, mapping=ctx.models.mapping)
        log.info("turn finished", extra={"turn_id": str(turn_id)})
        yield AnswerEvent(turn_id=turn_id, reply=reply)
    except Exception as exc:
        # The retry ladder ran out: two retries on the first model and one on
        # the weaker tier, all of them stopped. The Turn ends here with the
        # fixed fallback message, which is what makes the ladder's bound the
        # guarantee that a Turn can never hang.
        code = CODES.get(type(exc), "turn_failed")
        log.exception("turn failed", extra={"turn_id": str(turn_id), "code": code})
        yield ErrorEvent(turn_id=turn_id, code=code, message=FALLBACK_ERROR)
