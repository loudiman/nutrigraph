"""The graph. Two nodes in this slice: `load_profile` reads the Profile from
PostgreSQL, `compose` writes the CoachReply.

The state holds only what survives the Turn. Everything rebuilt every Turn
lives on the `TurnContext`, which travels in the config and therefore never
enters the checkpoint.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .deps import Deps
from .models import CoachReply, Profile, ReplyPart

# The key is underscore-prefixed so LangGraph keeps it out of checkpoint metadata.
TURN_CONTEXT_KEY = "__turn"

CHECKPOINTED = ("user_id", "messages", "pending_clarification")


class TurnState(TypedDict):
    user_id: str
    messages: Annotated[list[dict[str, str]], operator.add]
    pending_clarification: str | None


@dataclass
class TurnContext:
    """Rebuilt every Turn, never checkpointed."""

    deps: Deps
    turn_id: UUID
    raw_message: str
    profile: Profile | None = None
    reply: CoachReply | None = None
    nodes_run: list[str] = field(default_factory=list)


class UnknownUser(LookupError):
    """No Profile for this user_id. The Turn cannot run."""


def turn_context(config: RunnableConfig) -> TurnContext:
    return config["configurable"][TURN_CONTEXT_KEY]


async def load_profile(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    ctx = turn_context(config)
    profile = await ctx.deps.db.load_profile(state["user_id"])
    if profile is None:
        raise UnknownUser(state["user_id"])
    ctx.profile = profile
    ctx.nodes_run.append("load_profile")
    return {}


async def compose(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    ctx = turn_context(config)
    name = ctx.profile.name if ctx.profile else "there"
    # An echo. This slice proves the path, not the coaching.
    text = f"{name}, you said: {ctx.raw_message}"
    ctx.reply = CoachReply(text=text, parts=[ReplyPart(intent="echo", text=text)])
    ctx.nodes_run.append("compose")
    return {"messages": [{"role": "coach", "text": text}]}


def build_graph(checkpointer: Any):
    builder = StateGraph(TurnState)
    builder.add_node("load_profile", load_profile)
    builder.add_node("compose", compose)
    builder.add_edge(START, "load_profile")
    builder.add_edge("load_profile", "compose")
    builder.add_edge("compose", END)
    return builder.compile(checkpointer=checkpointer)
