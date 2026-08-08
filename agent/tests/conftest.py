from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from nutrigraph_agent import graph as graph_module
from nutrigraph_agent._windows import selector_event_loop_policy
from nutrigraph_agent.deps import Deps, manila_now
from nutrigraph_agent.graph import build_graph
from nutrigraph_agent.models import AnswerEvent, TurnEvent
from nutrigraph_agent.turn import run_turn

from .fakes import FakeDatabase, FakeFoodSearch, FakeProvider

SCHEMA_MODEL = "gemini-3.5-flash-lite"
PROSE_MODEL = "gemini-3.5-flash"


@dataclass
class TurnSeam:
    """The agent turn seam: a `user_id` and a message in, a validated
    `CoachReply` and the emitted node events out. Everything below is real —
    including the redaction wrapper, which sits above the faked provider."""

    db: FakeDatabase
    provider: FakeProvider
    checkpointer: InMemorySaver
    food: FakeFoodSearch = field(default_factory=FakeFoodSearch)
    # The clock the Turn reads. A file that puts a Meal on a fixed day pins this
    # to the same one, so "today" is a fact of the test and not of the wall.
    now: Callable[[], datetime] = manila_now

    def reconnect(self) -> None:
        """A Session ends and a new one opens. The Thread continues."""
        self.graph = build_graph(self.checkpointer)

    def __post_init__(self) -> None:
        self.deps = Deps(
            db=self.db,
            models=self.provider.models(
                schema_model=SCHEMA_MODEL, prose_model=PROSE_MODEL
            ),
            food=self.food,
            now=self.now,
        )
        self.reconnect()

    async def turn(self, message: str, user_id: str = "demo-user-1",
                   turn_id: UUID | None = None) -> list[TurnEvent]:
        turn_id = turn_id or uuid4()
        return [
            event
            async for event in run_turn(
                self.graph, self.deps, user_id=user_id, turn_id=turn_id, message=message
            )
        ]

    def state(self, user_id: str = "demo-user-1") -> dict:
        return self.graph.get_state({"configurable": {"thread_id": user_id}}).values


class Interloper(BaseModel):
    """A schema no node asks for, until a test double asks for it."""

    noted: bool = True


def also_calls_the_provider(seam: TurnSeam, monkeypatch) -> TurnSeam:
    """Make the Turn call the provider once more than the graph does.

    The patch is on `load_profile` — a node every Turn runs, and no test that
    uses this is about. A node cannot be added to a compiled graph, so the patch
    is on the module global `build_graph` resolves and the seam is rebuilt after
    it.

    This is the regression guard for issues #54 and #59. Anything a test binds
    to the *call* — a scripted answer read by schema, an attempt read by
    `attempts_on` — reads the same under it. Anything bound to a position in a
    list moves by one and lands on the wrong node.
    """
    unpatched = graph_module.load_profile

    async def loads_and_calls_the_provider(state, config):
        loaded = await unpatched(state, config)
        ctx = graph_module.turn_context(config)
        await ctx.models.fill(Interloper, system="unrelated", user="unrelated")
        return loaded

    monkeypatch.setattr(graph_module, "load_profile", loads_and_calls_the_provider)
    seam.reconnect()
    seam.provider.script(Interloper())
    return seam


@pytest.fixture(scope="session")
def event_loop_policy():
    import asyncio

    return selector_event_loop_policy() or asyncio.get_event_loop_policy()


@pytest.fixture
def seam() -> TurnSeam:
    return TurnSeam(
        db=FakeDatabase(), provider=FakeProvider(), checkpointer=InMemorySaver()
    )


def answer(events: list[TurnEvent]) -> AnswerEvent:
    return next(e for e in events if isinstance(e, AnswerEvent))


@pytest.fixture(scope="session")
def integration_database_url() -> str:
    url = os.environ.get("NUTRIGRAPH_TEST_DATABASE_URL")
    if not url:
        pytest.skip("NUTRIGRAPH_TEST_DATABASE_URL is not set")
    return url
