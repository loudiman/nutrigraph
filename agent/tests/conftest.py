from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from nutrigraph_agent._windows import selector_event_loop_policy
from nutrigraph_agent.deps import Deps
from nutrigraph_agent.graph import build_graph
from nutrigraph_agent.models import AnswerEvent, TurnEvent
from nutrigraph_agent.turn import run_turn

from .fakes import FakeDatabase, FakeProvider

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

    def reconnect(self) -> None:
        """A Session ends and a new one opens. The Thread continues."""
        self.graph = build_graph(self.checkpointer)

    def __post_init__(self) -> None:
        self.deps = Deps(
            db=self.db,
            models=self.provider.models(
                schema_model=SCHEMA_MODEL, prose_model=PROSE_MODEL
            ),
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
