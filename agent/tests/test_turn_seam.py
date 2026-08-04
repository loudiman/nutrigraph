"""One Turn, at the agent turn seam. A node is never tested on its own."""

from __future__ import annotations

from uuid import uuid4

import pytest

from nutrigraph_agent.graph import CHECKPOINTED
from nutrigraph_agent.models import AnswerEvent, CoachReply, ErrorEvent, NodeEvent

from .conftest import answer


async def test_turn_echoes_the_message(seam):
    events = await seam.turn("I ate two eggs and pandesal")

    reply = answer(events).reply
    assert isinstance(reply, CoachReply)
    assert "I ate two eggs and pandesal" in reply.text
    assert "Lou" in reply.text


async def test_raw_message_is_stored_with_the_turn_identifier(seam):
    turn_id = uuid4()
    await seam.turn("I ate two eggs", turn_id=turn_id)

    stored = seam.db.messages
    assert [m.role for m in stored] == ["user", "coach"]
    assert stored[0].raw_text == "I ate two eggs"
    assert {m.turn_id for m in stored} == {turn_id}


async def test_node_events_arrive_before_one_answer_event(seam):
    events = await seam.turn("hello")

    kinds = [type(e) for e in events]
    assert kinds == [NodeEvent, NodeEvent, AnswerEvent]
    assert [e.node for e in events[:2]] == ["load_profile", "compose"]


async def test_every_event_carries_the_turn_identifier(seam):
    turn_id = uuid4()
    events = await seam.turn("hello", turn_id=turn_id)

    assert {e.turn_id for e in events} == {turn_id}


async def test_a_failure_mid_turn_ends_the_stream_with_a_typed_error(seam):
    seam.db.fail_on_load = True
    turn_id = uuid4()

    events = await seam.turn("hello", turn_id=turn_id)

    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].turn_id == turn_id
    assert not any(isinstance(e, AnswerEvent) for e in events)
    assert seam.db.messages == []


async def test_an_unknown_user_is_a_typed_error_not_a_crash(seam):
    events = await seam.turn("hello", user_id="nobody")

    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == "unknown_user"


async def test_the_checkpoint_holds_only_the_three_thread_keys(seam):
    await seam.turn("hello")

    assert set(seam.state()) == set(CHECKPOINTED)


async def test_reconnecting_continues_the_same_thread(seam):
    await seam.turn("first")
    seam.reconnect()  # the Session ended; the Thread did not
    await seam.turn("second")

    texts = [m["text"] for m in seam.state()["messages"]]
    assert texts[0] == "first"
    assert "second" in texts[-1]
    assert len(texts) == 4


async def test_a_turn_reaches_no_provider(seam):
    # Gemini and FoodData Central are faked by an object that raises on touch,
    # so an answer event is proof that no node called a provider.
    events = await seam.turn("hello")

    assert isinstance(events[-1], AnswerEvent)
    with pytest.raises(RuntimeError):
        seam.deps.models.invoke
    with pytest.raises(RuntimeError):
        seam.deps.food.search
