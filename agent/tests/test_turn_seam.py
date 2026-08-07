"""One Turn, at the agent turn seam. A node is never tested on its own."""

from __future__ import annotations

from uuid import uuid4

from nutrigraph_agent.graph import CHECKPOINTED, CONFIDENCE_FLOOR
from nutrigraph_agent.models import (
    INTENTS,
    AnswerEvent,
    CoachReply,
    ComposedReply,
    DayRequest,
    ErrorEvent,
    NodeEvent,
    RouterDecision,
)

from .conftest import PROSE_MODEL, SCHEMA_MODEL, answer

UNSURE = RouterDecision(intents=[], confidence=0.2)
# An Intent whose path is not built yet, so this file reads the stub and the
# seam rather than an Intent's own behaviour, which its own file tests.
SURE = RouterDecision(intents=["recommend"], confidence=0.95)


async def test_a_turn_calls_the_router_once_and_dispatches_what_it_decided(seam):
    seam.provider.script(
        RouterDecision(intents=["review_day", "recommend"], confidence=0.9),
        DayRequest(days_ago=0),
        ComposedReply(text="Lou, your day and what to eat next."),
    )

    events = await seam.turn("how did my day go, and what should I eat tonight?")

    reply = answer(events).reply
    assert isinstance(reply, CoachReply)
    assert [p.intent for p in reply.parts] == ["review_day", "recommend"]
    # The router and the review path's own day question on the schema tier, and
    # one composer call on the prose tier. `recommend` is the one Intent still
    # standing on the stub, so it asked a model nothing of its own.
    assert [c.model for c in seam.provider.seen] == [
        SCHEMA_MODEL, SCHEMA_MODEL, PROSE_MODEL,
    ]


async def test_the_intent_list_is_never_longer_than_two_and_holds_only_the_five(seam):
    events = await seam.turn("I ate two eggs")

    parts = answer(events).reply.parts
    assert len(parts) <= 2
    assert all(p.intent in INTENTS for p in parts)


async def test_node_events_arrive_before_one_answer_event(seam):
    events = await seam.turn("I ate two eggs")

    assert [type(e) for e in events] == [NodeEvent] * 5 + [AnswerEvent]
    assert [e.node for e in events[:5]] == [
        "load_profile", "guard", "route", "dispatch", "compose_reply",
    ]


async def test_raw_message_is_stored_unredacted_with_the_turn_identifier(seam):
    turn_id = uuid4()

    await seam.turn("I'm Lou, lou@example.com, I ate two eggs", turn_id=turn_id)

    stored = seam.db.messages
    assert [m.role for m in stored] == ["user", "coach"]
    # The database keeps the truth; the provider saw a cleaned copy.
    assert stored[0].raw_text == "I'm Lou, lou@example.com, I ate two eggs"
    assert {m.turn_id for m in stored} == {turn_id}


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
    # No Profile, so no Redactor, so nothing may reach the provider.
    assert seam.provider.seen == []


async def test_the_checkpoint_holds_only_the_three_thread_keys(seam):
    await seam.turn("hello")

    assert set(seam.state()) == set(CHECKPOINTED)


async def test_reconnecting_continues_the_same_thread(seam):
    await seam.turn("first")
    seam.reconnect()  # the Session ended; the Thread did not
    await seam.turn("second")

    texts = [m["text"] for m in seam.state()["messages"]]
    assert texts[0] == "first"
    assert texts[2] == "second"
    assert len(texts) == 4


# --- the clarify path ---------------------------------------------------------


async def test_a_low_confidence_turn_ends_with_one_short_question(seam):
    seam.provider.script(UNSURE)

    events = await seam.turn("hmm")

    assert [e.node for e in events[:4]] == ["load_profile", "guard", "route", "clarify"]
    reply = answer(events).reply
    assert reply.text.endswith("?")
    assert [p.intent for p in reply.parts] == ["clarify"]
    assert seam.state()["pending_clarification"] == reply.text


async def test_the_clarifying_question_addresses_the_user_by_name(seam):
    seam.provider.script(UNSURE)

    events = await seam.turn("hmm")

    # The provider saw a placeholder; the User is answered by name.
    assert "Lou" not in seam.provider.seen[-1].sent
    assert answer(events).reply.text.startswith("Lou,")


async def test_the_prose_tier_writes_the_question_and_the_schema_tier_routes(seam):
    seam.provider.script(UNSURE)

    await seam.turn("hmm")

    assert [c.model for c in seam.provider.seen] == [SCHEMA_MODEL, PROSE_MODEL]


async def test_a_confident_turn_clears_the_pending_clarification(seam):
    seam.provider.script(UNSURE, SURE)

    await seam.turn("hmm")
    assert seam.state()["pending_clarification"] is not None

    await seam.turn("I ate two eggs")

    assert seam.state()["pending_clarification"] is None


async def test_a_second_clarify_turn_replaces_the_pending_value(seam):
    seam.provider.script(UNSURE, UNSURE)

    await seam.turn("hmm")
    first_state = seam.state()
    await seam.turn("still hmm")
    second_state = seam.state()

    # One value, overwritten. Not a list, and not a second key.
    assert isinstance(second_state["pending_clarification"], str)
    assert set(second_state) == set(first_state) == set(CHECKPOINTED)
    assert len(second_state["messages"]) == 4


async def test_the_floor_is_the_boundary_not_a_range(seam):
    seam.provider.script(RouterDecision(intents=["recommend"], confidence=CONFIDENCE_FLOOR))

    events = await seam.turn("what should I eat")

    assert [e.node for e in events[:4]] == ["load_profile", "guard", "route", "dispatch"]


# --- the metric record --------------------------------------------------------


async def test_every_node_writes_an_interaction_event_row(seam):
    turn_id = uuid4()
    seam.provider.script(SURE)

    await seam.turn("I ate two eggs", turn_id=turn_id)

    rows = seam.db.events
    assert [r.node for r in rows] == [
        "load_profile", "guard", "route", "dispatch", "compose_reply",
    ]
    assert {r.turn_id for r in rows} == {turn_id}
    assert all(r.latency_ms >= 0 for r in rows)

    router = next(r for r in rows if r.node == "route")
    assert router.model == SCHEMA_MODEL
    assert router.intent == "recommend"
    assert (router.input_tokens, router.output_tokens) == (11, 7)
    assert router.cost_usd > 0

    reader = next(r for r in rows if r.node == "load_profile")
    assert reader.model is None
    assert reader.cost_usd == 0


async def test_the_turn_identifier_ties_the_message_and_the_metric_rows_together(seam):
    turn_id = uuid4()

    await seam.turn("I ate two eggs", turn_id=turn_id)

    assert {m.turn_id for m in seam.db.messages} == {turn_id}
    assert {e.turn_id for e in seam.db.events} == {turn_id}


async def test_the_placeholder_mapping_is_kept_in_its_own_table(seam):
    turn_id = uuid4()

    await seam.turn("I'm Lou and I ate two eggs", turn_id=turn_id)

    assert seam.db.redaction_maps[turn_id] == {"[NAME_1]": "Lou"}
