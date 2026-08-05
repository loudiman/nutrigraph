"""The `update_profile` Intent, at the agent turn seam.

The Profile lives in PostgreSQL alone, so every test here reads the change back
out of the database fake rather than out of the graph state — and the last group
is the trap: the allergy check must never run on this path.
"""

from __future__ import annotations

import pytest

from nutrigraph_agent.db import PostgresDatabase
from nutrigraph_agent.graph import ALLERGY_CHECKED_INTENTS, CHECKPOINTED
from nutrigraph_agent.models import (
    UPDATABLE_FIELDS,
    AnswerEvent,
    NodeEvent,
    Profile,
    ProfileUpdate,
    RouterDecision,
)

from .conftest import PROSE_MODEL, SCHEMA_MODEL, answer

STATES_A_FACT = RouterDecision(intents=["update_profile"], confidence=0.93)
RECOMMENDS = RouterDecision(intents=["recommend"], confidence=0.9)

SHRIMP = ProfileUpdate(field="allergies", old_value="peanut", new_value="shrimp")
TARGET = ProfileUpdate(field="target_weight_kg", old_value="72", new_value="70")
# The extractor could not pin the message to one field.
VAGUE = ProfileUpdate(field=None)


def profile(seam):
    return seam.db.profiles["demo-user-1"]


# --- the change reaches PostgreSQL --------------------------------------------


async def test_an_allergy_stated_in_a_message_is_added_to_the_profile(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP)

    events = await seam.turn("I am allergic to shrimp")

    assert profile(seam).allergies == ["peanut", "shrimp"]
    assert "shrimp" in answer(events).reply.text


async def test_a_target_weight_stated_in_a_message_is_written_as_a_number(seam):
    seam.provider.script(STATES_A_FACT, TARGET)

    await seam.turn("My target is 70 kilograms")

    assert profile(seam).target_weight_kg == 70.0


async def test_the_confirmation_names_the_field_the_old_value_and_the_new_one(seam):
    seam.provider.script(STATES_A_FACT, TARGET)

    events = await seam.turn("My target is 70 kilograms")

    reply = answer(events).reply
    assert "target weight" in reply.text
    assert "72" in reply.text and "70" in reply.text
    assert [p.intent for p in reply.parts] == ["update_profile"]


async def test_the_old_value_is_the_profiles_own_not_the_extractors(seam):
    """A hallucinated old value would tell the User the Coach changed something
    it did not."""
    seam.provider.script(
        STATES_A_FACT,
        ProfileUpdate(field="target_weight_kg", old_value="65", new_value="70"),
    )

    events = await seam.turn("My target is 70 kilograms")

    assert "65" not in answer(events).reply.text
    assert "72" in answer(events).reply.text


async def test_the_write_goes_to_postgresql_and_the_checkpoint_holds_no_profile(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP)

    await seam.turn("I am allergic to shrimp")

    assert profile(seam).allergies == ["peanut", "shrimp"]
    state = seam.state()
    # The Coach's own sentence is in the transcript, as every reply is. The
    # Profile itself is nowhere in the checkpoint, so no second copy can drift.
    assert set(state) == set(CHECKPOINTED)
    assert not any(isinstance(value, Profile) for value in state.values())
    assert not any(name in state for name in UPDATABLE_FIELDS)


async def test_the_changed_value_is_read_back_by_the_next_turn_in_a_new_session(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP, STATES_A_FACT,
                         ProfileUpdate(field="allergies", new_value="crab"))

    await seam.turn("I am allergic to shrimp")
    seam.reconnect()  # the Session ended; the Thread did not
    events = await seam.turn("I am allergic to crab")

    # The second Turn's old value is the first Turn's new one, which it could
    # only have got by reading PostgreSQL again.
    assert "from peanut, shrimp to peanut, shrimp, crab" in answer(events).reply.text
    assert profile(seam).allergies == ["peanut", "shrimp", "crab"]


async def test_a_new_allergy_is_honoured_by_the_next_recommendation(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP, RECOMMENDS)

    await seam.turn("I am allergic to shrimp")
    seam.reconnect()  # no restart of the graph, no reseed of the database
    await seam.turn("what should I have for dinner?")

    # The Profile the Recommendation Turn was handed, not the one the store
    # happens to hold now.
    assert seam.db.loaded[-1].allergies == ["peanut", "shrimp"]


async def test_the_same_food_stated_twice_is_not_added_twice(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP, STATES_A_FACT, SHRIMP)

    await seam.turn("I am allergic to shrimp")
    await seam.turn("I am allergic to shrimp")

    assert profile(seam).allergies == ["peanut", "shrimp"]


# --- an ambiguous statement asks rather than guesses ---------------------------


async def test_a_statement_the_extractor_cannot_pin_to_one_field_goes_to_clarify(seam):
    seam.provider.script(STATES_A_FACT, VAGUE)

    events = await seam.turn("I'm bigger these days")

    nodes = [e.node for e in events if isinstance(e, NodeEvent)]
    assert nodes == ["load_profile", "route", "update_profile", "clarify"]
    assert answer(events).reply.text.endswith("?")
    assert seam.state()["pending_clarification"] == answer(events).reply.text


async def test_an_ambiguous_statement_writes_nothing(seam):
    seam.provider.script(STATES_A_FACT, VAGUE)
    before = profile(seam)

    await seam.turn("I'm bigger these days")

    assert profile(seam) == before


async def test_a_value_the_field_cannot_hold_asks_rather_than_writing_it(seam):
    seam.provider.script(
        STATES_A_FACT, ProfileUpdate(field="age", old_value="24", new_value="quite old")
    )

    events = await seam.turn("I'm quite old now")

    assert profile(seam).age == 24
    assert [e.node for e in events if isinstance(e, NodeEvent)][-1] == "clarify"


# --- the seam, the models, and the metric row ---------------------------------


async def test_the_extraction_is_a_schema_call_and_costs_one_call_on_top_of_routing(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP)

    await seam.turn("I am allergic to shrimp")

    assert [c.model for c in seam.provider.seen] == [SCHEMA_MODEL, SCHEMA_MODEL]


async def test_the_confirmation_is_not_a_prose_call(seam):
    """The three values are known, so the sentence is written, not generated —
    nothing can hallucinate a Profile value into it."""
    seam.provider.script(STATES_A_FACT, SHRIMP)

    await seam.turn("I am allergic to shrimp")

    assert PROSE_MODEL not in [c.model for c in seam.provider.seen]


async def test_the_node_leaves_a_metric_row_naming_the_intent(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP)

    await seam.turn("I am allergic to shrimp")

    row = next(r for r in seam.db.events if r.node == "update_profile")
    assert row.intent == "update_profile"
    assert row.model == SCHEMA_MODEL


async def test_the_extraction_prompt_is_redacted_like_every_other_call(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP)

    await seam.turn("I'm Lou and I am allergic to shrimp")

    for call in seam.provider.seen:
        assert "Lou" not in call.sent
    # What the Coach cannot work without still goes: the Profile it holds now.
    assert "peanut" in seam.provider.seen[-1].sent


def test_no_field_reaches_sql_that_the_extractors_schema_could_not_produce():
    """The column name cannot be a query parameter, so the whitelist at the
    database seam and the schema the model fills are the same tuple."""
    assert set(ProfileUpdate.model_fields["field"].annotation.__args__[0].__args__) == set(
        UPDATABLE_FIELDS
    )
    assert "user_id" not in UPDATABLE_FIELDS and "name" not in UPDATABLE_FIELDS


async def test_the_database_seam_refuses_a_field_that_is_not_a_profile_field():
    """The whitelist runs before the pool is touched, so a bad field never
    reaches a connection, let alone a query string."""
    db = PostgresDatabase(pool=None)

    with pytest.raises(ValueError):
        await db.update_profile("demo-user-1", field="user_id; drop table", value="x")


# --- the trap: the allergy check must not run on this path ---------------------
#
# A correct confirmation of "I am allergic to shrimp" contains the word shrimp.
# An allergy check on this path reads its own confirmation as a violation and
# destroys it. These three fail if a later change puts the check back.


def test_the_allergy_check_does_not_list_this_intent():
    assert "update_profile" not in ALLERGY_CHECKED_INTENTS
    assert set(ALLERGY_CHECKED_INTENTS) == {"recommend", "log_meal"}


async def test_no_node_beyond_the_intent_path_runs_on_an_update_profile_turn(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP)

    events = await seam.turn("I am allergic to shrimp")

    nodes = [e.node for e in events if isinstance(e, NodeEvent)]
    assert nodes == ["load_profile", "route", "update_profile"], (
        "a node was added to the update_profile path; if it is an allergy or "
        "guardrail check, it must not be"
    )


async def test_the_confirmation_still_names_the_allergen_the_user_just_added(seam):
    seam.provider.script(STATES_A_FACT, SHRIMP)

    events = await seam.turn("I am allergic to shrimp")

    reply = answer(events).reply
    assert "shrimp" in reply.text
    assert "peanut" in reply.text  # the allergy the Profile already held, too
    assert isinstance(next(e for e in events if isinstance(e, AnswerEvent)), AnswerEvent)
