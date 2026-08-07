"""The two-Intent Turn, and the one composer, at the agent turn seam.

The sentence the specification names is "I ate two eggs, is that enough
protein". It is one message that carries two Intents, and the second has to read
what the first produced, so the Meal is written before the day is totalled and
the totals the question is answered from already hold it.

The prototype this slice replaces let each Intent path write its own reply. The
second Intent did its work and the User never saw it, which is what
`test_neither_part_of_a_two_intent_reply_is_silent` fails on if it comes back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nutrigraph_agent.db import DayTotal
from nutrigraph_agent.graph import (
    CHECKPOINTED,
    COULD_NOT_COMPOSE,
    INTENT_PATHS,
    IntentResult,
    TurnContext,
    pending_intent,
)
from nutrigraph_agent.meal import NOTHING_COUNTED_TODAY, day_line
from nutrigraph_agent.models import (
    Answer,
    Citation,
    ComposedReply,
    FoodChoice,
    ParsedItem,
    ParsedMeal,
    ReplyPart,
    RouterDecision,
)

from .conftest import PROSE_MODEL, SCHEMA_MODEL, answer
from .fakes import EGG, EGGS_CHUNK

SOURCE = Path(__file__).resolve().parents[1] / "src" / "nutrigraph_agent"

# What the router returns for the sentence the specification names. The order is
# the whole point: the Meal is logged, then the question is answered.
BOTH = RouterDecision(intents=["log_meal", "ask_question"], confidence=0.95)

TWO_EGGS = ParsedMeal(items=[ParsedItem(name="egg", quantity=2, unit="piece")])
CHOSEN_EGG = FoodChoice(fdc_id="748967", reason="the plain whole egg")

CITED = Answer(
    text="Eggs are one of the protein foods the guidelines name, alongside "
    "seafood, beans, and nuts.",
    citations=[Citation(document=EGGS_CHUNK.document, locator=EGGS_CHUNK.locator)],
)

COMPOSED = ComposedReply(
    text="[NAME_1], I logged breakfast: egg, and that puts you at 25.2 g of "
    "protein today. Eggs are one of the protein foods the guidelines name."
)


@pytest.fixture
def two_intents(seam):
    """The sentence the specification names, scripted end to end: the router,
    the parse, the food choice, the cited Answer, and the composer."""
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(BOTH, TWO_EGGS, CHOSEN_EGG, CITED, COMPOSED)
    return seam


# --- the Turn that does two jobs ----------------------------------------------


async def test_the_meal_is_logged_first_and_the_question_answered_after(two_intents):
    events = await two_intents.turn("I ate two eggs, is that enough protein?")

    nodes = [e.node for e in events[:7]]
    assert nodes == [
        "load_profile", "guard", "route",
        "log_meal", "retrieve", "answer_question", "compose_reply",
    ]
    # The Meal was written, not merely talked about.
    assert len(two_intents.db.meals) == 1
    assert two_intents.db.items[0].row.said_as == "egg"


async def test_the_question_is_answered_from_totals_that_hold_the_meal(two_intents):
    await two_intents.turn("I ate two eggs, is that enough protein?")

    # The facts the composer was given. Two eggs at the declared serving is
    # twice the per-100 g figure, and the number reached the composer's prompt.
    composing = two_intents.provider.seen[-1].sent
    assert "25.2 g protein" in composing
    assert "this Meal included" in composing


async def test_neither_part_of_a_two_intent_reply_is_silent(two_intents):
    """The fault the prototype exposed: with each path writing its own reply,
    the second Intent did its work and the User never saw it."""
    events = await two_intents.turn("I ate two eggs, is that enough protein?")

    reply = answer(events).reply
    assert [p.intent for p in reply.parts] == ["log_meal", "ask_question"]
    assert all(p.text.strip() for p in reply.parts)
    # One reply, and it is the composer's, not either path's own.
    assert "logged" in reply.text and "protein" in reply.text


async def test_the_citation_the_question_earned_survives_onto_its_part(two_intents):
    events = await two_intents.turn("I ate two eggs, is that enough protein?")

    question = answer(events).reply.parts[1]
    assert [(c.document, c.locator) for c in question.citations] == [
        (EGGS_CHUNK.document, EGGS_CHUNK.locator)
    ]
    # The Meal part carries none, because logging a Meal makes no Corpus claim.
    assert answer(events).reply.parts[0].citations == []


async def test_exactly_one_node_produces_the_coach_reply(two_intents):
    events = await two_intents.turn("I ate two eggs, is that enough protein?")

    # The node that emitted the answer is the last one that ran, and there is
    # one answer event however many Intents the Turn carried.
    assert [e.node for e in events if getattr(e, "node", None) == "compose_reply"]
    assert len([e for e in events if hasattr(e, "reply")]) == 1


async def test_the_composer_writes_prose_so_it_uses_flash(two_intents):
    await two_intents.turn("I ate two eggs, is that enough protein?")

    models = [c.model for c in two_intents.provider.seen]
    # The router, the parse, the food choice and the Answer fill schemas; only
    # the composer writes for the User, and only it takes the prose tier.
    assert models == [SCHEMA_MODEL] * 4 + [PROSE_MODEL]


async def test_the_user_is_named_in_the_reply_and_never_in_the_prompt(two_intents):
    events = await two_intents.turn("I ate two eggs, is that enough protein?")

    assert "Lou" not in two_intents.provider.seen[-1].sent
    assert "Lou" in answer(events).reply.text


async def test_intent_results_never_enters_the_checkpoint(two_intents):
    await two_intents.turn("I ate two eggs, is that enough protein?")

    state = two_intents.state()
    assert set(state) == set(CHECKPOINTED)
    assert "intent_results" not in state
    # One coach message, not one for each Intent that ran.
    assert [m["role"] for m in state["messages"]] == ["user", "coach"]


# --- the disclaimer a path raised ---------------------------------------------


async def test_a_marking_raised_on_a_path_survives_a_composer_that_dropped_it(seam):
    """The composer is a model, and a model can be brief. A marking saying the
    portion was assumed is not the composer's to drop, so it is put back."""
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        BOTH, TWO_EGGS, CHOSEN_EGG, CITED,
        ComposedReply(text="[NAME_1], two eggs logged. Eggs are a protein food."),
    )

    events = await seam.turn("I ate two eggs, is that enough protein?")

    reply = answer(events).reply
    assert any("my assumption and not a measurement" in d for d in reply.disclaimers)
    assert "my assumption and not a measurement" in reply.text


async def test_a_marking_the_composer_kept_is_not_said_twice(seam):
    kept = (
        "Where you did not give a weight I counted 100 g for one serving, which "
        "is my assumption and not a measurement."
    )
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        BOTH, TWO_EGGS, CHOSEN_EGG, CITED,
        ComposedReply(text=f"[NAME_1], two eggs logged. {kept}"),
    )

    events = await seam.turn("I ate two eggs, is that enough protein?")

    assert answer(events).reply.text.count(kept) == 1


async def test_a_one_intent_turn_keeps_every_marking_without_a_composer_call(seam):
    """One part is handed through. Nothing is joined, so no model is asked to
    rewrite sentences the path assembled from facts."""
    seam.provider.script(
        RouterDecision(intents=["log_meal"], confidence=0.95),
        ParsedMeal(items=[ParsedItem(name="adobo", quantity=1)]),
    )

    events = await seam.turn("I had adobo for lunch")

    reply = answer(events).reply
    assert "stand-in" in reply.text
    assert any("stand-in" in d for d in reply.disclaimers)
    assert [c.model for c in seam.provider.seen] == [SCHEMA_MODEL] * 2


# --- the schema failure and the fallback --------------------------------------


async def test_a_composer_schema_failure_retries_once_with_the_error(seam):
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        BOTH, TWO_EGGS, CHOSEN_EGG, CITED,
        "text was missing",           # the first attempt does not fit the schema
        COMPOSED,                      # the retry does
    )

    events = await seam.turn("I ate two eggs, is that enough protein?")

    prose = [c for c in seam.provider.seen if c.model == PROSE_MODEL]
    # Both attempts are calls in their own right, so both appear in the trace.
    assert len(prose) == 2
    assert "did not fit the schema" in prose[1].sent
    assert "text was missing" in prose[1].sent
    assert answer(events).reply.text.startswith("Lou,")


async def test_a_second_failure_ends_the_turn_on_the_fixed_message(seam):
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        BOTH, TWO_EGGS, CHOSEN_EGG, CITED, "no text", "still no text"
    )

    events = await seam.turn("I ate two eggs, is that enough protein?")

    reply = answer(events).reply
    assert COULD_NOT_COMPOSE in reply.text
    # At most one extra call: two attempts, and no third.
    assert len([c for c in seam.provider.seen if c.model == PROSE_MODEL]) == 2
    # The Meal was written before the composer ran, so nothing was lost with the
    # words, and the record still says both Intents ran.
    assert len(seam.db.meals) == 1
    assert [p.intent for p in reply.parts] == ["log_meal", "ask_question"]


async def test_the_fixed_message_still_carries_the_marking_the_path_raised(seam):
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(BOTH, TWO_EGGS, CHOSEN_EGG, CITED, "no text", "no text")

    events = await seam.turn("I ate two eggs, is that enough protein?")

    assert "my assumption and not a measurement" in answer(events).reply.text


# --- what the router may return ------------------------------------------------


def test_a_third_intent_is_dropped_rather_than_run():
    decision = RouterDecision(
        intents=["log_meal", "ask_question", "recommend"], confidence=0.9
    )

    assert decision.intents == ["log_meal", "ask_question"]


def test_the_same_intent_twice_is_one_intent():
    """The second Intent runs on what the first produced, so the same path twice
    is a loop reading its own output."""
    decision = RouterDecision(intents=["log_meal", "log_meal"], confidence=0.9)

    assert decision.intents == ["log_meal"]


async def test_a_router_that_named_three_still_runs_the_first_two(seam):
    """The drop happens in validation, so the Turn never sees the third and no
    retry is spent on a message the Coach understood perfectly well."""
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        RouterDecision(
            intents=["log_meal", "ask_question", "update_profile"], confidence=0.95
        ),
        TWO_EGGS, CHOSEN_EGG, CITED, COMPOSED,
    )

    events = await seam.turn("I ate two eggs, is that enough protein, I weigh 78 kg")

    assert [p.intent for p in answer(events).reply.parts] == ["log_meal", "ask_question"]
    assert "update_profile" not in [getattr(e, "node", None) for e in events]


# --- the bookkeeping ----------------------------------------------------------


def test_the_position_in_the_list_is_the_count_of_results(deps_free_context):
    ctx = deps_free_context
    ctx.decision = RouterDecision(intents=["log_meal", "ask_question"], confidence=0.9)

    assert pending_intent(ctx) == "log_meal"
    ctx.intent_results.append(IntentResult(intent="log_meal", text="logged"))
    assert pending_intent(ctx) == "ask_question"
    ctx.intent_results.append(IntentResult(intent="ask_question", text="answered"))
    assert pending_intent(ctx) is None


def test_no_intent_path_builds_a_coach_reply_of_its_own():
    """The rule the prototype broke, read off the source tree rather than off
    one Turn. An Intent path that builds its own `CoachReply` is a path that can
    end the Turn with its own words, and then the other Intent is silent.

    Two modules may build one, and neither does it on an Intent path:
    `guardrail`, whose Refusal and safe message are templates in code rather
    than anybody's composition, and `graph`, exactly twice — `compose_reply`,
    which is the composer, and `clarify`, which asks the graph's one interrupt
    before any Intent has run.
    """
    reaching = {
        path.name
        for path in SOURCE.glob("*.py")
        # `models.py` is where the class is declared, which is not building one.
        if path.name != "models.py" and "CoachReply(" in path.read_text(encoding="utf-8")
    }
    assert reaching == {"graph.py", "guardrail.py"}, (
        "an Intent path builds its own reply; append an IntentResult instead"
    )

    graph = (SOURCE / "graph.py").read_text(encoding="utf-8")
    assert graph.count("CoachReply(") == 2


def test_every_intent_the_router_may_name_has_a_path_or_the_stub():
    """A new path is one line here and a node. Until then the stub reports it,
    and it still appends a result like any other path."""
    assert set(INTENT_PATHS) <= set(
        ["log_meal", "ask_question", "review_day", "recommend", "update_profile"]
    )


@pytest.fixture
def deps_free_context() -> TurnContext:
    """The bookkeeping alone. No database, no provider, no graph."""
    from uuid import uuid4

    return TurnContext(deps=None, turn_id=uuid4(), raw_message="")  # type: ignore[arg-type]


# --- the day line -------------------------------------------------------------


def test_the_day_line_names_a_short_total_rather_than_summing_it_as_complete():
    total = DayTotal(
        counted=2,
        not_counted=1,
        values={"kcal": 286.0, "protein_g": 25.2},
        missing={"kcal": 0, "protein_g": 1},
    )

    line = day_line(total)

    assert "286.0 kcal" in line and "25.2 g protein" in line
    assert "protein total is short" in line
    assert "1 food logged today could not be counted" in line


def test_a_day_with_nothing_counted_says_so_rather_than_reporting_zeros():
    assert day_line(DayTotal()) == NOTHING_COUNTED_TODAY
    assert "0" not in NOTHING_COUNTED_TODAY


def test_a_reply_part_carries_the_intent_that_produced_it():
    part = ReplyPart(intent="log_meal", text="logged")

    assert (part.intent, part.citations) == ("log_meal", [])
