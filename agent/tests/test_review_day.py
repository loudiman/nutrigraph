"""The day, reviewed: the totals, the targets the Goal produces, and the gap.

The honesty rule is what most of this file is about. An Item the Coach could not
match contributes nothing to the sum, so the review names those foods and says
the total is short by an unknown amount; a day with nothing logged says so
rather than presenting a set of zeros as a result; and a nutrient the source
does not print stays absent instead of being counted as none.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from nutrigraph_agent.db import DAY_TOTAL, MealItemRow
from nutrigraph_agent.food import COLUMNS
from nutrigraph_agent.meal import MANILA
from nutrigraph_agent.models import (
    UPDATABLE_FIELDS,
    ComposedReply,
    DayRequest,
    Profile,
    RouterDecision,
)
from nutrigraph_agent.review import DAILY_SHIFT, review_day, targets_for

from .conftest import (
    PROSE_MODEL,
    SCHEMA_MODEL,
    TurnSeam,
    also_calls_the_provider,
    answer,
)
from .fakes import DEMO_PROFILE, SUGGESTED, FakeDatabase, FakeProvider

# A fixed clock, so "today" is the same day every time this file runs and a
# test can put a Meal on the day before it. The seam fixture below hands this
# same clock to the Turn: a Meal on a fixed day and a Turn reading the wall
# clock agree about "today" on exactly one date, and go red on every other.
NOW = datetime(2026, 8, 8, 19, 30, tzinfo=MANILA)
YESTERDAY = NOW - timedelta(days=1)

# 200 g of pandesal, as the local dish table scales it. Fibre and sodium are
# absent on purpose: PhilFCT prints nothing for them on part of its items.
PANDESAL = {"kcal": 618.0, "protein_g": 18.0, "fat_g": 6.0, "carb_g": 122.0}

# A second food that does carry the two the first one lacks.
EGGS = {
    "kcal": 143.0, "protein_g": 12.6, "fat_g": 9.5, "carb_g": 0.7,
    "fibre_g": 0.0, "sodium_mg": 142.0,
}


def matched(said_as: str, values: dict[str, float], *, ordinal: int = 0) -> MealItemRow:
    return MealItemRow(
        ordinal=ordinal, said_as=said_as, status="matched", grams=100.0,
        source="local", food_name=said_as.title(), value_kind="direct",
        match_note="the local Filipino dish table", values=values,
    )


def unmatched(said_as: str, *, ordinal: int = 1) -> MealItemRow:
    """An Item the Coach could not attach to a Food. No nutrient value at all —
    not a zero, and not a guess."""
    return MealItemRow(ordinal=ordinal, said_as=said_as, status="unmatched")


async def a_day(db: FakeDatabase, *rows: MealItemRow, at: datetime = NOW) -> None:
    from uuid import uuid4

    await db.store_meal(
        user_id=DEMO_PROFILE.user_id, turn_id=uuid4(), eaten_at=at,
        meal_type="breakfast", items=list(rows),
    )


async def review(
    db: FakeDatabase,
    *,
    message: str = "how did I eat today?",
    days_ago: int = 0,
    profile: Profile = DEMO_PROFILE,
):
    """The review path, with the provider and the database faked, and the real
    redaction wrapper between the two."""
    provider = FakeProvider().script(DayRequest(days_ago=days_ago))
    turn = provider.models(
        schema_model=SCHEMA_MODEL, prose_model=PROSE_MODEL
    ).for_turn(known_names=[profile.name])
    return await review_day(
        db=db, turn=turn, profile=profile, message=message, now=NOW
    )


@pytest.fixture
def db() -> FakeDatabase:
    return FakeDatabase()


@pytest.fixture(params=["as the graph is", "with a new call elsewhere in the graph"])
def seam(request, monkeypatch) -> TurnSeam:
    """This file's seam: the clock pinned to `NOW`, so a Meal put on that day is
    on the day the Turn calls today.

    Parametrized, so every seam test below also runs against a Turn that makes a
    provider call the graph does not make. That is what #38 did to this file's
    script, and what #34 did to the ladder tests in #54: a scripted answer read
    by position lands on the next node along. Read by schema it does not move.
    """
    seam = TurnSeam(
        db=FakeDatabase(),
        provider=FakeProvider(),
        checkpointer=InMemorySaver(),
        now=lambda: NOW,
    )
    if request.param == "as the graph is":
        return seam
    return also_calls_the_provider(seam, monkeypatch)


# --- the targets ------------------------------------------------------------


def test_the_targets_are_derived_from_the_profile_and_the_goal():
    """The User states facts about themselves and a body-weight Goal. Every
    daily number follows from those two, and the User calculates nothing."""
    targets = targets_for(DEMO_PROFILE).values

    assert set(targets) == set(COLUMNS)
    # Mifflin-St Jeor for a 24-year-old man of 78 kg and 172 cm, at the light
    # activity multiplier, less the daily shift a Goal below the current weight
    # spreads over half a kilogram a week.
    assert targets["kcal"] == pytest.approx(1740.0 * 1.375 - DAILY_SHIFT)
    # Protein is per kilogram of the weight the Goal names, not the current one.
    assert targets["protein_g"] == pytest.approx(1.6 * 72)


def test_the_user_supplies_no_target_directly():
    """`target_weight_kg` is the Goal, and it is the only target-shaped thing a
    User may state. No daily energy or macronutrient target is settable."""
    assert [f for f in UPDATABLE_FIELDS if f.startswith("target")] == ["target_weight_kg"]


def test_a_goal_above_the_current_weight_raises_the_energy_target():
    gaining = DEMO_PROFILE.model_copy(update={"target_weight_kg": 84.0})

    assert targets_for(gaining).values["kcal"] - targets_for(DEMO_PROFILE).values[
        "kcal"
    ] == pytest.approx(2 * DAILY_SHIFT)


def test_a_profile_that_cannot_produce_targets_says_what_it_is_short_of():
    thin = DEMO_PROFILE.model_copy(update={"height_cm": None, "activity_level": None})

    targets = targets_for(thin)

    assert targets.values == {}
    assert sorted(targets.missing) == ["how active you are", "your height"]


async def test_a_review_without_targets_gives_the_total_and_no_gap(db):
    thin = DEMO_PROFILE.model_copy(update={"height_cm": None})
    await a_day(db, matched("pandesal", PANDESAL))

    day = await review(db, profile=thin)

    assert day.target_delta == {}
    assert "I cannot work out your targets until I know your height" in day.text
    assert "618 kcal" in day.text


# --- the totals and the gap -------------------------------------------------


async def test_how_did_i_eat_today_returns_the_totals_for_the_six_nutrients(db):
    await a_day(db, matched("pandesal", PANDESAL), matched("egg", EGGS, ordinal=1))

    day = await review(db)

    assert set(day.totals) == set(COLUMNS)
    assert day.totals["kcal"] == pytest.approx(618.0 + 143.0)
    assert day.totals["sodium_mg"] == pytest.approx(142.0)


async def test_the_review_states_the_gap_against_the_target_for_each_nutrient(db):
    await a_day(db, matched("pandesal", PANDESAL), matched("egg", EGGS, ordinal=1))

    day = await review(db)
    targets = targets_for(DEMO_PROFILE).values

    assert set(day.target_delta) == set(COLUMNS)
    assert day.target_delta["kcal"] == pytest.approx(761.0 - targets["kcal"])
    text = day.text
    assert "Against your targets" in text
    for nutrient in ("energy", "protein", "fat", "carbohydrate", "fibre", "sodium"):
        assert nutrient in text
    # The direction, not only the size: 761 kcal against a target near 1,842.
    assert "energy is 1,082 kcal under" in text


async def test_every_value_the_review_states_is_metric(db):
    await a_day(db, matched("pandesal", PANDESAL), matched("egg", EGGS, ordinal=1))

    text = (await review(db)).text

    assert " kcal" in text and " g of" in text and " mg of" in text
    for imperial in ("oz", "ounce", "lb", "pound", "cal/oz"):
        assert imperial not in text


# --- the honesty rule -------------------------------------------------------


async def test_an_unmatched_item_contributes_nothing_and_is_named(db):
    await a_day(db, matched("pandesal", PANDESAL), unmatched("kwek kwek"))

    day = await review(db)

    # The sum is the matched Item's alone. The unmatched one added nothing —
    # not a zero, and not a guess at what it might have been.
    assert day.totals["kcal"] == pytest.approx(618.0)
    assert day.unmatched == ["kwek kwek"]
    assert "kwek kwek" in day.text


async def test_a_total_holding_an_unmatched_food_says_it_is_short_by_an_unknown_amount(db):
    await a_day(db, matched("pandesal", PANDESAL), unmatched("kwek kwek"))

    day = await review(db)

    assert "short by an unknown amount" in day.text
    # And it is a marking, not only a sentence: the composer carries it as a
    # disclaimer and puts it back if the words it wrote dropped it.
    assert any("short by an unknown amount" in d for d in day.disclaimers)
    assert any("kwek kwek" in d for d in day.disclaimers)


async def test_a_day_of_nothing_but_unmatched_food_gives_no_total_at_all(db):
    await a_day(db, unmatched("kwek kwek", ordinal=0))

    day = await review(db)

    assert day.totals == {}
    text = day.text
    # Named, and said plainly — with no total above for it to be short against.
    assert "the only thing you logged today was kwek kwek" in text
    assert "no total to give you at all" in text
    assert "the total above" not in text


async def test_a_day_with_no_meals_says_so_rather_than_reporting_zeros(db):
    day = await review(db)

    assert day.totals == {} and day.target_delta == {} and day.unmatched == []
    text = day.text
    assert "you have logged nothing for today" in text
    # Nothing logged is a different fact from nothing eaten, so there is not a
    # number anywhere in this answer to be read as a result.
    assert not any(character.isdigit() for character in text)
    # The whole answer is the marking, so a composer joining it to another part
    # cannot reduce it to silence.
    assert day.disclaimers == [text]


async def test_a_nutrient_the_source_does_not_print_is_absent_never_zero(db):
    await a_day(db, matched("pandesal", PANDESAL))

    day = await review(db)

    assert "sodium_mg" not in day.totals and "fibre_g" not in day.totals
    # And no gap is stated against a target for a nutrient with no total.
    assert "sodium_mg" not in day.target_delta
    # Nothing carried either, which is a different fact from a total of zero.
    assert (
        "My source prints no fibre and sodium for anything you ate"
        in day.text
    )


async def test_a_total_the_source_only_partly_covers_says_it_is_short(db):
    """One food carries sodium and the other does not, so the sodium total is a
    real number that is nonetheless incomplete. It says so."""
    await a_day(db, matched("pandesal", PANDESAL), matched("egg", EGGS, ordinal=1))

    day = await review(db)

    assert day.total.complete("kcal") and not day.total.complete("sodium_mg")
    assert "My source prints no fibre and sodium for part of what you ate" in day.text


# --- which day, and where it is read from -----------------------------------


async def test_a_question_about_an_earlier_day_reads_that_days_meals(db):
    await a_day(db, matched("pandesal", PANDESAL), at=NOW)
    await a_day(db, matched("egg", EGGS), at=YESTERDAY)

    day = await review(db, message="how did I eat yesterday?", days_ago=1)

    assert day.totals["kcal"] == pytest.approx(143.0)
    assert day.day.date() == YESTERDAY.date()
    assert "yesterday you have" in day.text


async def test_the_totals_come_from_one_sql_sum_over_the_numeric_columns(db):
    """One query, one sum for each column, and no second read of the snapshot."""
    assert DAY_TOTAL.lower().count("select") == 1
    assert all(f"sum(mi.{column})" in DAY_TOTAL for column in COLUMNS)
    assert "nutrients" not in DAY_TOTAL

    await a_day(db, matched("pandesal", PANDESAL))
    calls: list[tuple[datetime, datetime]] = []
    summed = db.day_total

    async def counted(user_id, *, start, end):
        calls.append((start, end))
        return await summed(user_id, start=start, end=end)

    db.day_total = counted  # type: ignore[method-assign]
    await review(db)

    assert len(calls) == 1
    # The day the User is living in, bounded in Manila.
    assert calls[0][1] - calls[0][0] == timedelta(days=1)


def test_the_review_never_re_parses_the_nutrient_snapshot():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "nutrigraph_agent" / "review.py"
    ).read_text(encoding="utf-8")

    assert "nutrients" not in source


# --- the turn seam ----------------------------------------------------------


async def test_the_review_is_reachable_from_the_turn_seam(seam):
    await a_day(seam.db, matched("pandesal", PANDESAL), unmatched("kwek kwek"))
    seam.provider.script(
        RouterDecision(intents=["review_day"], confidence=0.95), DayRequest(days_ago=0)
    )

    events = await seam.turn("how did I eat today?")

    assert [e.node for e in events[:5]] == [
        "load_profile", "guard", "route", "review_day", "compose_reply",
    ]
    reply = answer(events).reply
    # One part, so there is nothing to join and no composer call: the sentences
    # this path assembled from facts reach the User as they were written.
    assert reply.text.startswith("Lou, today you have 618 kcal of energy")
    assert "kwek kwek" in reply.text and "short by an unknown amount" in reply.text
    assert [p.intent for p in reply.parts] == ["review_day"]
    assert [r.intent for r in seam.db.events if r.node == "review_day"] == ["review_day"]


async def test_a_marking_the_composer_dropped_is_put_back(seam):
    """The composer writes the words on a two-Intent Turn, and a model being
    brief must not be able to drop the sentence that says the total is short."""
    await a_day(seam.db, matched("pandesal", PANDESAL), unmatched("kwek kwek"))
    seam.provider.script(
        RouterDecision(intents=["review_day", "recommend"], confidence=0.95),
        DayRequest(days_ago=0),
        SUGGESTED,
        # Every marking gone, and a total presented as if it were the whole day.
        ComposedReply(text="Lou, you have had 618 kcal today. Have some fish."),
    )

    events = await seam.turn("how did I eat, and what should I eat next?")

    reply = answer(events).reply
    assert "kwek kwek" in reply.text
    assert "short by an unknown amount" in reply.text
    assert any("short by an unknown amount" in d for d in reply.disclaimers)
    assert [p.intent for p in reply.parts] == ["review_day", "recommend"]


async def test_a_day_with_no_meals_says_so_even_when_the_composer_writes_the_words(seam):
    """The composer joined another part's numbers in, so the answer holds digits
    the review did not produce. What may not happen is the day reading as a
    result, and the sentence that says it is not one survives whole."""
    seam.provider.script(
        RouterDecision(intents=["review_day", "recommend"], confidence=0.95),
        DayRequest(days_ago=0),
        SUGGESTED,
        ComposedReply(text="Lou, your day is at 0 kcal. Try 2 eggs."),
    )

    events = await seam.turn("how did I eat, and what should I eat next?")

    assert "you have logged nothing for today" in answer(events).reply.text
