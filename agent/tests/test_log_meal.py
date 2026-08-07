"""The `log_meal` Intent at the agent turn seam, with Gemini, FoodData Central
and the database faked. Everything between them is real: the router, the
redaction wrapper, the matching order, the scaling, and the answer.

The local dish table the tests match against is the real seed file, so a proxy
row is a real proxy row and a null is a real null.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pytest

from nutrigraph_agent.food import candidate
from nutrigraph_agent.guardrail import scan_reply
from nutrigraph_agent.meal import (
    DECLARED_SERVING_G,
    MANILA,
    TELL_ME,
    day_bounds,
    grams_for,
    meal_type_for,
    normalize,
    scale,
    stand_in,
)
from nutrigraph_agent.models import (
    MEAL_TYPES,
    FoodChoice,
    ParsedItem,
    ParsedMeal,
    RouterDecision,
)

from .conftest import SCHEMA_MODEL, answer
from .fakes import EGG, EGG_FOOD
from .test_filipino_dishes import DISHES

LOGGED = RouterDecision(intents=["log_meal"], confidence=0.95)

EGG_NOODLE = {
    "fdcId": 168874,
    "description": "Noodles, egg, dry, enriched",
    "dataType": "SR Legacy",
    "foodNutrients": [{"nutrientNumber": "208", "value": 384.0}],
}
CHOSEN_EGG = FoodChoice(fdc_id="748967", reason="the plain whole egg, not a noodle")

QUAIL_FOOD = {
    "fdcId": 172194,
    "description": "Egg, quail, whole, fresh, raw",
    "foodNutrients": [
        {"nutrientNumber": "208", "value": 158.0},
        {"nutrientNumber": "203", "value": 13.05},
    ],
}
CHOSEN_QUAIL = FoodChoice(fdc_id="172194", reason="quail eggs, which is what kwek kwek is")


def eggs_and_pandesal() -> ParsedMeal:
    return ParsedMeal(
        items=[
            ParsedItem(name="egg", quantity=2, unit="piece", confidence=0.98),
            ParsedItem(name="pandesal", quantity=1, unit="piece", confidence=0.96),
        ]
    )


@pytest.fixture
def eggs(seam):
    """The sentence the specification names, scripted end to end."""
    seam.food.results = {"egg": [EGG, candidate(EGG_NOODLE)]}
    seam.provider.script(LOGGED, eggs_and_pandesal(), CHOSEN_EGG)
    return seam


# --- one sentence, one Meal ---------------------------------------------------


async def test_two_eggs_and_pandesal_is_one_meal_with_two_items_and_a_meal_type(eggs):
    await eggs.turn("I ate two eggs and pandesal")

    assert len(eggs.db.meals) == 1
    meal = eggs.db.meals[0]
    assert meal.meal_type in MEAL_TYPES
    items = [item.row for item in eggs.db.items]
    assert [(i.said_as, i.quantity, i.unit) for i in items] == [
        ("egg", 2.0, "piece"),
        ("pandesal", 1.0, "piece"),
    ]
    assert [i.ordinal for i in items] == [0, 1]
    assert all(i.status == "matched" for i in items)


async def test_the_meal_type_comes_from_the_users_words_when_they_name_one():
    parsed = ParsedMeal(items=[], meal_type="dinner")

    # Nine in the morning, and the User said dinner. Their words win.
    at = datetime(2026, 8, 7, 9, 0, tzinfo=MANILA)
    assert meal_type_for(parsed, at) == ("dinner", "the User's own words")


@pytest.mark.parametrize(
    "hour,expected",
    [(2, "snack"), (7, "breakfast"), (12, "lunch"), (19, "dinner"), (23, "snack")],
)
def test_the_meal_type_comes_from_the_clock_when_they_do_not(hour, expected):
    at = datetime(2026, 8, 7, hour, 0, tzinfo=MANILA)

    meal_type, why = meal_type_for(ParsedMeal(items=[]), at)

    assert meal_type == expected
    assert "Manila" in why


# --- the matching order -------------------------------------------------------


async def test_a_local_dish_matches_from_the_local_table_with_no_fdc_call(seam):
    """The local table runs before every FoodData Central call, and when it
    answers there is no call at all."""
    seam.provider.script(
        LOGGED, ParsedMeal(items=[ParsedItem(name="pandesal", quantity=2, unit="piece")])
    )

    await seam.turn("I had two pandesal")

    assert seam.food.searched == []
    item = seam.db.items[0].row
    assert item.source == "local"
    assert item.local_food_id is not None
    # The router and the parse, and no third call: nothing had to be chosen.
    assert [c.model for c in seam.provider.seen] == [SCHEMA_MODEL] * 2


async def test_an_alias_and_a_spelling_the_user_used_reach_the_same_dish(seam):
    seam.provider.script(
        LOGGED, ParsedMeal(items=[ParsedItem(name="Pork Adobo!", quantity=1)])
    )

    await seam.turn("I had pork adobo")

    assert seam.db.items[0].row.food_name == "Adobo (pork)"
    assert seam.food.searched == []


async def test_a_food_absent_from_the_local_table_is_searched_then_chosen(eggs):
    await eggs.turn("I ate two eggs and pandesal")

    assert eggs.food.searched == ["egg"]
    egg = eggs.db.items[0].row
    assert (egg.source, egg.fdc_id) == ("fdc", "748967")
    assert egg.food_name == "Egg, whole, raw, fresh"


async def test_the_chosen_candidate_and_the_reason_are_recorded(eggs, caplog):
    with caplog.at_level(logging.INFO, logger="nutrigraph.agent.meal"):
        await eggs.turn("I ate two eggs and pandesal")

    note = eggs.db.items[0].row.match_note
    assert "748967" in note
    assert CHOSEN_EGG.reason in note
    assert "2 candidates" in note
    # The same string in the trace, so a match is auditable from either end.
    assert any(note in record.getMessage() for record in caplog.records)


async def test_both_schema_calls_the_node_made_land_on_one_metric_row(eggs):
    await eggs.turn("I ate two eggs and pandesal")

    row = next(r for r in eggs.db.events if r.node == "log_meal")
    assert row.intent == "log_meal"
    assert row.model == SCHEMA_MODEL
    # The parse and the one choice, added together: two calls of eleven tokens.
    assert (row.input_tokens, row.output_tokens) == (22, 14)


# --- when nothing matches -----------------------------------------------------


async def test_a_food_that_matches_nothing_is_saved_uncounted_and_the_meal_is_written(seam):
    seam.provider.script(
        LOGGED,
        ParsedMeal(
            items=[
                ParsedItem(name="pandesal", quantity=1),
                ParsedItem(name="kwek kwek", quantity=3),
            ]
        ),
    )

    events = await seam.turn("I ate pandesal and three kwek kwek")

    assert len(seam.db.meals) == 1, "the Meal is written whatever the matching found"
    unmatched = seam.db.items[1].row
    assert unmatched.said_as == "kwek kwek"
    assert unmatched.status == "unmatched"
    assert unmatched.values == {} and unmatched.nutrients is None
    assert (unmatched.food_name, unmatched.source, unmatched.grams) == (None, None, None)
    # Nothing the User said is lost: the quantity is kept too.
    assert unmatched.quantity == 3.0
    text = answer(events).reply.text
    assert "pandesal" in text and "kwek kwek" in text
    assert "not counted" in text and TELL_ME in text


async def test_a_food_data_source_that_is_down_does_not_lose_the_meal(seam):
    """FoodData Central answers a search that matches nothing with an HTTP 400,
    which was seen live. Losing what the User said would be the worse failure by
    far, so the food is uncounted and the record says why."""

    async def broken(query, *, limit=10):
        raise ConnectionError("the search endpoint did not answer")

    seam.food.search = broken
    seam.provider.script(
        LOGGED,
        ParsedMeal(items=[ParsedItem(name="pandesal"), ParsedItem(name="taho")]),
    )

    events = await seam.turn("I had pandesal and taho")

    assert len(seam.db.meals) == 1
    pandesal, taho = (item.row for item in seam.db.items)
    assert pandesal.status == "matched", "the local table needs no web request"
    assert taho.status == "unmatched"
    assert "could not be asked" in taho.match_note
    assert "taho" in answer(events).reply.text


async def test_a_candidate_list_with_nothing_right_in_it_is_not_matched_anyway(seam):
    """A wrong match puts the wrong numbers in the User's day and says nothing.
    A null choice says so out loud instead."""
    seam.food.results = {"taho": [candidate(EGG_NOODLE)]}
    seam.provider.script(
        LOGGED,
        ParsedMeal(items=[ParsedItem(name="taho")]),
        FoodChoice(fdc_id=None, reason="dry egg noodles are not taho"),
    )

    await seam.turn("I had taho")

    item = seam.db.items[0].row
    assert item.status == "unmatched" and item.values == {}
    assert "dry egg noodles are not taho" in item.match_note


async def test_logging_a_meal_never_interrupts_the_turn(seam):
    """`clarify` is the only interrupt in the graph, and meal logging is not it."""
    seam.provider.script(LOGGED, ParsedMeal(items=[ParsedItem(name="kwek kwek")]))

    events = await seam.turn("I ate kwek kwek")

    assert [e.node for e in events[:4]] == ["load_profile", "guard", "route", "log_meal"]
    assert "clarify" not in [getattr(e, "node", None) for e in events]
    assert seam.state()["pending_clarification"] is None


async def test_a_message_with_no_food_in_it_still_ends_the_turn(seam):
    seam.provider.script(LOGGED, ParsedMeal(items=[]))

    events = await seam.turn("I ate")

    assert seam.db.meals == []
    assert TELL_ME in answer(events).reply.text


async def test_an_entry_the_parse_is_not_sure_is_a_food_is_dropped(seam):
    seam.provider.script(
        LOGGED,
        ParsedMeal(
            items=[
                ParsedItem(name="pandesal", confidence=0.9),
                ParsedItem(name="a bad day", confidence=0.2),
            ]
        ),
    )

    await seam.turn("I had pandesal after a bad day")

    assert [i.row.said_as for i in seam.db.items] == ["pandesal"]


# --- the correction -----------------------------------------------------------


async def test_a_follow_up_correcting_an_unmatched_food_updates_the_item_and_the_total(seam):
    seam.provider.script(LOGGED, ParsedMeal(items=[ParsedItem(name="kwek kwek", quantity=1)]))
    await seam.turn("I ate kwek kwek")
    logged = seam.db.items[0]
    assert logged.row.status == "unmatched"

    seam.food.results = {"quail egg": [candidate(QUAIL_FOOD)]}
    seam.provider.script(
        LOGGED,
        ParsedMeal(
            items=[
                ParsedItem(name="quail egg", quantity=1, unit="piece", corrects="kwek kwek")
            ]
        ),
        CHOSEN_QUAIL,
    )
    events = await seam.turn("the kwek kwek was quail eggs")

    # The Item was filled in where it stood. No second Meal, and no second Item.
    assert len(seam.db.meals) == 1 and len(seam.db.items) == 1
    assert seam.db.items[0].meal_item_id == logged.meal_item_id
    corrected = seam.db.items[0].row
    assert corrected.status == "matched"
    assert corrected.fdc_id == "172194"
    assert corrected.values["kcal"] == pytest.approx(158.0)

    start, end = day_bounds(datetime.now(MANILA))
    total = await seam.db.day_total("demo-user-1", start=start, end=end)
    assert (total.counted, total.not_counted) == (1, 0)
    assert total.values["kcal"] == pytest.approx(158.0)
    assert "earlier" in answer(events).reply.text


async def test_a_correction_naming_a_food_this_user_never_logged_is_a_new_item(seam):
    """The parse names a food word; the identifier is looked up here, so a model
    cannot reach a row that is not this User's."""
    seam.food.results = {"quail egg": [candidate(QUAIL_FOOD)]}
    seam.provider.script(
        LOGGED,
        ParsedMeal(items=[ParsedItem(name="quail egg", corrects="something else")]),
        CHOSEN_QUAIL,
    )

    await seam.turn("the something else was quail eggs")

    assert len(seam.db.meals) == 1
    assert seam.db.items[0].row.said_as == "quail egg"


# --- what the numbers are -----------------------------------------------------


async def test_a_proxy_value_is_marked_in_the_answer(seam):
    """Presenting a canned adobo's numbers as home-cooked adobo without saying
    so is the failure `value_kind` exists to prevent."""
    seam.provider.script(LOGGED, ParsedMeal(items=[ParsedItem(name="adobo", quantity=1)]))

    events = await seam.turn("I had adobo for lunch")

    item = seam.db.items[0].row
    assert item.value_kind == "proxy"
    text = answer(events).reply.text
    assert "stand-in" in text
    assert "CANNED COMMERCIAL PRODUCT" in text


async def test_a_calculated_value_is_not_presented_as_measured(seam):
    seam.provider.script(LOGGED, ParsedMeal(items=[ParsedItem(name="sinigang", quantity=1)]))

    events = await seam.turn("I had sinigang")

    assert seam.db.items[0].row.value_kind == "calculated"
    text = answer(events).reply.text
    assert "calculated from component foods rather than measured" in text
    assert "understated" in text


async def test_a_direct_value_is_marked_as_nothing_at_all(seam):
    seam.provider.script(LOGGED, ParsedMeal(items=[ParsedItem(name="pandesal", quantity=1)]))

    events = await seam.turn("I had pandesal")

    assert seam.db.items[0].row.value_kind == "direct"
    text = answer(events).reply.text
    assert "stand-in" not in text and "calculated" not in text


async def test_an_assumed_portion_is_said_to_be_an_assumption(seam):
    seam.provider.script(
        LOGGED, ParsedMeal(items=[ParsedItem(name="pandesal", quantity=2, unit="piece")])
    )

    events = await seam.turn("I had two pandesal")

    item = seam.db.items[0].row
    assert item.portion_assumed and item.grams == 2 * DECLARED_SERVING_G
    assert "my assumption and not a measurement" in answer(events).reply.text


async def test_a_weight_the_user_stated_is_not_an_assumption(seam):
    seam.provider.script(
        LOGGED, ParsedMeal(items=[ParsedItem(name="pandesal", quantity=60, unit="g")])
    )

    events = await seam.turn("I had 60 g of pandesal")

    item = seam.db.items[0].row
    assert (item.grams, item.portion_assumed) == (60.0, False)
    assert "assumption" not in answer(events).reply.text


# --- the nulls ----------------------------------------------------------------


async def test_a_null_the_source_prints_nothing_for_is_never_a_zero(seam):
    """PhilFCT prints no sodium for the dinuguan row. The column is null, the
    day total says it is short, and nothing anywhere writes a zero."""
    seam.provider.script(LOGGED, ParsedMeal(items=[ParsedItem(name="dinuguan", quantity=1)]))

    events = await seam.turn("I had dinuguan")

    item = seam.db.items[0].row
    assert item.status == "matched"
    assert "sodium_mg" not in item.values, "a missing value is not a zero"
    assert item.values["kcal"] > 0

    start, end = day_bounds(datetime.now(MANILA))
    total = await seam.db.day_total("demo-user-1", start=start, end=end)
    assert total.missing["sodium_mg"] == 1
    assert not total.complete("sodium_mg") and total.complete("kcal")
    assert "prints no sodium" in answer(events).reply.text


async def test_the_whole_source_response_is_kept_beside_the_six_columns(eggs):
    await eggs.turn("I ate two eggs and pandesal")

    egg, pandesal = (item.row for item in eggs.db.items)
    # The FoodData Central food object, and the whole seed row, as they came.
    assert egg.nutrients == EGG_FOOD
    assert pandesal.nutrients["philfct_food_id"] == "A042"
    assert pandesal.nutrients["per_100g"]["proximates"]
    # Two eggs at the declared serving: twice the per-100 g figure.
    assert egg.values["kcal"] == pytest.approx(286.0)
    assert egg.values["protein_g"] == pytest.approx(25.2)


# --- the units ----------------------------------------------------------------


def test_a_missing_nutrient_stays_missing_through_the_scaling():
    scaled = scale({"kcal": 100.0}, 250.0)

    assert scaled == {"kcal": 250.0}
    assert "sodium_mg" not in scaled


@pytest.mark.parametrize(
    "quantity,unit,grams,assumed",
    [
        (2, "piece", 2 * DECLARED_SERVING_G, True),
        (None, None, DECLARED_SERVING_G, True),
        (1.5, "cup", 1.5 * DECLARED_SERVING_G, True),
        (250, "g", 250.0, False),
        (0.5, "kg", 500.0, False),
        (200, "ml", 200.0, False),
    ],
)
def test_what_a_portion_weighs_and_whether_it_was_assumed(quantity, unit, grams, assumed):
    assert grams_for(ParsedItem(name="x", quantity=quantity, unit=unit)) == (grams, assumed)


def test_one_spelling_for_a_name_however_the_user_wrote_it():
    assert normalize("Kare-kare (beef)") == "kare kare beef"
    assert normalize("  PANDESAL!  ") == "pandesal"


def test_every_marking_a_dish_could_carry_survives_the_reply_scan():
    """The answer is scanned for medical claims before it is sent, and a text
    that fails is replaced whole. FNRI's vocabulary and the scan's collide —
    tapa is the 'dried/cured jerky form' — so every dish in the table is checked
    here, rather than a User discovering that logging tapa erases the answer."""
    for dish in DISHES:
        if dish["value_kind"] == "direct":
            continue
        marking = stand_in(dish["source_note"])
        assert marking and scan_reply(marking) is None, dish["name"]


def test_a_note_the_scan_would_block_still_names_what_stood_in():
    blocked = (
        "PROXY: PhilFCT F243 'Pork jerky' (Tapa, baboy) is the dried/cured jerky "
        "form, not the pan-fried tapa served in a tapsilog. Values per 100 g."
    )

    assert stand_in(blocked) == "the PhilFCT entry 'Pork jerky'."


def test_a_day_starts_at_midnight_where_the_user_ate():
    start, end = day_bounds(datetime(2026, 8, 7, 23, 30, tzinfo=MANILA))

    assert (start.hour, start.tzinfo) == (0, MANILA)
    assert end - start == timedelta(days=1)
    assert start.day == 7
