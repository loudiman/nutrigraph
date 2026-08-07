"""What to eat next, at the agent turn seam.

**The order is what these tests are about.** Code finds and the model ranks, so
the assertions that matter are the ones about what reached the provider: the
candidate list it was shown, the foods it was allowed to name, and what happens
when it names one it was not. A node is never tested on its own.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from nutrigraph_agent.db import DayTotal
from nutrigraph_agent.meal import MANILA
from nutrigraph_agent.models import (
    ComposedReply,
    DayRequest,
    FoodChoice,
    ParsedItem,
    ParsedMeal,
    Profile,
    ProfileUpdate,
    Recommendation,
    RouterDecision,
)
from nutrigraph_agent.recommend import (
    COLD_START,
    DIET_CONFLICTS,
    RANKED,
    InventedFood,
    check_foods,
    gap_for,
)
from nutrigraph_agent.review import targets_for

from .conftest import PROSE_MODEL, SCHEMA_MODEL, answer
from .fakes import DEMO_PROFILE, EGG, SUGGESTED, StoredItem, StoredMeal

WANTS = RouterDecision(intents=["recommend"], confidence=0.95)
ATE = RouterDecision(intents=["log_meal"], confidence=0.95)

ASK = "what should I eat tonight?"


def prompts(seam) -> list[str]:
    """Everything that reached the provider this Turn, as one string each."""
    return [call.sent for call in seam.provider.seen]


def ranking_prompt(seam) -> str:
    """What the ranker was shown. It is the last prose-tier call on a one-Intent
    Turn, and the only place a candidate list can be."""
    return next(c.sent for c in seam.provider.seen if c.asked_for is Recommendation)


async def a_meal(db, *, grams: float, values: dict, name="Pandesal", source="local",
                 fdc_id=None, local_food_id=None, when=None, nutrients=None):
    """One counted Item, written the way `store_meal` writes it."""
    from nutrigraph_agent.db import MealItemRow

    when = when or datetime.now(MANILA)
    meal = StoredMeal(uuid4(), DEMO_PROFILE.user_id, uuid4(), when, "lunch")
    db.meals.append(meal)
    db.items.append(
        StoredItem(
            uuid4(), meal.meal_id, DEMO_PROFILE.user_id, when,
            MealItemRow(
                ordinal=0, said_as=name.lower(), status="matched", grams=grams,
                source=source, food_name=name, local_food_id=local_food_id,
                fdc_id=fdc_id, values=values, nutrients=nutrients,
            ),
        )
    )
    return meal


# --- the gap is arithmetic over the Goal's targets, not a model's opinion ------


def test_the_gap_is_the_goal_targets_minus_todays_meals():
    targets = targets_for(DEMO_PROFILE)
    total = DayTotal(counted=1, values={"kcal": 618.0, "protein_g": 18.0})

    gap = gap_for(targets, total)

    assert gap.values["kcal"] == pytest.approx(targets.values["kcal"] - 618.0)
    assert gap.values["protein_g"] == pytest.approx(targets.values["protein_g"] - 18.0)
    # The one derivation of the targets is `review.targets_for`, reused. A second
    # one here would be two Coaches disagreeing about the same User.
    assert set(gap.values) <= set(targets.values)


def test_a_day_with_nothing_counted_is_short_by_the_whole_target():
    """The cold start as arithmetic: nothing logged is nothing eaten, which is a
    fact, and it is what lets a first suggestion come from the Goal alone."""
    targets = targets_for(DEMO_PROFILE)

    gap = gap_for(targets, DayTotal())

    assert gap.values["kcal"] == pytest.approx(targets.values["kcal"])
    assert gap.nutrient is not None
    assert gap.unmeasured == []


def test_a_nutrient_the_source_did_not_print_is_an_unknown_gap_not_a_whole_one():
    """A day holding a dish whose sodium PhilFCT does not print has an unknown
    sodium gap. Treating the null as a zero would make it the largest gap on the
    page and send the Coach after salt."""
    targets = targets_for(DEMO_PROFILE)
    total = DayTotal(counted=1, values={"kcal": 618.0}, missing={"fibre_g": 1})

    gap = gap_for(targets, total)

    assert "fibre_g" not in gap.values
    assert "fibre_g" in gap.unmeasured
    assert gap.nutrient == "kcal"


def test_sodium_is_never_the_nutrient_a_suggestion_is_built_on():
    """Its target is a ceiling. 'Closing the sodium gap' means recommending
    salt, so the sixth nutrient is computed, reported, and never ranked on."""
    gap = gap_for(targets_for(DEMO_PROFILE), DayTotal())

    assert "sodium_mg" in gap.values
    assert "sodium_mg" not in RANKED
    assert gap.nutrient != "sodium_mg"


# --- the filters run in the query, before any model call ----------------------


async def test_an_allergen_is_removed_before_the_model_sees_anything(seam):
    """Kare-kare's name says no peanut at all; the dish table's `peanut` tag is
    the only structured place that knows, and the filter reads it."""
    seam.provider.script(WANTS, SUGGESTED)

    await seam.turn(ASK)

    assert seam.db.candidate_queries == [
        {"blocked": ["peanut"], "conflicts": [], "nutrient": "kcal"}
    ]
    assert "Kare-kare" not in ranking_prompt(seam)
    assert "peanut" not in ranking_prompt(seam).lower()


async def test_a_disliked_food_is_removed_in_the_same_place(seam):
    seam.db.profiles[DEMO_PROFILE.user_id] = DEMO_PROFILE.model_copy(
        update={"disliked_foods": ["dinuguan"]}
    )
    seam.provider.script(WANTS, SUGGESTED)

    await seam.turn(ASK)

    assert "Dinuguan" not in ranking_prompt(seam)


@pytest.mark.parametrize(
    ("pattern", "gone"),
    [("vegan", "Champorado"), ("vegetarian", "Sisig"), ("pescatarian", "Tinola"),
     ("halal", "Dinuguan")],
)
async def test_a_diet_pattern_conflict_is_removed_by_the_tags(seam, pattern, gone):
    seam.db.profiles[DEMO_PROFILE.user_id] = DEMO_PROFILE.model_copy(
        update={"diet_pattern": pattern}
    )
    seam.provider.script(WANTS, SUGGESTED)

    await seam.turn(ASK)

    shown = ranking_prompt(seam)
    assert gone not in shown
    assert seam.db.candidate_queries[0]["conflicts"] == list(DIET_CONFLICTS[pattern])


async def test_the_filters_run_before_the_provider_is_asked_anything(seam):
    """The ordering the whole slice rests on: the candidate query has already
    answered by the time the ranker is called, so a prompt cannot be what keeps
    an allergen off the list."""
    seam.provider.script(WANTS, SUGGESTED)

    await seam.turn(ASK)

    assert len(seam.db.candidate_queries) == 1
    # The router is the only provider call that happened before the filter, and
    # it never saw a candidate.
    assert "Lechon manok" not in prompts(seam)[0]


# --- the model ranks, and it may not invent -----------------------------------


async def test_the_model_is_shown_the_survivors_and_nothing_else(seam):
    seam.provider.script(WANTS, SUGGESTED)

    await seam.turn(ASK)

    shown = ranking_prompt(seam)
    names = {c.name for c in await seam.db.candidate_foods(
        DEMO_PROFILE.user_id, blocked=["peanut"], conflicts=[], nutrient="kcal",
        gap=1842.0, limit=12,
    )}
    assert names
    assert all(name in shown for name in names)


async def test_a_suggestion_naming_a_food_that_was_not_a_candidate_fails(seam):
    """The test that makes 'the model never invents a food' a test rather than a
    hope. The invented name never reaches the User, and what does is built from
    a row."""
    seam.provider.script(
        WANTS,
        Recommendation(
            suggestion="[NAME_1], have a tuna poke bowl.",
            reason="It is high in protein.",
            foods=["tuna poke bowl"],
        ),
    )

    events = await seam.turn(ASK)

    text = answer(events).reply.text
    assert "poke" not in text.lower()
    # What the User reads instead came off the candidate rows.
    assert any(c.name in text for c in await seam.db.candidate_foods(
        DEMO_PROFILE.user_id, blocked=["peanut"], conflicts=[], nutrient="kcal",
        gap=1842.0, limit=12,
    ))
    assert seam.db.recommendations[-1].foods


def test_the_check_is_exact_and_it_raises():
    from nutrigraph_agent.db import Candidate

    candidates = [Candidate(source="local", source_id="1", name="Lechon manok")]

    assert check_foods(["lechon MANOK"], candidates) == ["Lechon manok"]
    with pytest.raises(InventedFood):
        check_foods(["Lechon manok", "tuna poke bowl"], candidates)


async def test_every_suggestion_carries_a_reason(seam):
    seam.provider.script(WANTS, SUGGESTED)

    events = await seam.turn(ASK)

    assert SUGGESTED.reason in answer(events).reply.text
    assert seam.db.recommendations[-1].reason == SUGGESTED.reason


async def test_a_suggestion_is_written_down_with_the_foods_it_named(seam):
    turn_id = uuid4()
    seam.provider.script(WANTS, SUGGESTED)

    await seam.turn(ASK, turn_id=turn_id)

    written = seam.db.recommendations[-1]
    assert written.foods == ["Lechon manok"]
    assert written.turn_id == turn_id
    assert written.gap_nutrient == "kcal"
    # Null until the User says. That is the acceptance signal, unanswered.
    assert written.accepted is None


# --- the routing rule ---------------------------------------------------------


async def test_the_path_writes_prose_so_it_uses_flash(seam):
    seam.provider.script(WANTS, SUGGESTED)

    await seam.turn(ASK)

    # The router classifies, on the schema tier. The ranker writes what the User
    # reads, so by the routing rule it is the prose tier.
    assert [c.model for c in seam.provider.seen] == [SCHEMA_MODEL, PROSE_MODEL]


async def test_the_ranker_climbs_the_same_ladder_as_every_other_call(seam):
    from .fakes import ResourceExhausted

    seam.provider.script(WANTS, SUGGESTED)
    seam.provider.fail_on(Recommendation, *[ResourceExhausted("429")] * 3)

    events = await seam.turn(ASK)

    assert seam.provider.attempts_on(Recommendation) == [
        PROSE_MODEL, PROSE_MODEL, PROSE_MODEL, SCHEMA_MODEL
    ]
    assert "Lechon manok" in answer(events).reply.text


# --- the similarity, and the cold start ---------------------------------------


async def test_a_user_with_no_meal_and_no_accepted_suggestion_still_gets_one(seam):
    """From the Goal, the diet pattern and the dish table alone. No onboarding
    conversation, and no similarity to lean on."""
    seam.provider.script(WANTS, SUGGESTED)

    events = await seam.turn(ASK)

    reply = answer(events).reply
    assert "Lechon manok" in reply.text
    assert COLD_START in reply.text
    assert all(c.similarity is None for c in await seam.db.candidate_foods(
        DEMO_PROFILE.user_id, blocked=[], conflicts=[], nutrient="kcal",
        gap=1842.0, limit=12,
    ))


async def test_the_ordering_is_influenced_by_what_this_user_has_eaten(seam):
    """The personalisation is a similarity query, not a phrase in a prompt: two
    Users with the same gap and the same rows see a different order because one
    of them has eaten something."""
    table = {f.local_food_id: f for f in seam.db.local_foods.values()}
    dishes = sorted(table.values(), key=lambda f: f.name)
    liked, other = dishes[0], dishes[-1]
    # A vector for each dish, and a Meal holding one of them. The stored vectors
    # are what the fake takes a centroid of, exactly as `avg(embedding)` does.
    for position, dish in enumerate(dishes):
        seam.db.food_embeddings[("local", str(dish.local_food_id))] = (
            dish.name, [1.0 if n == position else 0.0 for n in range(len(dishes))]
        )
    await a_meal(
        seam.db, grams=100.0, values={"kcal": 10.0}, name=liked.name,
        local_food_id=liked.local_food_id,
    )

    ordered = await seam.db.candidate_foods(
        DEMO_PROFILE.user_id, blocked=[], conflicts=[], nutrient=None, gap=0.0, limit=12
    )

    similarity = {c.name: c.similarity for c in ordered}
    assert similarity[liked.name] == pytest.approx(1.0)
    assert similarity[other.name] == pytest.approx(0.0)
    assert ordered[0].name == liked.name


async def test_an_accepted_suggestion_feeds_the_similarity_the_same_way(seam):
    """The second half of 'ate or accepted'. A suggestion the User said yes to
    is a food they chose, and it counts even though no Meal followed."""
    dishes = sorted(
        {f.local_food_id: f for f in seam.db.local_foods.values()}.values(),
        key=lambda f: f.name,
    )
    for position, dish in enumerate(dishes):
        seam.db.food_embeddings[("local", str(dish.local_food_id))] = (
            dish.name, [1.0 if n == position else 0.0 for n in range(len(dishes))]
        )
    accepted = dishes[3]
    recommendation_id = await seam.db.store_recommendation(
        user_id=DEMO_PROFILE.user_id, turn_id=uuid4(), gap_nutrient="kcal",
        gap_amount=100.0, suggestion="…", reason="…", foods=[accepted.name],
    )
    await seam.db.answer_recommendation(recommendation_id, accepted=True)

    ordered = await seam.db.candidate_foods(
        DEMO_PROFILE.user_id, blocked=[], conflicts=[], nutrient=None, gap=0.0, limit=12
    )

    assert ordered[0].name == accepted.name


# --- the candidates come from two sources -------------------------------------


async def test_a_food_this_user_logged_is_a_candidate_next_time(seam):
    """The second source. Cheddar is not in the Filipino dish table, so it can
    only be on the list because this User logged it once."""
    await a_meal(
        seam.db, grams=100.0, values={"kcal": 403.0, "protein_g": 23.0},
        name="Cheese, cheddar", source="fdc", fdc_id="328637",
    )
    seam.provider.script(WANTS, SUGGESTED)

    await seam.turn(ASK)

    assert "Cheese, cheddar" in ranking_prompt(seam)


async def test_a_logged_food_is_offered_on_the_per_100_g_basis_it_was_measured_on(seam):
    """A logged Item holds the values for the portion that was eaten. Offering
    those as the food's own numbers would say a 200 g serving is what 100 g
    carries."""
    await a_meal(
        seam.db, grams=200.0, values={"kcal": 286.0}, name="Egg, whole, raw, fresh",
        source="fdc", fdc_id="748967",
    )

    found = await seam.db.candidate_foods(
        DEMO_PROFILE.user_id, blocked=[], conflicts=[], nutrient="kcal",
        gap=1000.0, limit=30,
    )

    egg = next(c for c in found if c.name == "Egg, whole, raw, fresh")
    assert egg.per_100g["kcal"] == pytest.approx(143.0)


# --- the dish table is verified and off limits --------------------------------


async def test_a_proxy_row_is_marked_and_a_calculated_row_is_never_measured(seam):
    """Nothing here adjusts a transcribed value; it says what the value is."""
    seam.provider.script(
        WANTS,
        Recommendation(
            suggestion="[NAME_1], try Sisig.",
            reason="It closes the protein gap.",
            foods=["Sisig"],
        ),
    )

    events = await seam.turn(ASK)

    reply = answer(events).reply
    assert any("calculated from component foods" in d for d in reply.disclaimers)
    assert "calculated from component foods" in reply.text


async def test_the_targets_are_marked_as_worked_out_rather_than_measured(seam):
    seam.provider.script(WANTS, SUGGESTED)

    events = await seam.turn(ASK)

    assert any("worked out rather than measured" in d
               for d in answer(events).reply.disclaimers)


async def test_a_profile_that_cannot_produce_targets_still_suggests_something(seam):
    """No Goal targets is not no answer. There is no gap to rank on, so the
    ordering rests on what this User eats, and the reply says which of the two
    it is rather than presenting a guess as a target."""
    seam.db.profiles["thin"] = Profile(user_id="thin", name="Ana")
    seam.provider.script(WANTS, SUGGESTED)

    events = await seam.turn(ASK, user_id="thin")

    reply = answer(events).reply
    assert "I cannot work out your targets" in reply.text
    dishes = {f.name for f in seam.db.local_foods.values()}
    assert seam.db.recommendations[-1].foods
    assert set(seam.db.recommendations[-1].foods) <= dishes
    assert seam.db.recommendations[-1].gap_nutrient is None


# --- the allergy check is a second line of defence, not the filter -------------


async def test_the_allergy_check_never_fires_on_a_correct_path(seam):
    """The SQL filter removed the allergens before the model saw a list, so
    there is nothing left for the prose scan to strike. This failing means the
    filter stopped being the thing that protects the User."""
    seam.provider.script(WANTS, SUGGESTED)

    events = await seam.turn(ASK)

    reply = answer(events).reply
    assert "Lechon manok" in reply.text
    assert "could not finish" not in reply.text


async def test_a_food_the_check_strikes_is_said_again_from_the_rows(seam):
    """The one regeneration goes back to the candidate rows, never to a model:
    asking for another draft is how a Turn learns to loop."""
    seam.provider.script(
        WANTS,
        Recommendation(
            suggestion="[NAME_1], try Lechon manok with a peanut sauce.",
            reason="It closes the protein gap.",
            foods=["Lechon manok"],
        ),
    )

    events = await seam.turn(ASK)

    reply = answer(events).reply
    assert "peanut" not in reply.text
    # One ranker call, and no second one: the answer was rebuilt from the rows.
    assert len(seam.provider.attempts_on(Recommendation)) == 1


# --- the measurement ----------------------------------------------------------


async def test_accepting_a_suggestion_writes_the_accepted_column(seam):
    seam.provider.script(WANTS, SUGGESTED)
    await seam.turn(ASK)
    written = seam.db.recommendations[-1]

    assert await seam.db.answer_recommendation(written.recommendation_id, accepted=True)

    assert written.accepted is True
    assert written.responded_at is not None
    # Answered once. What the User said the first time is the measurement.
    assert not await seam.db.answer_recommendation(
        written.recommendation_id, accepted=False
    )
    assert written.accepted is True


async def test_rejecting_one_writes_it_too(seam):
    seam.provider.script(WANTS, SUGGESTED)
    await seam.turn(ASK)
    written = seam.db.recommendations[-1]

    await seam.db.answer_recommendation(written.recommendation_id, accepted=False)

    assert written.accepted is False


async def test_a_meal_holding_the_recommended_food_within_a_day_is_the_second_signal(
    seam,
):
    """Acceptance alone cannot tell a polite yes from a real change, which is
    why this exists. It is a query over `recommendation.foods` and `meal_item`,
    and it needed no new column."""
    seam.provider.script(WANTS, SUGGESTED)
    await seam.turn(ASK)
    written = seam.db.recommendations[-1]
    assert [o.followed for o in
            await seam.db.recommendation_outcomes(DEMO_PROFILE.user_id)] == [False]

    await a_meal(
        seam.db, grams=200.0, values={"kcal": 452.0}, name="Lechon manok",
        when=written.created_at + timedelta(hours=6),
    )

    outcomes = await seam.db.recommendation_outcomes(DEMO_PROFILE.user_id)
    assert [o.followed for o in outcomes] == [True]
    assert outcomes[0].accepted is None  # followed without ever saying yes


async def test_a_meal_a_day_later_is_not_following_the_suggestion(seam):
    seam.provider.script(WANTS, SUGGESTED)
    await seam.turn(ASK)
    written = seam.db.recommendations[-1]

    await a_meal(
        seam.db, grams=200.0, values={"kcal": 452.0}, name="Lechon manok",
        when=written.created_at + timedelta(hours=25),
    )

    assert [o.followed for o in
            await seam.db.recommendation_outcomes(DEMO_PROFILE.user_id)] == [False]


# --- the vector table fills itself --------------------------------------------


async def test_a_newly_matched_fdc_food_is_embedded_the_first_time_it_is_seen(seam):
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        ATE,
        ParsedMeal(items=[ParsedItem(name="egg", quantity=2, unit="piece")]),
        FoodChoice(fdc_id="748967", reason="the plain whole egg"),
    )

    await seam.turn("I ate two eggs")

    assert seam.db.food_embeddings[("fdc", "748967")][0] == "Egg, whole, raw, fresh"
    assert "Egg, whole, raw, fresh" in seam.provider.embedded


async def test_the_second_time_it_is_seen_it_is_not_embedded_again(seam):
    seam.db.food_embeddings[("fdc", "748967")] = ("Egg, whole, raw, fresh", [0.0] * 768)
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        ATE,
        ParsedMeal(items=[ParsedItem(name="egg", quantity=2, unit="piece")]),
        FoodChoice(fdc_id="748967", reason="the plain whole egg"),
    )

    await seam.turn("I ate two eggs")

    assert seam.provider.embedded == []


async def test_an_embedding_that_cannot_be_written_does_not_lose_the_meal(seam):
    """The vector is a nice-to-have; the Meal is the record. A provider that
    stopped at every rung leaves the food unembedded and the Meal written."""
    from nutrigraph_agent.providers import MAX_ATTEMPTS

    from .fakes import EMBED, ResourceExhausted

    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        ATE,
        ParsedMeal(items=[ParsedItem(name="egg", quantity=2, unit="piece")]),
        FoodChoice(fdc_id="748967", reason="the plain whole egg"),
    )
    seam.provider.fail_on(EMBED, *[ResourceExhausted("429")] * MAX_ATTEMPTS)

    events = await seam.turn("I ate two eggs")

    assert len(seam.db.meals) == 1
    assert ("fdc", "748967") not in seam.db.food_embeddings
    assert "could not match" not in answer(events).reply.text


# --- the whole path, from the seam --------------------------------------------


async def test_the_second_intent_suggests_against_a_total_that_holds_the_first(seam):
    """The two-Intent Turn, on this path: a Meal logged in the same breath is in
    the day total the gap is computed from."""
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        RouterDecision(intents=["log_meal", "recommend"], confidence=0.95),
        ParsedMeal(items=[ParsedItem(name="egg", quantity=2, unit="piece")]),
        FoodChoice(fdc_id="748967", reason="the plain whole egg"),
        SUGGESTED,
        ComposedReply(text="Lou, logged the eggs, and try Lechon manok next."),
    )

    events = await seam.turn("I ate two eggs, what should I have tonight?")

    reply = answer(events).reply
    assert [p.intent for p in reply.parts] == ["log_meal", "recommend"]
    # The egg is a candidate on the same Turn that logged it.
    assert "Egg, whole, raw, fresh" in ranking_prompt(seam)


async def test_a_new_allergy_stated_this_turn_is_honoured_by_the_suggestion(seam):
    seam.provider.script(
        RouterDecision(intents=["update_profile", "recommend"], confidence=0.95),
        ProfileUpdate(field="allergies", old_value="peanut", new_value="pork"),
        SUGGESTED,
        ComposedReply(text="Lou, noted, and try Lechon manok next."),
    )

    await seam.turn("I am allergic to pork, what should I eat tonight?")

    assert seam.db.candidate_queries[0]["blocked"] == ["peanut", "pork"]
    assert "Dinuguan" not in ranking_prompt(seam)
