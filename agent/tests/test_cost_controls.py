"""The three rules that keep a free-tier system usable and its cost measurable:
what may be cached, what happens when a call fails, and what fits in a Turn.

The cache tests are the ones that matter most. The cache holds lookups, never
answers, and nothing that read the Profile or today's Meals is ever written to
it — that rule is what makes a stale nutrition answer impossible, and
`test_nothing_that_read_the_profile_or_todays_meals_is_ever_cached` is the test
that proves it, by driving every shape of Turn there is and then reading every
row the cache holds.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

import pytest

from nutrigraph_agent.budget import (
    BUDGET_TOKENS,
    HISTORY_TURNS,
    MAX_CHUNKS,
    estimate_tokens,
    fit,
    last_turns,
)
from nutrigraph_agent.db import FOOD_MATCH_DAYS, RETRIEVAL_SIMILARITY
from nutrigraph_agent.meal import normalize
from nutrigraph_agent.models import (
    Answer,
    Citation,
    FoodChoice,
    ParsedItem,
    ParsedMeal,
    ProfileUpdate,
    RouterDecision,
)
from nutrigraph_agent.providers import (
    BACKOFF_SECONDS,
    MAX_ATTEMPTS,
    is_transient,
    ladder,
)
from nutrigraph_agent.turn import FALLBACK_ERROR

from .conftest import PROSE_MODEL, SCHEMA_MODEL, answer
from .fakes import EGG, EGGS_CHUNK, BadApiKey, ResourceExhausted, now_utc

ASKED = RouterDecision(intents=["ask_question"], confidence=0.95)
ATE = RouterDecision(intents=["log_meal"], confidence=0.95)
SURE = RouterDecision(intents=["review_day"], confidence=0.95)
UNSURE = RouterDecision(intents=[], confidence=0.2)

CITED = Answer(
    text="Eggs are one of the protein foods the guidelines name.",
    citations=[Citation(document=EGGS_CHUNK.document, locator=EGGS_CHUNK.locator)],
)

ATE_EGGS = ParsedMeal(items=[ParsedItem(name="egg", quantity=2, unit="piece")])
CHOSE_EGG = FoodChoice(fdc_id="748967", reason="the plain whole egg")

QUESTION = "is an egg a good source of protein?"


def cache_dump(seam) -> str:
    """Every row of the lookup cache, as one string a test can search."""
    return json.dumps(
        [
            {"kind": e.kind, "key_text": e.key_text, "value": e.value}
            for e in seam.db.cache
        ],
        default=str,
    )


# --- the retrieval half of the cache ------------------------------------------


async def test_a_repeated_question_is_served_from_the_cache_and_no_retrieval_runs(seam):
    seam.provider.script(ASKED, CITED, ASKED, CITED)

    first = await seam.turn(QUESTION)
    second = await seam.turn(QUESTION)

    # One search, for the first question. The second was served from the entry
    # the first one wrote, and the Corpus was not touched.
    assert len(seam.db.searched) == 1
    assert [e.hits for e in seam.db.cache if e.kind == "retrieval"] == [1]
    assert answer(first).reply.text == answer(second).reply.text


async def test_a_question_below_the_similarity_floor_misses_the_cache(seam):
    seam.provider.script(ASKED, CITED, ASKED, CITED)

    await seam.turn(QUESTION)
    await seam.turn("how much sodium should I have in a day?")

    # A different question embeds to a different vector, so the entry is not
    # near enough to serve it and the Corpus is searched again.
    assert len(seam.db.searched) == 2
    assert len([e for e in seam.db.cache if e.kind == "retrieval"]) == 2


@pytest.mark.parametrize(
    ("similarity", "served"),
    [(1.0, True), (RETRIEVAL_SIMILARITY, True), (RETRIEVAL_SIMILARITY - 0.01, False)],
)
async def test_the_floor_is_a_cosine_of_0_95_and_it_is_the_boundary(
    seam, similarity, served
):
    """Two hand-built unit vectors whose dot product — and so whose cosine — is
    exactly the number under test. 0.95 hits; a hair below it does not."""
    stored = [1.0] + [0.0] * 767
    asked = [similarity, (1 - similarity**2) ** 0.5] + [0.0] * 766
    await seam.db.store_cached_retrieval(
        key_text="a question", embedding=stored, chunks=[EGGS_CHUNK]
    )

    found = await seam.db.cached_retrieval(asked)

    assert (found is not None) == served


async def test_re_ingesting_the_corpus_invalidates_every_retrieval_entry(seam):
    seam.provider.script(ASKED, CITED, ASKED, CITED)
    await seam.turn(QUESTION)
    assert len(seam.db.searched) == 1

    seam.db.reingest()
    await seam.turn(QUESTION)

    # The same question, and the Corpus is searched again: the entry was
    # written against a Corpus that no longer exists.
    assert len(seam.db.searched) == 2


# --- the food match half of the cache -----------------------------------------


async def test_a_repeated_food_name_skips_the_search_and_the_choice(seam):
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(ATE, ATE_EGGS, CHOSE_EGG, ATE, ATE_EGGS)

    first = await seam.turn("I ate two eggs")
    second = await seam.turn("I ate two eggs")

    # One FoodData Central search, and one choice call. The second Turn made
    # neither: the router and the parse are all it cost.
    assert seam.food.searched == ["egg"]
    assert [c.model for c in seam.provider.seen] == [SCHEMA_MODEL] * 5
    assert [e.key_text for e in seam.db.cache if e.kind == "food_match"] == ["egg"]
    # And the food was still counted, from the cached match.
    for events in (first, second):
        assert "could not match" not in answer(events).reply.text


async def test_a_food_match_entry_expires_after_thirty_days(seam):
    await seam.db.store_cached_food_match("pandesal", {"source": "fdc"})
    assert await seam.db.cached_food_match("pandesal") == {"source": "fdc"}

    entry = next(e for e in seam.db.cache if e.kind == "food_match")
    entry.created_at = now_utc() - timedelta(days=FOOD_MATCH_DAYS, seconds=1)

    assert await seam.db.cached_food_match("pandesal") is None


async def test_the_key_is_the_exact_lowercased_name(seam):
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        ATE,
        ParsedMeal(items=[ParsedItem(name="Egg", quantity=1, unit="piece")]),
        CHOSE_EGG,
    )

    await seam.turn("I ate an Egg")

    assert [e.key_text for e in seam.db.cache if e.kind == "food_match"] == ["egg"]
    assert normalize("Egg") == "egg"


async def test_a_food_nothing_matched_is_not_cached_as_a_no(seam):
    """A cached 'no' would keep the Coach from ever counting a food the
    catalogue gains later, and the catalogue is the thing that moves."""
    seam.provider.script(
        ATE, ParsedMeal(items=[ParsedItem(name="zzzfoodzzz", quantity=1)])
    )

    await seam.turn("I ate zzzfoodzzz")

    assert [e for e in seam.db.cache if e.kind == "food_match"] == []


# --- what may never be cached -------------------------------------------------


async def test_nothing_that_read_the_profile_or_todays_meals_is_ever_cached(seam):
    """The rule that makes a stale nutrition answer impossible.

    Every shape of Turn runs: one that changes the Profile, one that logs a
    Meal the Coach could not match, one that logs a Meal after it — which reads
    today's Meals to offer the correction — one that answers from the Corpus,
    and one the guardrail refuses. Then every row the cache holds is read.
    """
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        RouterDecision(intents=["update_profile"], confidence=0.95),
        ProfileUpdate(field="weight_kg", new_value="80", old_value="78"),
        ATE,
        ParsedMeal(items=[ParsedItem(name="zzzfoodzzz", quantity=1)]),
        ATE,
        ATE_EGGS,
        CHOSE_EGG,
        ASKED,
        CITED,
    )

    replies = [
        answer(await seam.turn("I weigh 80 kg now")).reply.text,
        answer(await seam.turn("I ate zzzfoodzzz")).reply.text,
        answer(await seam.turn("I ate two eggs")).reply.text,
        answer(await seam.turn(f"I'm Lou — {QUESTION}")).reply.text,
    ]
    await seam.turn("do I have diabetes?")

    assert seam.db.cache, "the Turns wrote nothing at all, so this proves nothing"
    # Two kinds, and no third. Neither of them is an answer.
    assert {e.kind for e in seam.db.cache} <= {"retrieval", "food_match"}

    held = cache_dump(seam)
    # Nothing the Profile holds, and nothing that identifies the User. The
    # question was cached under the redacted text, so the name is not there
    # although the User typed it.
    for private in ("Lou", "peanut", "demo-user-1", "80", "weight"):
        assert private not in held, f"{private!r} is in the lookup cache"
    # Today's Meals: the food the Coach could not match was read back on the
    # next Turn to offer a correction, and no entry was written for it.
    assert "zzzfoodzzz" not in held
    # And no whole answer, of any shape.
    for reply in replies:
        assert reply not in held


async def test_a_refused_turn_and_a_clarified_turn_cache_nothing(seam):
    seam.provider.script(UNSURE)

    await seam.turn("hmm")
    await seam.turn("what should I take for my blood pressure?")

    assert seam.db.cache == []


# --- the retry ladder ---------------------------------------------------------


async def test_a_transient_failure_retries_twice_about_one_second_and_then_three(seam):
    seam.provider.fail(ResourceExhausted("429"), ResourceExhausted("429")).script(SURE)

    events = await seam.turn("how did my day go?")

    assert seam.provider.slept == [1.0, 3.0] == list(BACKOFF_SECONDS)
    assert [c.model for c in seam.provider.seen] == [SCHEMA_MODEL] * 3
    # The Turn answered. A rate-limit stop is normal traffic on a free tier.
    assert answer(events).reply.parts[0].intent == "review_day"


async def test_two_failures_on_flash_fall_back_to_flash_lite_for_the_same_call(seam):
    # The router answers, then the prose tier stops three times over.
    stops = [ResourceExhausted("429")] * 3
    seam.provider.fail(None, *stops).script(UNSURE)

    events = await seam.turn("hmm")

    models = [c.model for c in seam.provider.seen]
    assert models == [SCHEMA_MODEL, PROSE_MODEL, PROSE_MODEL, PROSE_MODEL, SCHEMA_MODEL]
    # The same call, not a different one: one prompt across all four rungs.
    assert len({c.sent for c in seam.provider.seen[1:]}) == 1
    assert answer(events).reply.text.endswith("?")


async def test_a_failure_at_every_step_ends_the_turn_with_the_fallback_and_an_error(seam):
    seam.provider.fail(None, *[ResourceExhausted("429")] * MAX_ATTEMPTS).script(UNSURE)

    events = await seam.turn("hmm")

    assert events[-1].code == "provider_unavailable"
    assert events[-1].message == FALLBACK_ERROR
    assert not any(type(e).__name__ == "AnswerEvent" for e in events)


async def test_no_turn_exceeds_four_attempts(seam):
    seam.provider.fail(None, *[ResourceExhausted("429")] * 9).script(UNSURE)

    await seam.turn("hmm")

    # One router call, then the ladder, and the ladder stops at four.
    assert len(seam.provider.seen) - 1 == MAX_ATTEMPTS == 4
    assert len(seam.provider.slept) == len(BACKOFF_SECONDS)


async def test_a_stop_that_is_not_transient_is_raised_at_once(seam):
    """Three more attempts at a bad key change nothing, and cost three seconds
    of a User's time to find that out."""
    seam.provider.fail(BadApiKey("401"))

    events = await seam.turn("hello")

    assert len(seam.provider.seen) == 1
    assert seam.provider.slept == []
    assert events[-1].code == "turn_failed"
    assert events[-1].message == FALLBACK_ERROR


def test_the_ladder_has_four_rungs_and_the_last_one_is_the_weaker_model():
    plan = ladder("gemini-3.5-flash", "gemini-3.5-flash-lite")

    assert plan == [
        ("gemini-3.5-flash", 0.0),
        ("gemini-3.5-flash", 1.0),
        ("gemini-3.5-flash", 3.0),
        ("gemini-3.5-flash-lite", 0.0),
    ]
    assert len(plan) == MAX_ATTEMPTS


def test_there_is_no_rung_below_the_weaker_model():
    """The schema tier is already Flash-Lite, so a fourth attempt would be the
    same call to the same model, and only another second of waiting."""
    plan = ladder("gemini-3.5-flash-lite", "gemini-3.5-flash-lite")

    assert [model for model, _ in plan] == ["gemini-3.5-flash-lite"] * 3


@pytest.mark.parametrize(
    "exc",
    [
        ResourceExhausted("rate limited"),
        TimeoutError("deadline"),
        ConnectionError("reset"),
        type("ServiceUnavailable", (Exception,), {})(),
        type("Whatever", (Exception,), {"status_code": 503})(),
    ],
)
def test_what_the_ladder_treats_as_transient(exc):
    assert is_transient(exc)


@pytest.mark.parametrize(
    "exc", [BadApiKey("401"), ValueError("bad schema"), type("Nope", (Exception,), {"status_code": 400})()]
)
def test_what_the_ladder_does_not_retry(exc):
    assert not is_transient(exc)


# --- the token budget ---------------------------------------------------------


def a_thread(turns: int, chars: int = 9_000) -> list[dict[str, str]]:
    """A Thread of `turns` turns, each one long enough to matter."""
    messages: list[dict[str, str]] = []
    for i in range(1, turns + 1):
        messages.append({"role": "user", "text": f"marker-{i} " + "eaten rice " * (chars // 11)})
        messages.append({"role": "coach", "text": f"noted marker-{i}"})
    return messages


def test_the_history_is_cut_to_the_last_six_turns_and_nothing_is_summarised():
    history = a_thread(10)

    kept = last_turns(history)

    # Six turns, both halves of each, verbatim. No summary, and no call that
    # could have written one.
    assert len(kept) == 2 * HISTORY_TURNS
    assert kept[0]["text"].startswith("marker-5 ")
    assert kept == history[-2 * HISTORY_TURNS :]


def test_a_shorter_thread_is_not_padded_or_cut():
    history = a_thread(3)

    assert last_turns(history) == history


def test_the_profile_and_todays_meals_survive_trimming_in_every_case():
    """`keep` is measured and never cut. The proof is that the Turn is still
    counted as costing every token of the Profile and today's Meals after the
    trimming has run, and is reported as over budget rather than made to fit by
    dropping one of them."""
    profile = "the Profile: " + "allergic to shrimp, " * 400
    meals = "today's Meals: " + "one cup of rice at 07:00, " * 400

    fitted = fit(keep=[profile, meals], history=a_thread(10), passages=[EGGS_CHUNK] * 20)

    assert fitted.tokens >= estimate_tokens(profile, meals)
    assert fitted.over_budget is True
    assert len(fitted.history) == 2 * HISTORY_TURNS
    assert len(fitted.passages) == MAX_CHUNKS
    assert fitted.trimmed == (
        f"history to the last {HISTORY_TURNS} turns",
        f"passages to {MAX_CHUNKS}",
    )


def test_retrieved_chunks_are_capped_at_five():
    long_chunk = replace(EGGS_CHUNK, text="protein " * 2_000)

    fitted = fit(keep=["a question"], passages=[long_chunk] * 8)

    assert len(fitted.passages) == MAX_CHUNKS


def test_a_turn_that_fits_is_not_trimmed_at_all():
    fitted = fit(keep=["a question"], history=a_thread(10, chars=40), passages=[EGGS_CHUNK])

    assert fitted.trimmed == ()
    assert fitted.over_budget is False
    assert len(fitted.history) == 20
    assert fitted.tokens < BUDGET_TOKENS


async def test_a_turn_over_budget_trims_the_history_and_records_the_overrun(seam):
    for i in range(1, 9):
        await seam.turn(f"marker-{i} " + "eaten rice " * 800)
    before = len(seam.provider.seen)

    await seam.turn("marker-9 and what about that?")

    router = seam.provider.seen[-1].sent
    # One call for the Turn: the router. Nothing was summarised, because a
    # summary would be a second call and there is not one.
    assert len(seam.provider.seen) - before == 1
    # The last six turns, in full. The two before them are gone.
    for kept in range(3, 9):
        assert f"marker-{kept}" in router
    assert "marker-2" not in router
    assert "marker-1" not in router
    # Still over after the trimming, so the Turn ran anyway and said so.
    row = next(r for r in seam.db.events if r.node == "route" and r.turn_id == seam.db.events[-1].turn_id)
    assert row.over_budget is True


async def test_a_turn_inside_the_budget_records_no_overrun(seam):
    await seam.turn("I ate two eggs")

    assert [r.over_budget for r in seam.db.events] == [False] * len(seam.db.events)


async def test_input_tokens_for_a_turn_are_readable_without_the_tracing_tool(seam):
    seam.provider.script(SURE)

    await seam.turn("how did my day go?")

    turn_id = seam.db.events[0].turn_id
    rows = [r for r in seam.db.events if r.turn_id == turn_id]
    # One number, out of one table, with no LangSmith in the loop.
    assert sum(r.input_tokens for r in rows) == 11
    assert sum(r.cost_usd for r in rows) > 0
