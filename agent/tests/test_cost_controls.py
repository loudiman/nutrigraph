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
import re
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest

from nutrigraph_agent.budget import (
    BUDGET_TOKENS,
    HISTORY_TURNS,
    MAX_CHUNKS,
    estimate_tokens,
    fit,
    last_turns,
    render_history,
)
from nutrigraph_agent.db import FOOD_MATCH_DAYS, RETRIEVAL_SIMILARITY
from nutrigraph_agent.meal import normalize
from nutrigraph_agent.models import (
    Answer,
    Citation,
    ComposedReply,
    DayRequest,
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

from .conftest import (
    PROSE_MODEL,
    SCHEMA_MODEL,
    Interloper,
    also_calls_the_provider,
    answer,
)
from .fakes import (
    EGG,
    EGGS_CHUNK,
    SUGGESTED,
    WRITE,
    BadApiKey,
    ResourceExhausted,
    now_utc,
)

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

# A Turn with two Intents, so `compose_reply` makes its prose-tier call. Every
# ladder assertion here reads `attempts_on(ComposedReply)`, so what the two
# paths call on the way is not this test's business — that is issue #54's rule.
TWO_INTENTS = RouterDecision(intents=["review_day", "recommend"], confidence=0.95)
COMPOSED = ComposedReply(text="[NAME_1], here is both of those.")

# The sentence issue #36 names: two Intents, and the second answers from a day
# total the first has already written to.
BOTH = RouterDecision(intents=["log_meal", "ask_question"], confidence=0.95)


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
#
# Every test below reads `attempts_on(<the call>)`, never the whole of
# `seen`. The ladder is a property of one call, and what else a Turn happens to
# ask the provider is not these tests' business. Asserting the Turn's whole
# sequence is what issue #54 was: #34 gave `review_day` a `DayRequest` call of
# its own, three ladder tests went red, and the ladder was untouched. The
# `ladder_seam` fixture runs each of them a second time with a provider call
# added elsewhere in the graph, so that cannot come back.


@pytest.fixture(params=["as the graph is", "with a new call elsewhere in the graph"])
def ladder_seam(request, seam, monkeypatch):
    """The regression guard for issue #54, as a fixture.

    The second parameter makes the Turn call the provider once more than the
    graph does. A ladder test that reads the whole of `seen` fails under it; one
    that reads only its own call's attempts does not notice. The double itself
    is in `conftest`, because `test_review_day` guards its script with it too.
    """
    # Which half of the guard this run is, for the one test that asserts both.
    seam.noisy = request.param != "as the graph is"
    return seam if not seam.noisy else also_calls_the_provider(seam, monkeypatch)


async def test_a_call_added_elsewhere_does_not_move_one_call_s_attempts(ladder_seam):
    """The guard's own guard, and the defect issue #54 named, in one test.

    Under the second parameter the Turn makes a provider call the graph does not
    make. That is what `#34` did to these tests when they read the whole of
    `seen`. Read one call's attempts instead and the two runs are identical.
    """
    ladder_seam.provider.fail_on(RouterDecision, ResourceExhausted("429"))
    ladder_seam.provider.script(SURE, DayRequest(days_ago=0))

    await ladder_seam.turn("how did my day go?")

    # The double really does reach the provider — a fixture that quietly
    # stopped would make every ladder test below pass for the wrong reason.
    assert ladder_seam.provider.attempts_on(Interloper) == (
        [SCHEMA_MODEL] if ladder_seam.noisy else []
    )
    # And the router's own rungs read the same either way.
    assert ladder_seam.provider.attempts_on(RouterDecision) == [SCHEMA_MODEL] * 2


async def test_a_transient_failure_retries_twice_about_one_second_and_then_three(ladder_seam):
    ladder_seam.provider.fail_on(
        RouterDecision, ResourceExhausted("429"), ResourceExhausted("429")
    ).script(SURE, DayRequest(days_ago=0))

    events = await ladder_seam.turn("how did my day go?")

    assert ladder_seam.provider.slept == [1.0, 3.0] == list(BACKOFF_SECONDS)
    # Three rungs for the router: the call and its two retries, all on the tier
    # it started on, because the schema tier is already the weaker one.
    assert ladder_seam.provider.attempts_on(RouterDecision) == [SCHEMA_MODEL] * 3
    # The Turn answered. A rate-limit stop is normal traffic on a free tier.
    assert answer(events).reply.parts[0].intent == "review_day"


async def test_two_failures_on_flash_fall_back_to_flash_lite_for_the_same_call(ladder_seam):
    """The prose call `clarify` makes, which asks for no schema and so starts
    on Flash — the tier that has a rung below it."""
    ladder_seam.provider.fail_on(WRITE, *[ResourceExhausted("429")] * 3).script(UNSURE)

    events = await ladder_seam.turn("hmm")

    assert ladder_seam.provider.attempts_on(WRITE) == [
        PROSE_MODEL, PROSE_MODEL, PROSE_MODEL, SCHEMA_MODEL
    ]
    # The same call, not a different one: one prompt across all four rungs.
    written = [c for c in ladder_seam.provider.seen if c.asked_for == WRITE]
    assert len({c.sent for c in written}) == 1
    assert answer(events).reply.text.endswith("?")


async def test_a_failure_at_every_step_ends_the_turn_with_the_fallback_and_an_error(
    ladder_seam,
):
    ladder_seam.provider.fail_on(
        WRITE, *[ResourceExhausted("429")] * MAX_ATTEMPTS
    ).script(UNSURE)

    events = await ladder_seam.turn("hmm")

    assert events[-1].code == "provider_unavailable"
    assert events[-1].message == FALLBACK_ERROR
    assert not any(type(e).__name__ == "AnswerEvent" for e in events)


async def test_no_turn_exceeds_four_attempts(ladder_seam):
    ladder_seam.provider.fail_on(WRITE, *[ResourceExhausted("429")] * 9).script(UNSURE)

    await ladder_seam.turn("hmm")

    # Nine stops were queued and the ladder took four of them, because there is
    # no fifth attempt to make.
    assert len(ladder_seam.provider.attempts_on(WRITE)) == MAX_ATTEMPTS == 4
    assert len(ladder_seam.provider.slept) == len(BACKOFF_SECONDS)


async def test_a_stop_that_is_not_transient_is_raised_at_once(ladder_seam):
    """Three more attempts at a bad key change nothing, and cost three seconds
    of a User's time to find that out."""
    ladder_seam.provider.fail_on(RouterDecision, BadApiKey("401"))

    events = await ladder_seam.turn("hello")

    assert ladder_seam.provider.attempts_on(RouterDecision) == [SCHEMA_MODEL]
    assert ladder_seam.provider.slept == []
    assert events[-1].code == "turn_failed"
    assert events[-1].message == FALLBACK_ERROR


async def test_the_composer_climbs_the_same_ladder_as_every_other_call(ladder_seam):
    """`compose` is a provider call, so it is on the ladder — and it starts on
    Flash, so unlike the schema tier it has a rung below it to fall to."""
    ladder_seam.provider.script(TWO_INTENTS, DayRequest(days_ago=0), SUGGESTED, COMPOSED)
    ladder_seam.provider.fail_on(ComposedReply, *[ResourceExhausted("429")] * 3)

    events = await ladder_seam.turn("how did my day go, and what should I eat tonight?")

    assert ladder_seam.provider.attempts_on(ComposedReply) == [
        PROSE_MODEL, PROSE_MODEL, PROSE_MODEL, SCHEMA_MODEL
    ]
    assert ladder_seam.provider.slept == list(BACKOFF_SECONDS)
    # And the words arrived, with the User's name put back into them.
    assert answer(events).reply.text.startswith("Lou,")


async def test_a_composer_the_provider_never_answers_ends_the_turn_with_the_fallback(
    ladder_seam,
):
    """Not `COULD_NOT_COMPOSE`, which is what a schema the model could not fill
    twice ends on. A provider that stopped at every rung is the ladder running
    out, and that ends the Turn on the fixed message with an `error` event."""
    ladder_seam.provider.script(TWO_INTENTS, DayRequest(days_ago=0), SUGGESTED, COMPOSED)
    ladder_seam.provider.fail_on(ComposedReply, *[ResourceExhausted("429")] * MAX_ATTEMPTS)

    events = await ladder_seam.turn("how did my day go, and what should I eat tonight?")

    assert events[-1].code == "provider_unavailable"
    assert events[-1].message == FALLBACK_ERROR
    assert len(ladder_seam.provider.attempts_on(ComposedReply)) == MAX_ATTEMPTS


async def test_the_composers_prompt_is_never_trimmed_and_the_overrun_is_recorded(seam):
    """The node the 'true by construction' argument did not cover.

    `compose_reply` assembles a prompt out of parts, and one of those parts
    carries the day's totals — today's Meals. Every part is a `keep`, so a Turn
    far over budget reaches the composer with the totals whole and records the
    overrun instead.
    """
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(BOTH, ATE_EGGS, CHOSE_EGG, CITED, COMPOSED)

    await seam.turn(
        "I ate two eggs, is that enough protein? " + "eaten rice " * 5_000
    )

    composing = seam.provider.seen[-1].sent
    # Today's Meals, in full, in the prompt of a Turn that was far over budget.
    assert "25.2 g protein" in composing
    assert "this Meal included" in composing
    row = next(r for r in seam.db.events if r.node == "compose_reply")
    assert row.over_budget is True


async def test_a_provider_that_accepts_and_never_answers_climbs_the_ladder():
    """Counting attempts is not by itself a bound on time.

    A provider that takes the connection and then says nothing would hold a Turn
    open for ever, which is the one thing the four-attempt bound exists to
    prevent. The eval gate found it: two cases in sixty-four hung on a call that
    never answered. A timed-out attempt is transient, so it climbs.
    """
    import asyncio

    from nutrigraph_agent.providers import Models, ProviderUnavailable

    tried: list[str] = []

    class Hanging:
        def __init__(self, model: str) -> None:
            self.model = model

        def with_structured_output(self, schema, *, include_raw: bool = False):
            return self

        async def ainvoke(self, messages):
            tried.append(self.model)
            await asyncio.sleep(10)

    models = Models(
        factory=Hanging,
        schema_model=SCHEMA_MODEL,
        prose_model=PROSE_MODEL,
        attempt_seconds=0.01,
        sleep=lambda _seconds: asyncio.sleep(0),
    )

    with pytest.raises(ProviderUnavailable):
        await models.for_turn(known_names=[]).fill(
            RouterDecision, system="classify", user="anything"
        )

    # Three rungs, because the schema tier is already the weaker one and has
    # nothing below it to fall to — the ladder's own rule, not a fourth number.
    assert tried == [SCHEMA_MODEL] * len(ladder(SCHEMA_MODEL, SCHEMA_MODEL))


def test_the_ladder_has_four_rungs_and_the_last_one_is_the_weaker_model():
    plan = ladder("gemini-3.5-flash", "gemini-3.5-flash-lite")

    assert plan == [
        ("gemini-3.5-flash", 0.0),
        ("gemini-3.5-flash", 1.0),
        ("gemini-3.5-flash", 3.0),
        ("gemini-3.5-flash-lite", 0.0),
    ]
    assert len(plan) == MAX_ATTEMPTS


def test_a_turn_keeps_its_thirty_seconds_whatever_the_harness_asks_for():
    """The one thing the eval gate's patience may not do is become the Coach's.

    A User waiting on a Turn gets the bound issue #37 decided; the harness hands
    in its own, and the default is what everything that does not is left with.
    """
    from nutrigraph_agent.providers import ATTEMPT_SECONDS, Models

    default = Models(factory=lambda _m: None, schema_model="s", prose_model="p")
    patient = Models(
        factory=lambda _m: None, schema_model="s", prose_model="p", attempt_seconds=600.0
    )

    assert default.attempt_seconds == ATTEMPT_SECONDS == 30.0
    assert patient.attempt_seconds == 600.0


def test_the_rate_limiter_is_the_harness_and_production_builds_what_it_built(monkeypatch):
    """Pacing is what beats a quota, and a User is not what spends one.

    `langchain_factory` is asked for a model twice: once the way the process
    asks, and once the way the eval run does. Only the second carries a limiter,
    and the two models it hands back for the two tiers carry different ones,
    because the quota is counted per model.
    """
    import langchain.chat_models

    from nutrigraph_agent import providers

    built: list[dict] = []

    def fake_init_chat_model(model, **kwargs):
        built.append({"model": model, **kwargs})
        return kwargs

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", fake_init_chat_model)

    plain = providers.langchain_factory("google_genai")
    plain("gemini-3.5-flash")
    paced = providers.langchain_factory("google_genai", requests_per_minute=14.0)
    paced("gemini-3.5-flash")
    paced("gemini-3.5-flash-lite")
    paced("gemini-3.5-flash")

    # The production call is the one it always was: no limiter, not even a None.
    assert "rate_limiter" not in built[0]
    limiters = [call["rate_limiter"] for call in built[1:]]
    assert all(limiter is not None for limiter in limiters)
    # One bucket per model, and the same model gets the same bucket back.
    assert limiters[0] is limiters[2]
    assert limiters[0] is not limiters[1]
    assert limiters[0].requests_per_second == pytest.approx(14.0 / 60.0)


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


# What the Profile row and today's Meals look like once a node has rendered them
# for a prompt. `graph._held` writes the Profile as `field: value` lines, and
# `meal.day_line` opens with this sentence, so these strings appear where a node
# built one and nowhere else — the Thread's own history is the User's prose.
PROFILE_AND_TODAYS_MEALS = ("weight_kg:", "allergies:", "The day so far, this Meal included")


def modules_that_trim() -> list[Any]:
    """Every module holding a `fit`, found rather than listed.

    A node that assembles a prompt calls `fit`, and `from .budget import fit`
    binds a name in that module. Discovering them is what makes the test below a
    guard rather than a description: a node added next year is covered without
    anyone remembering this file exists.

    The source tree is read rather than every module imported, because importing
    `main` starts the process's configuration check, and `budget` is skipped
    because patching `fit` where it is defined would not reach the name a node
    already bound.
    """
    import importlib
    from pathlib import Path

    import nutrigraph_agent

    source = Path(nutrigraph_agent.__file__).parent
    found = []
    for path in sorted(source.glob("*.py")):
        if path.stem == "budget":
            continue
        if re.search(r"^from \.budget import .*\bfit\b", path.read_text("utf-8"), re.M):
            found.append(importlib.import_module(f"nutrigraph_agent.{path.stem}"))
    return found


@pytest.fixture
def trimmer(monkeypatch):
    """Every `fit` call any node makes, as it was made."""
    from nutrigraph_agent import budget

    calls: list[dict] = []
    real = budget.fit

    def recording(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    patched = modules_that_trim()
    assert patched, "no module assembles a prompt, which cannot be right"
    for module in patched:
        monkeypatch.setattr(module, "fit", recording)
    return calls


async def test_no_node_hands_the_profile_or_todays_meals_to_the_trimmer(seam, trimmer):
    """The guard the cost-controls slice could not give itself.

    It reported that "the Profile and today's Meals survive trimming" was true
    *by construction*: only some nodes assemble a multi-part prompt, and none of
    those held the Profile. That is a property of today's call sites, not a rule,
    and a node written next year could hand the Profile to `history=` and nothing
    would notice. This notices.

    Every `fit` call the graph makes is recorded, across every shape of Turn
    there is, and what may be cut — the history and the passages — is read for
    the Profile row and for the day's totals. `keep` is not read: `keep` is
    measured and never cut, which is the whole distinction.
    """
    seam.food.results = {"egg": [EGG]}
    seam.provider.script(
        RouterDecision(intents=["update_profile"], confidence=0.95),
        ProfileUpdate(field="weight_kg", new_value="80", old_value="78"),
        BOTH, ATE_EGGS, CHOSE_EGG, CITED, COMPOSED,
        TWO_INTENTS, DayRequest(days_ago=0), SUGGESTED, COMPOSED,
        ASKED, CITED,
    )
    # Far over budget, so every trimming branch runs rather than the early one.
    long = " " + "eaten rice " * 3_000
    await seam.turn("I weigh 80 kg now" + long)
    await seam.turn("I ate two eggs, is that enough protein?" + long)
    await seam.turn("how did my day go, and what should I eat tonight?" + long)
    await seam.turn(QUESTION + long)

    assert trimmer, "no node trimmed at all, so this proves nothing"
    for call in trimmer:
        trimmable = render_history(call.get("history") or []) + " ".join(
            getattr(passage, "text", "") for passage in call.get("passages") or ()
        )
        for rendered in PROFILE_AND_TODAYS_MEALS:
            assert rendered not in trimmable, (
                f"{rendered!r} was handed to the trimmer; the Profile and today's "
                f"Meals belong in `keep`, which is measured and never cut"
            )
    # And the battery really did put them in front of `fit`, in `keep`, rather
    # than passing because no node ever held them.
    kept = " ".join(text for call in trimmer for text in call["keep"])
    assert any(rendered in kept for rendered in PROFILE_AND_TODAYS_MEALS)


async def test_the_profile_and_todays_meals_reach_the_provider_on_an_over_budget_turn(seam):
    """The end of the same rule, at the seam. Only `route` and
    `answer_question` assemble a prompt out of parts, so they are the two nodes
    that trim; the Profile and today's Meals are never handed to the trimmer at
    all, and this is what that means for a Turn that is far over budget.
    """
    seam.provider.script(
        ATE, ParsedMeal(items=[ParsedItem(name="zzzfoodzzz", quantity=1)])
    )
    await seam.turn("I ate zzzfoodzzz")
    for i in range(1, 9):
        await seam.turn(f"marker-{i} " + "eaten rice " * 800)
    seam.provider.script(
        RouterDecision(intents=["update_profile"], confidence=0.95),
        ProfileUpdate(field="weight_kg", new_value="80", old_value="78"),
        ATE,
        ATE_EGGS,
        CHOSE_EGG,
    )
    seam.food.results = {"egg": [EGG]}
    before = len(seam.provider.seen)

    await seam.turn("I weigh 80 kg now")
    await seam.turn("I ate two eggs")

    prompts = "\n".join(c.sent for c in seam.provider.seen[before:])
    # The Profile, as the extractor is given it, and today's unmatched Meal
    # Item, which is what lets the next Turn offer a correction.
    assert "weight_kg: 78" in prompts
    assert "allergies: peanut" in prompts
    assert "zzzfoodzzz" in prompts
    # And the Thread was still cut, so this was an over-budget Turn.
    assert "marker-1" not in prompts


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
