"""The allergy check, at the agent turn seam.

Every assertion here is a plain comparison. Nothing in this file, and nothing in
the code it drives, asks a model whether an answer is safe — a judge that can be
talked out of a refusal is not a safety check, and the two comparisons the Coach
runs are a word match against a text array and a word match against a sentence.

The two paths that are checked are `recommend` and `log_meal`. `recommend` has
no path of its own yet, so the Turns that stand in for it here drive `run_turn`
with a graph that writes the draft the seam has to judge. That is the seam's
own contract — a `CoachReply` on the context, and the events out — so the check
is tested exactly where it runs, and it will keep testing `recommend` when the
real path arrives.

The trap tests are at the bottom, beside the ones in `test_update_profile.py`:
the check must not be added to `update_profile`, `ask_question` or `review_day`,
where a correct answer may name the allergen as a fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nutrigraph_agent.graph import ALLERGY_CHECKED_INTENTS, TURN_CONTEXT_KEY
from nutrigraph_agent.guardrail import (
    DISCLAIMER,
    HELPLINE,
    SAFE_MESSAGE,
    allergens_in_prose,
    allergens_named,
    food_sentences,
)
from nutrigraph_agent.meal import DECLARED_SERVING_G, NOTHING_TO_LOG, TELL_ME, item_words
from nutrigraph_agent.models import (
    INTENTS,
    Answer,
    Citation,
    CoachReply,
    ParsedItem,
    ParsedMeal,
    ReplyPart,
    RouterDecision,
)
from nutrigraph_agent.turn import run_turn

from .conftest import answer

LOGGED = RouterDecision(intents=["log_meal"], confidence=0.95)
ASKED = RouterDecision(intents=["ask_question"], confidence=0.95)

# Kare-kare is the case the structured comparison exists for. The User says
# "kare-kare", the dish table matches "Kare-kare (beef)", and neither word is
# peanut — the table's own `peanut` tag is the only structured place that knows.
KARE_KARE = ParsedMeal(items=[ParsedItem(name="kare-kare", quantity=1, unit="bowl")])

# Pancit molo is the case the prose scan exists for, and it is a real row rather
# than an invented one. The dish is tagged soup, pork, wheat and gluten, so no
# structured field says shrimp; FNRI's own note about the proxy does, in a
# sentence quoted into the answer.
PANCIT_MOLO = ParsedMeal(items=[ParsedItem(name="pancit molo", quantity=1, unit="bowl")])


def allergic_to(seam, *foods: str) -> None:
    """The Profile the Turn will load, with the allergy list under test."""
    user = "demo-user-1"
    seam.db.profiles[user] = seam.db.profiles[user].model_copy(
        update={"allergies": list(foods)}
    )


# --- the structured comparison -------------------------------------------------


async def test_a_logged_meal_whose_items_hold_an_allergen_warns_in_the_answer(seam):
    seam.provider.script(LOGGED, KARE_KARE)

    events = await seam.turn("I had a bowl of kare-kare")

    text = answer(events).reply.text
    assert "kare-kare matches peanut on your allergy list" in text
    # The Meal is still written. A record of what the User ate is not edited to
    # make the answer safe; the answer says what the record means instead.
    assert [i.row.said_as for i in seam.db.items] == ["kare-kare"]
    assert seam.db.items[0].row.status == "matched"


async def test_the_structured_check_compares_against_the_profile_row_with_no_join(seam):
    """`allergies` is a text array on the row `load_profile` already read, so
    the comparison needs no query of its own and makes none."""
    seam.provider.script(LOGGED, KARE_KARE)

    events = await seam.turn("I had a bowl of kare-kare")

    assert seam.db.loaded[-1].allergies == ["peanut"]
    # The only lookups the Turn made are the dish table's own, for the food.
    assert seam.db.looked_up == ["kare kare"]
    assert "peanut" in answer(events).reply.text


async def test_the_warning_names_every_allergy_the_item_matches(seam):
    allergic_to(seam, "peanut", "beef")
    seam.provider.script(LOGGED, KARE_KARE)

    events = await seam.turn("I had a bowl of kare-kare")

    assert "matches peanut and beef on your allergy list" in answer(events).reply.text


async def test_a_meal_with_no_allergen_in_it_carries_no_warning(seam):
    seam.provider.script(
        LOGGED, ParsedMeal(items=[ParsedItem(name="pandesal", quantity=1, unit="piece")])
    )

    events = await seam.turn("I had pandesal")

    assert "allergy list" not in answer(events).reply.text


def test_the_comparison_is_an_exact_word_match_and_not_a_substring_one():
    assert allergens_named(["peanut"], "a thick peanut sauce") == ["peanut"]
    assert allergens_named(["peanut"], "two peanuts") == ["peanut"]
    # The plural is the only inflection, and a word that merely contains the
    # allergen is not the allergen.
    assert allergens_named(["nut"], "a doughnut") == []
    assert allergens_named(["shrimp"], "shrimp-paste") == ["shrimp"]
    assert allergens_named([], "anything at all") == []
    assert allergens_named([" "], "anything at all") == []


def test_the_structured_words_are_the_names_and_the_dish_tables_tags():
    from nutrigraph_agent.db import MealItemRow

    row = MealItemRow(
        ordinal=0,
        said_as="kare-kare",
        status="matched",
        food_name="Kare-kare (beef)",
        nutrients={"name": "Kare-kare (beef)", "tags": ["beef", "canned", "peanut"]},
    )
    assert allergens_named(["peanut"], item_words(row)) == ["peanut"]
    # A FoodData Central row carries no tags, and then it is the two names.
    plain = MealItemRow(ordinal=0, said_as="egg", status="matched", nutrients={"fdcId": 1})
    assert item_words(plain).split() == ["egg"]


# --- the prose scan ------------------------------------------------------------


async def test_an_allergen_named_only_in_prose_is_struck_out_and_written_again(seam):
    """FNRI's note on the pancit molo proxy says the entry is not the
    pork-and-shrimp filling a household uses. No Item says shrimp, so only the
    prose scan can see it — and the answer that reaches the User does not."""
    allergic_to(seam, "shrimp")
    seam.provider.script(LOGGED, PANCIT_MOLO)

    events = await seam.turn("I had a bowl of pancit molo")

    text = answer(events).reply.text
    assert "shrimp" not in text.lower()
    # The answer is written again, not replaced: what was logged is still said.
    assert "pancit molo" in text
    assert text != SAFE_MESSAGE
    assert TELL_ME in text


async def test_the_regeneration_drops_the_quotation_and_keeps_the_marking(seam):
    """What was struck out is FNRI's sentence, not the fact that the figures are
    a stand-in. The marking is the thing the User depends on and it survives."""
    allergic_to(seam, "shrimp")
    seam.provider.script(LOGGED, PANCIT_MOLO)

    events = await seam.turn("I had a bowl of pancit molo")

    text = answer(events).reply.text
    assert "figures are a stand-in and not the dish itself" in text
    assert "the PhilFCT entry 'Wonton soup, prep, w/ MLP'." in text
    assert "pork-and-shrimp filling" not in text
    assert "I logged" in text and "pancit molo" in text


async def test_the_regeneration_costs_no_provider_call_and_no_model_judge(seam):
    """Two schema calls: the router, and the parse. Nothing was asked to judge
    the answer, and writing it again asked nobody anything."""
    allergic_to(seam, "shrimp")
    seam.provider.script(LOGGED, PANCIT_MOLO)

    await seam.turn("I had a bowl of pancit molo")

    assert len(seam.provider.seen) == 2


async def test_an_allergen_the_items_hold_does_not_trigger_the_prose_scan(seam):
    """The warning names the allergen on purpose. Reading the Coach's own
    warning as a violation would take the warning down with the answer."""
    seam.provider.script(LOGGED, KARE_KARE)

    events = await seam.turn("I had a bowl of kare-kare")

    text = answer(events).reply.text
    assert text != SAFE_MESSAGE
    assert "peanut" in text


def test_the_prose_scan_ignores_what_the_structured_items_already_hold():
    draft = "I logged lunch: kare-kare. Careful, kare-kare matches peanut on your allergy list."
    assert allergens_in_prose(["peanut"], draft, ["kare-kare Kare-kare (beef) peanut"]) == []
    # The same sentence with nothing structured behind it is a conflict.
    assert allergens_in_prose(["peanut"], draft, []) == ["peanut"]


def test_the_prose_scan_reads_only_the_food_sentences_of_the_draft():
    """The Coach's own fixed sentences are this codebase's words, not a model's,
    and none of them offers the User a food."""
    fixed = (DISCLAIMER, HELPLINE, TELL_ME,
             f"Where you did not give a weight I counted {int(DECLARED_SERVING_G)} g "
             f"for one serving, which is my assumption and not a measurement.",
             "My source prints no fibre and sodium for part of that, so those "
             "totals are short rather than complete.")
    for sentence in fixed:
        assert food_sentences(sentence) == [], sentence

    draft = "I logged dinner: rice. " + TELL_ME
    assert food_sentences(draft) == ["I logged dinner: rice."]


# --- exactly one regeneration, and then the safe message ------------------------


@dataclass
class DraftingGraph:
    """A graph that writes the drafts the seam has to judge, and counts how many
    times it was asked for another one.

    This stands in for `recommend`, which has no path of its own yet. It talks
    to `run_turn` through the seam's own contract — the reply, the Intent, the
    structured foods, and the way to write the answer again — so what it proves
    is what the real path will meet.
    """

    drafts: list[str]
    intent: str = "recommend"
    foods: list[str] = field(default_factory=list)
    regenerates: bool = True
    asked: list[list[str]] = field(default_factory=list)

    async def astream(self, inputs, config, stream_mode):
        ctx = config["configurable"][TURN_CONTEXT_KEY]
        ctx.profile = await ctx.deps.db.load_profile(inputs["user_id"])
        ctx.intent = self.intent
        ctx.foods = self.foods
        ctx.reply = self._reply(self.drafts[0])
        if self.regenerates:
            ctx.redraft = self._again
        yield {self.intent: {}}

    def _reply(self, text: str) -> CoachReply:
        return CoachReply(text=text, parts=[ReplyPart(intent=self.intent, text=text)])

    async def _again(self, without):
        self.asked.append(list(without))
        return self._reply(self.drafts[len(self.asked)])


async def drive(seam, graph, message: str = "what should I eat tonight?"):
    from uuid import uuid4

    return [
        event
        async for event in run_turn(
            graph, seam.deps, user_id="demo-user-1", turn_id=uuid4(), message=message
        )
    ]


async def test_a_recommendation_naming_an_allergen_is_never_returned_to_the_user(seam):
    graph = DraftingGraph(
        drafts=["Try grilled bangus tonight, with a peanut sauce on the side.",
                "Try grilled bangus tonight, with a peanut sauce on the side."]
    )

    events = await drive(seam, graph)

    assert answer(events).reply.text == SAFE_MESSAGE
    assert "peanut" not in answer(events).reply.text


async def test_a_conflict_forces_exactly_one_regeneration(seam):
    graph = DraftingGraph(
        drafts=["Try bangus with a peanut sauce.", "Try bangus with calamansi."]
    )

    events = await drive(seam, graph)

    assert graph.asked == [["peanut"]]  # asked once, and told what to remove
    assert answer(events).reply.text == "Try bangus with calamansi."


async def test_a_second_conflict_returns_the_fixed_safe_message_and_the_turn_ends(seam):
    graph = DraftingGraph(
        drafts=["Try bangus with a peanut sauce.", "Try adobo with peanuts instead."]
    )

    events = await drive(seam, graph)

    assert len(graph.asked) == 1  # one regeneration, and no third attempt
    assert answer(events).reply.text == SAFE_MESSAGE
    assert answer(events) is events[-1]


async def test_a_path_that_cannot_write_the_answer_again_goes_straight_to_safe(seam):
    graph = DraftingGraph(drafts=["Try bangus with a peanut sauce."], regenerates=False)

    events = await drive(seam, graph)

    assert answer(events).reply.text == SAFE_MESSAGE


async def test_the_blocked_answer_is_what_is_stored_not_the_draft(seam):
    """The transcript holds what the User was told. A draft the check struck
    down was never an answer."""
    graph = DraftingGraph(drafts=["Try a peanut sauce."] * 2)

    await drive(seam, graph)

    assert [m.raw_text for m in seam.db.messages][-1] == SAFE_MESSAGE


# --- the trap: the check runs on two paths and no others -----------------------


def test_the_check_lists_recommend_and_log_meal_and_nothing_else():
    assert set(ALLERGY_CHECKED_INTENTS) == {"recommend", "log_meal"}
    assert set(ALLERGY_CHECKED_INTENTS) <= set(INTENTS)


@pytest.mark.parametrize("intent", ["update_profile", "ask_question", "review_day"])
async def test_the_check_does_not_run_on_a_forbidden_path(seam, intent):
    """A correct answer on these paths may name the allergen as a fact. This
    fails if one of them is added to `ALLERGY_CHECKED_INTENTS`."""
    assert intent not in ALLERGY_CHECKED_INTENTS, (
        f"{intent} was added to the allergy check; a correct answer on that path "
        f"may name the allergen, and the check would destroy it"
    )
    draft = "Peanut allergy is one of the most common food allergies."
    graph = DraftingGraph(drafts=[draft, "something else"], intent=intent)

    events = await drive(seam, graph, "tell me about peanut allergy")

    assert answer(events).reply.text == draft
    assert graph.asked == []


async def test_a_cited_answer_about_the_users_own_allergen_is_not_blocked(seam):
    """`ask_question` end to end, through the real graph. The Corpus answer names
    peanut, the User is allergic to peanut, and the answer still goes out."""
    seam.provider.script(
        ASKED,
        Answer(
            text="Peanut is one of the most common food allergens.",
            citations=[Citation(document="Dietary Guidelines for Americans, "
                                         "2025-2030", locator="page 3")],
        ),
    )

    events = await seam.turn("is peanut a common allergen?")

    assert "Peanut" in answer(events).reply.text


async def test_a_meal_the_coach_could_not_read_is_not_blocked_by_the_check(seam):
    """Nothing logged, nothing structured, and no allergen. The check has to be
    silent rather than turning an empty answer into the safe message."""
    allergic_to(seam, "shrimp")
    seam.provider.script(LOGGED, ParsedMeal(items=[]))

    events = await seam.turn("I ate")

    assert NOTHING_TO_LOG in answer(events).reply.text
