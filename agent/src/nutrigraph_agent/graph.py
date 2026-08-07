"""The graph. `load_profile` reads the Profile, `guard` runs the deterministic
rule list with no model, `route` classifies the message into Intents with one
model call, and the Turn refuses, runs its Intent paths, or asks one clarifying
question.

`guard` sits before `route` because a message the rule list catches must never
reach an Intent path — including the Corpus, which an out-of-scope question is
never allowed to search. `refuse` is the only node that writes a Refusal, and
both detectors — the rule list and the router's `out_of_scope` flag — end there.

Three Intent paths are built, all after both detectors, and `INTENT_PATHS` names
the node each starts at. `update_profile` writes the change to PostgreSQL and to
nothing else, so the Profile the next Turn reads is the changed one.
`ask_question` is `retrieve` then `answer_question`: a question the guardrail
permits, a general chronic-disease question among them, passes through `guard`
untouched and is answered from the Corpus with a Citation on every claim.
`log_meal` writes the Meal and then reads the day it counts against.

**The multi-Intent Turn, and it costs one loop edge.** The router returns an
ordered list of at most two Intents. Each Intent path appends one `IntentResult`
to `ctx.intent_results` and formats nothing for the User, and `after_intent`
either sends the Turn back for the second Intent or on to the composer. So "I
ate two eggs, is that enough protein" logs the Meal first, and the question is
then answered from a day total that already holds it.

**One composer.** `compose_reply` is the only node on an Intent path that
produces a `CoachReply`. The prototype ran without it, each path writing its own
reply, and the second Intent did its work silently while the User saw only the
first — which is the fault this node exists to prevent. `parts` and
`disclaimers` are assembled in code from what the paths reported, so a model
cannot drop an Intent from the record or a marking from the answer; only the
words are asked of a model, on the prose tier, because the User reads them.

`refuse` and `clarify` are outside that rule and stay outside it: neither runs
an Intent, a Refusal is a template from `guardrail` rather than anybody's
composition, and `clarify` is the graph's one interrupt, asked before any Intent
path runs. Nothing composes there because nothing ran.

**Order.** The composer runs after every Intent path, which is after the allergy
check on the paths that have one, and before the guardrail text scan in
`run_turn` — the answer is held back until that scan passes, so the composer
writing the text does not put it on the screen.

The state holds only what survives the Turn. Everything rebuilt every Turn
lives on the `TurnContext`, which travels in the config and therefore never
enters the checkpoint — `intent_results` among them, which is this Turn's
working note and not part of the Thread.

Every node is wrapped in `measured`, so a node cannot be added without leaving
an `interaction_event` row behind.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Annotated, Any, TypedDict
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .budget import fit, render_history
from .db import InteractionEvent, RetrievedChunk
from .deps import Deps
from .guardrail import OUT_OF_SCOPE, Subject, match_rule, refusal
from .meal import MANILA, day_bounds, day_line
from .meal import log_meal as run_log_meal
from .models import (
    INTENTS,
    LIST_FIELDS,
    UPDATABLE_FIELDS,
    Answer,
    Citation,
    CoachReply,
    ComposedReply,
    Profile,
    ProfileUpdate,
    ReplyPart,
    RouterDecision,
)
from .providers import ModelCall, SchemaFailure, TurnModels

# The key is underscore-prefixed so LangGraph keeps it out of checkpoint metadata.
TURN_CONTEXT_KEY = "__turn"

log = logging.getLogger("nutrigraph.agent.graph")

CHECKPOINTED = ("user_id", "messages", "pending_clarification")

# Below this the Turn cannot be classified, so the Coach asks instead of guessing.
CONFIDENCE_FLOOR = 0.6

# The Intent paths that are built, and the node each one starts at. An Intent
# with no entry here goes to the stub, so adding a path is one line and a node.
INTENT_PATHS = {
    "update_profile": "update_profile",
    "ask_question": "retrieve",
    "log_meal": "log_meal",
}

# The Intents whose answer is scanned against the Profile's allergies.
#
# `update_profile` is absent, and must stay absent. A correct confirmation of
# "I am allergic to shrimp" names shrimp, and an allergy check on that path
# destroys the very answer the User asked for. The prototype found this.
#
# There are two places the allergen check can arrive: a node on this path, and
# the allergen half of `guardrail.scan_reply`, which `run_turn` runs on the
# finished text of every answer that is not a Refusal. Neither may see an
# `update_profile` confirmation. `tests/test_update_profile.py` fails at both.
ALLERGY_CHECKED_INTENTS = ("recommend", "log_meal")

ROUTER_SYSTEM = f"""You classify one message from a User to a nutrition Coach.

Return the Intents the message carries, in the order they must run, at most two.
The Intents are: {", ".join(INTENTS)}.

log_meal        the User says what they ate or drank
ask_question    the User asks a nutrition question
review_day      the User asks how the logged day went
recommend       the User asks what to eat next, or how to reach their Goal
update_profile  the User states a stable fact about themselves: weight, height,
                allergy, diet pattern, target weight, activity level

Set confidence to how sure you are of that classification. Set out_of_scope when
the request is not a nutrition Coach's job — a medical diagnosis, a prescription,
a mental-health crisis, or something unrelated to food. Still classify it; the
Coach decides what to say about it, not you.

Return no Intents when the message carries none."""

UPDATE_PROFILE_SYSTEM = f"""You read one message in which a User states a stable
fact about themselves, and you report the single Profile field it changes.

The fields: {", ".join(UPDATABLE_FIELDS)}.

age is whole years. height_cm, weight_kg and target_weight_kg are numbers in
those units. activity_level is one of sedentary, light, moderate, active.
diet_pattern is one of omnivore, pescatarian, vegetarian, vegan, halal. units is
metric or imperial. allergies and disliked_foods each hold a list of foods, and
a message adds one food to the list.

new_value is the value alone, with no unit and no sentence around it. For
allergies and disliked_foods it is the single food being added, lowercase and
singular. Convert to the field's unit when the User uses another one.

old_value is what the Profile holds now, which you are given.

Name no field when the message does not clearly change exactly one of them —
when it is vague, when it changes two, or when the value is missing. The Coach
then asks the User rather than guessing a field."""

CLARIFY_SYSTEM = """You are a nutrition Coach who did not understand one message.

Ask exactly one short question that would let you classify it. One sentence, no
preamble, no list, no apology. Address the User by the placeholder you are given,
written exactly as it appears."""

# How many Corpus passages one question is answered from.
PASSAGES = 5
# Cosine similarity below which the Corpus does not cover the question. Under
# it the Coach says so and makes no provider call at all, so there is no room
# for an answer from the model's memory.
RELEVANCE_FLOOR = 0.55

NOT_IN_THE_CORPUS = (
    "That is not in the nutrition guidance I cite from, so I would rather point "
    "you at a dietitian than guess at it."
)

COMPOSE_SYSTEM = """You are a nutrition Coach who has just done two things for
one User in one message, and you write the single reply that covers both.

You are given the parts, in the order they were done, each with the sentences
that part already produced and the facts behind it. Join them into one reply.

Keep every fact and every marking. A sentence saying a number was assumed, is a
stand-in for another food, is calculated rather than measured, or that a total
is short, has to survive into your reply — those markings are what stop a
number being read as more certain than it is.

Make no nutrition claim of your own. Everything you assert comes from a part you
were given; where a part answers a question, answer it from the facts and the
sentences in that part, and from nothing else. Where a part reports the day's
totals, use those numbers and not your own arithmetic.

At most five short sentences. The User is reading this while cooking. Address
them by the placeholder you are given, written exactly as it appears."""

# What a Turn ends with when the composer could not fill the schema twice. It is
# fixed text in code, like a Refusal: whatever went wrong with the words, the
# work the Intent paths did was already written down before this node ran.
COULD_NOT_COMPOSE = (
    "I did what you asked, but I could not put it into words this time. Nothing "
    "you told me was lost. Ask me again and I will read it back to you."
)

ANSWER_SYSTEM = """You are a nutrition Coach answering one question from the
Corpus passages given to you, and from nothing else.

Answer in at most three short sentences: the User is reading this while cooking.

Every nutrition claim you make carries a Citation. A Citation names the
passage's document and its section or page, copied exactly as they are written
above the passage. An answer that asserts a nutrition fact with no Citation is
rejected, so cite or do not claim.

If the passages do not answer the question, say so plainly in one sentence, set
makes_a_nutrition_claim to false, give no citations, and invent nothing. Never
answer a nutrition question from your own memory."""


class TurnState(TypedDict):
    user_id: str
    messages: Annotated[list[dict[str, str]], operator.add]
    # Survives until a Turn is understood. A clarify node replaces the value
    # rather than adding a second one, because this key is a plain overwrite.
    pending_clarification: str | None


@dataclass
class IntentResult:
    """What one Intent path did, for the composer to read.

    `text` is the sentences the path assembled itself — facts it already holds,
    which is why they are assembled in code and not asked of a model. `facts` is
    what the path knows that the User does not need read back word for word: the
    day's totals, for instance, which the next Intent answers against.

    `disclaimers` are the markings that may not be lost. The composer carries
    them onto the reply and puts back any the words dropped.
    """

    intent: str
    text: str
    facts: str = ""
    citations: list[Citation] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)


@dataclass
class TurnContext:
    """Rebuilt every Turn, never checkpointed."""

    deps: Deps
    turn_id: UUID
    raw_message: str
    profile: Profile | None = None
    models: TurnModels | None = None
    decision: RouterDecision | None = None
    reply: CoachReply | None = None
    # One for each Intent that ran, in the order they ran. This lives here and
    # not in `TurnState` because it is this Turn's working note: the Thread keeps
    # what was said, not how the Coach got there, so it never enters the
    # checkpoint. `CHECKPOINTED` is the list of keys that do.
    intent_results: list[IntentResult] = field(default_factory=list)
    # What retrieval found this Turn. Rebuilt every Turn, so it never enters the
    # checkpoint — a Corpus passage is not part of the Thread.
    passages: list[RetrievedChunk] = field(default_factory=list)
    # What the rule list caught, if anything, and whether the Turn was refused.
    # A Refusal is this codebase's own words, so the text scan does not read it.
    subject: Subject | None = None
    refused: bool = False
    nodes_run: list[str] = field(default_factory=list)
    # What the node currently running has to report on its metric row.
    call: ModelCall | None = None
    intent: str | None = None
    # The node's prompt was still over the token budget after trimming. It ran
    # anyway; the row says so.
    over_budget: bool = False

    def record(self, call: ModelCall) -> None:
        self.call = call

    @property
    def name(self) -> str:
        return self.profile.name if self.profile else "there"


class UnknownUser(LookupError):
    """No Profile for this user_id. The Turn cannot run."""


def turn_context(config: RunnableConfig) -> TurnContext:
    return config["configurable"][TURN_CONTEXT_KEY]


def models(ctx: TurnContext) -> TurnModels:
    if ctx.models is None:  # pragma: no cover - load_profile always runs first
        raise RuntimeError("no provider call may happen before the Profile is loaded")
    return ctx.models


async def load_profile(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    ctx = turn_context(config)
    profile = await ctx.deps.db.load_profile(state["user_id"])
    if profile is None:
        raise UnknownUser(state["user_id"])
    ctx.profile = profile
    # The Redactor is built from the names the Coach already holds, and every
    # provider call this Turn makes goes through the object it is bound to.
    ctx.models = ctx.deps.models.for_turn(known_names=[profile.name])
    return {}


async def guard(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The deterministic detector, before the router and with no model. What it
    catches goes straight to `refuse`, so no Intent path runs for it."""
    ctx = turn_context(config)
    ctx.subject = match_rule(ctx.raw_message)
    return {}


def after_guard(state: TurnState, config: RunnableConfig) -> str:
    return "refuse" if turn_context(config).subject else "route"


async def route(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """One call, one schema. No keyword list is maintained for routing."""
    ctx = turn_context(config)
    pending = state.get("pending_clarification")
    asked = f"\n\nYou last asked the User: {pending}" if pending else ""
    # The Thread, because a message that only makes sense after the last one
    # cannot be classified without it. The current message is already the last
    # entry, so the history is everything before it — trimmed to the last six
    # turns when the Turn is over budget, and never summarised.
    fitted = fit(
        keep=[ROUTER_SYSTEM, asked, ctx.raw_message],
        history=state.get("messages", [])[:-1],
    )
    ctx.over_budget = fitted.over_budget
    user = ctx.raw_message
    if fitted.history:
        user = (
            f"Earlier in this Thread:\n{render_history(fitted.history)}\n\n"
            f"The message to classify: {ctx.raw_message}"
        )
    decision, call = await models(ctx).fill(
        RouterDecision, system=ROUTER_SYSTEM + asked, user=user
    )
    ctx.decision = decision
    ctx.record(call)
    ctx.intent = decision.intents[0] if decision.intents else None
    if decision.confidence >= CONFIDENCE_FLOOR and not decision.out_of_scope:
        # The Turn was understood, so the pending Clarification is answered. A
        # Refusal turn takes neither branch, so a Refusal does not clear it.
        return {"pending_clarification": None}
    return {}


def pending_intent(ctx: TurnContext) -> str | None:
    """The Intent whose turn it is: the next one the router named that has not
    reported a result yet. Each path appends exactly one, so the count is the
    position, and there is no second bookkeeping to fall out of step."""
    decision = ctx.decision
    assert decision is not None
    at = len(ctx.intent_results)
    return decision.intents[at] if at < len(decision.intents) else None


def next_node(state: TurnState, config: RunnableConfig) -> str:
    decision = turn_context(config).decision
    assert decision is not None
    # Out of scope beats an unsure classification, and beats every Intent path:
    # refusing is the safer answer to a request the Coach may not take either
    # way. Both this and the rule list in `guard` are decided before a path is
    # chosen, so an out-of-scope message never reaches one — the Corpus among
    # them, which such a question is never allowed to search.
    if decision.out_of_scope:
        return "refuse"
    if decision.confidence < CONFIDENCE_FLOOR:
        return "clarify"
    # The first Intent, because the order matters and the second reads what the
    # first produced. `after_intent` brings the Turn back here for the second.
    first = decision.intents[0] if decision.intents else None
    return INTENT_PATHS.get(first, "dispatch")


def after_intent(state: TurnState, config: RunnableConfig) -> str:
    """The loop edge. Back to the second Intent's path, or on to the composer.

    Every Intent path ends here, so the composer runs after all of them and
    after the allergy check on the paths that have one — and no path can reach
    the end of the Turn without its result being composed into the reply.
    """
    ctx = turn_context(config)
    remaining = pending_intent(ctx)
    if remaining is None:
        return "compose_reply"
    return INTENT_PATHS.get(remaining, "dispatch")


async def refuse(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The only node that writes a Refusal, and it writes a template. Whichever
    detector fired, the wording comes from `guardrail`, never from a model."""
    ctx = turn_context(config)
    ctx.reply = refusal(ctx.subject or OUT_OF_SCOPE)
    ctx.refused = True
    ctx.intent = "refusal"
    # `pending_clarification` is not returned, so a Refusal leaves it standing.
    return {"messages": [{"role": "coach", "text": ctx.reply.text}]}


async def clarify(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """One short question, and the Turn ends. The only point at which the Coach
    stops and waits for the User."""
    ctx = turn_context(config)
    question, call = await models(ctx).write(
        system=CLARIFY_SYSTEM,
        user=f"The User, whose name is {ctx.name}, wrote: {ctx.raw_message}",
    )
    ctx.record(call)
    ctx.reply = CoachReply(text=question, parts=[ReplyPart(intent="clarify", text=question)])
    return {
        "pending_clarification": question,
        "messages": [{"role": "coach", "text": question}],
    }


async def retrieve(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The question becomes a vector and the Corpus answers with passages. One
    embedding call, through the same wrapper every other provider call uses.

    The lookup cache sits between the vector and the Corpus. It holds the
    passages a near-enough question already found — public guidance, keyed on
    the embedding of the question and on nothing this User has — so a repeat
    skips the search entirely. It never holds the answer written from them: the
    Profile is what makes an answer this User's, and nothing that read the
    Profile is ever cached.
    """
    ctx = turn_context(config)
    ctx.intent = "ask_question"
    vector, key_text, call = await models(ctx).embed_query(ctx.raw_message)
    ctx.record(call)
    found = await ctx.deps.db.cached_retrieval(vector)
    if found is None:
        found = await ctx.deps.db.search_corpus(vector, limit=PASSAGES)
        await ctx.deps.db.store_cached_retrieval(
            key_text=key_text, embedding=vector, chunks=found
        )
    ctx.passages = [chunk for chunk in found if chunk.score >= RELEVANCE_FLOOR]
    return {}


def _passages(ctx: TurnContext) -> str:
    return "\n\n".join(
        f"document: {chunk.document}\nsection or page: {chunk.locator}\n{chunk.text}"
        for chunk in ctx.passages
    )


async def answer_question(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The cited Answer. When nothing was retrieved the Coach says so, and no
    provider call is made — there is nowhere for an invented claim to come from."""
    ctx = turn_context(config)
    ctx.intent = "ask_question"
    # The passages are what the budget trims here; the question is not, and
    # neither is the instruction that every claim carries a Citation.
    fitted = fit(keep=[ANSWER_SYSTEM, ctx.raw_message], passages=ctx.passages)
    ctx.over_budget = fitted.over_budget
    ctx.passages = fitted.passages
    if not ctx.passages:
        answer = Answer(text=NOT_IN_THE_CORPUS, makes_a_nutrition_claim=False)
    else:
        turn = models(ctx)
        answer, call = await turn.fill(
            Answer,
            system=ANSWER_SYSTEM,
            user=f"Question: {ctx.raw_message}\n\nPassages:\n\n{_passages(ctx)}",
        )
        ctx.record(call)
        # `fill` hands back what the provider wrote, which still holds the
        # placeholders. Put the identifiers back before the User reads it.
        answer = answer.model_copy(
            update={
                "text": turn.restore(answer.text),
                "citations": [
                    citation.model_copy(
                        update={
                            "document": turn.restore(citation.document),
                            "locator": turn.restore(citation.locator),
                        }
                    )
                    for citation in answer.citations
                ],
            }
        )
    ctx.intent_results.append(
        IntentResult(
            intent="ask_question", text=answer.text, citations=answer.citations
        )
    )
    return {}


def _say(value: Any) -> str:
    """One Profile value as the User should read it back."""
    if isinstance(value, list):
        return ", ".join(value) or "nothing"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # a target weight is 70, not 70.0
    return "not set" if value in (None, "") else str(value)


def _held(profile: Profile) -> str:
    """What the Profile holds now, for the extractor to report as the old value."""
    return "\n".join(f"{name}: {_say(getattr(profile, name))}" for name in UPDATABLE_FIELDS)


def _changed(profile: Profile, update: ProfileUpdate) -> tuple[Any, Any]:
    """The old value and the new one, typed as the Profile holds them.

    `Profile` does the coercion, so "70" becomes 70.0 and a value the field
    cannot hold raises instead of reaching PostgreSQL.
    """
    assert update.field is not None
    old = getattr(profile, update.field)
    if update.field in LIST_FIELDS:
        item = update.new_value.strip().lower()
        if not item:
            raise ValueError("no food named")
        # ponytail: a statement adds to the list. Removing an allergy by saying
        # so is its own slice, and needs the extractor to report the direction.
        candidate: Any = old if item in old else [*old, item]
    else:
        candidate = update.new_value
    validated = Profile.model_validate({**profile.model_dump(), update.field: candidate})
    return old, getattr(validated, update.field)


async def update_profile(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The User changes a Profile fact by saying it.

    The write goes to PostgreSQL and to nothing else. The Profile is not in the
    state and never enters the checkpoint, so there is no second copy to drift,
    and the next Turn's `load_profile` reads the change back — in this Session
    or in a new one.

    Appending no result sends the Turn to `clarify`: an ambiguous Profile
    statement is a question to the User, not a guessed field.
    """
    ctx = turn_context(config)
    ctx.intent = "update_profile"
    profile = ctx.profile
    assert profile is not None
    update, call = await models(ctx).fill(
        ProfileUpdate,
        system=UPDATE_PROFILE_SYSTEM,
        user=f"The Profile holds:\n{_held(profile)}\n\nThe User wrote: {ctx.raw_message}",
    )
    ctx.record(call)
    if update.field is None:
        log.info("no Profile field named", extra={"turn_id": str(ctx.turn_id)})
        return {}
    try:
        # The extractor reports an old value too; this is the Profile's own,
        # so the confirmation cannot name a value the User never had.
        old, new = _changed(profile, update)
    except (ValidationError, ValueError):
        log.info("value does not fit the field", extra={"field": update.field})
        return {}
    await ctx.deps.db.update_profile(profile.user_id, field=update.field, value=new)
    text = (
        f"{ctx.name}, I changed your {update.field.replace('_', ' ')} "
        f"from {_say(old)} to {_say(new)}."
    )
    ctx.intent_results.append(IntentResult(intent="update_profile", text=text))
    return {}


def after_update(state: TurnState, config: RunnableConfig) -> str:
    """A Profile statement the extractor could not pin to one field goes to the
    clarify path rather than to a guess.

    The Intent still standing after the node ran is how that is known, and it
    reads the same whether this was the Turn's first Intent or its second.
    """
    ctx = turn_context(config)
    if pending_intent(ctx) == "update_profile":
        return "clarify"
    return after_intent(state, config)


async def log_meal(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The User says what they ate, and it is written down.

    A food the Coach could not match does not send the Turn to `clarify`: the
    answer names it and invites a correction instead, because `clarify` is the
    only interrupt in the graph and meal logging is not it.

    The day's totals are read *after* the Meal is written, and they travel on the
    result as facts. That ordering is what makes "I ate two eggs, is that enough
    protein" one Turn rather than two: the second Intent answers against a total
    that already holds this Meal.
    """
    ctx = turn_context(config)
    ctx.intent = "log_meal"
    profile = ctx.profile
    assert profile is not None
    # One clock for the Meal Type, the `eaten_at` stamp and the day it counts
    # against, so the three cannot disagree about which day it was.
    now = datetime.now(MANILA)
    logged = await run_log_meal(
        db=ctx.deps.db,
        food=ctx.deps.food,
        turn=models(ctx),
        profile=profile,
        message=ctx.raw_message,
        turn_id=ctx.turn_id,
        now=now,
    )
    # Both schema calls the node made — the parse and any choice — on one row.
    ctx.record(logged.call)
    start, end = day_bounds(now)
    total = await ctx.deps.db.day_total(profile.user_id, start=start, end=end)
    ctx.intent_results.append(
        IntentResult(
            intent="log_meal",
            text=logged.text,
            facts=day_line(total),
            disclaimers=logged.disclaimers,
        )
    )
    return {}


async def dispatch(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The stub each remaining Intent path replaces. It reports what the router
    decided for one Intent, like any other path, and the composer says it.

    A Turn whose two Intents both land here runs it twice, once for each, which
    is exactly what a Turn with two built paths does.
    """
    ctx = turn_context(config)
    intent = pending_intent(ctx)
    ctx.intent = intent
    named = intent or "nothing I handle"
    ctx.intent_results.append(
        IntentResult(intent=intent or "none", text=f"{ctx.name}, I read that as: {named}.")
    )
    return {}


def _for_composer(ctx: TurnContext) -> str:
    """Every part, in the order it ran, as the composer reads it."""
    blocks = [f"The User wrote: {ctx.raw_message}"]
    for position, result in enumerate(ctx.intent_results, start=1):
        block = f"Part {position}, {result.intent}:\n{result.text}"
        if result.facts:
            block += f"\nThe facts behind it: {result.facts}"
        blocks.append(block)
    return "\n\n".join(blocks)


async def compose_reply(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The one node that produces a `CoachReply` for an Intent path.

    One part needs no composing: there is nothing to join, and a model asked to
    rewrite sentences a path already assembled from facts can only lose one. Two
    parts are worth a call, and it goes to the prose tier because the User reads
    what comes back. The schema failure retries once with the error inside
    `compose`, so both attempts are two calls in the trace, and a second failure
    ends the Turn on the fixed message rather than on a half-written answer.

    Whichever way the words arrive, `parts` and `disclaimers` are built here from
    what the paths reported. So `parts` holds one entry for every Intent that
    ran, and a marking raised on any path survives even if the words dropped it.

    Nothing in this prompt may be trimmed, and the budget is told so rather than
    left to infer it: one of the parts carries the day's totals, which is
    today's Meals. A Turn over budget here runs anyway and records the overrun,
    like every other node.
    """
    ctx = turn_context(config)
    results = ctx.intent_results
    ctx.intent = results[-1].intent if results else None
    disclaimers = list(dict.fromkeys(d for r in results for d in r.disclaimers))

    if len(results) <= 1:
        text = results[0].text if results else COULD_NOT_COMPOSE
    else:
        turn = models(ctx)
        prompt = _for_composer(ctx)
        # Every part of this prompt is a `keep`. The sentences a path assembled
        # are facts it already holds, and the facts behind them are the day's
        # totals — which is today's Meals. There is nothing here the budget may
        # trim, so all it can do is say the Turn went over, and the row carries
        # it. This is the node that would otherwise be the hole in the rule: it
        # assembles a prompt out of parts, and one of those parts is exactly
        # what may never be dropped.
        ctx.over_budget = fit(keep=[COMPOSE_SYSTEM, prompt]).over_budget
        try:
            composed, call = await turn.compose(
                ComposedReply, system=COMPOSE_SYSTEM, user=prompt
            )
            ctx.record(call)
            # The provider wrote the placeholders back; the User reads the name.
            text = turn.restore(composed.text)
        except SchemaFailure:
            log.warning(
                "the composer did not fill the schema twice",
                extra={"turn_id": str(ctx.turn_id)},
            )
            text = COULD_NOT_COMPOSE

    # A marking the words left out is put back rather than lost. It is the one
    # thing in the reply a model does not get to be brief about.
    text = " ".join([text, *(d for d in disclaimers if d not in text)])
    ctx.reply = CoachReply(
        text=text,
        parts=[
            ReplyPart(intent=r.intent, text=r.text, citations=r.citations)
            for r in results
        ],
        disclaimers=disclaimers,
    )
    return {"messages": [{"role": "coach", "text": text}]}


def measured(name: str, node: Any) -> Any:
    """Every node writes an `interaction_event` row: node, Intent, model,
    latency, token counts, and cost. The wrapping happens in `build_graph`, so
    a new node cannot be added without one."""

    async def run(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        ctx = turn_context(config)
        ctx.call, ctx.intent, ctx.over_budget = None, None, False
        started = perf_counter()
        try:
            return await node(state, config)
        finally:
            ctx.nodes_run.append(name)
            event = InteractionEvent(
                turn_id=ctx.turn_id,
                user_id=state["user_id"],
                node=name,
                intent=ctx.intent,
                model=ctx.call.model if ctx.call else None,
                latency_ms=int((perf_counter() - started) * 1000),
                input_tokens=ctx.call.input_tokens if ctx.call else 0,
                output_tokens=ctx.call.output_tokens if ctx.call else 0,
                cost_usd=ctx.call.cost_usd if ctx.call else 0.0,
                over_budget=ctx.over_budget,
            )
            try:
                await ctx.deps.db.store_interaction_event(event)
            except Exception:
                # A metric is not worth a Turn. An unknown user_id has no row to
                # point the foreign key at, and that node still ran.
                log.warning("interaction_event not written", extra={"node": name})

    return run


def build_graph(checkpointer: Any):
    builder = StateGraph(TurnState)
    for name, node in (
        ("load_profile", load_profile),
        ("guard", guard),
        ("route", route),
        ("clarify", clarify),
        ("update_profile", update_profile),
        ("log_meal", log_meal),
        ("retrieve", retrieve),
        ("answer_question", answer_question),
        ("dispatch", dispatch),
        ("compose_reply", compose_reply),
        ("refuse", refuse),
    ):
        builder.add_node(name, measured(name, node))
    builder.add_edge(START, "load_profile")
    builder.add_edge("load_profile", "guard")
    builder.add_conditional_edges("guard", after_guard, ["refuse", "route"])
    builder.add_conditional_edges(
        "route",
        next_node,
        ["clarify", "update_profile", "log_meal", "retrieve", "dispatch", "refuse"],
    )
    # Where every Intent path ends: back for the second Intent, or on to the one
    # composer. The loop edge is these lists naming each other's entry nodes.
    onwards = ["update_profile", "log_meal", "retrieve", "dispatch", "compose_reply"]
    builder.add_conditional_edges("update_profile", after_update, [*onwards, "clarify"])
    # Logging a Meal never interrupts the Turn, so `clarify` is not among these.
    builder.add_conditional_edges("log_meal", after_intent, onwards)
    builder.add_edge("retrieve", "answer_question")
    builder.add_conditional_edges("answer_question", after_intent, onwards)
    builder.add_conditional_edges("dispatch", after_intent, onwards)
    builder.add_edge("compose_reply", END)
    builder.add_edge("clarify", END)
    builder.add_edge("refuse", END)
    return builder.compile(checkpointer=checkpointer)
