"""The graph. `load_profile` reads the Profile, `guard` runs the deterministic
rule list with no model, `route` classifies the message into Intents with one
model call, and the Turn refuses, runs an Intent path, dispatches, or asks one
clarifying question.

`guard` sits before `route` because a message the rule list catches must never
reach an Intent path. `refuse` is the only node that writes a Refusal, and both
detectors — the rule list and the router's `out_of_scope` flag — end there.

`update_profile` is the first Intent path, and it sits after both detectors. It
writes the change to PostgreSQL and to nothing else, so the Profile the next
Turn reads is the changed one.

The state holds only what survives the Turn. Everything rebuilt every Turn
lives on the `TurnContext`, which travels in the config and therefore never
enters the checkpoint.

Every node is wrapped in `measured`, so a node cannot be added without leaving
an `interaction_event` row behind.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated, Any, TypedDict
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .db import InteractionEvent
from .deps import Deps
from .guardrail import OUT_OF_SCOPE, Subject, match_rule, refusal
from .models import (
    INTENTS,
    LIST_FIELDS,
    UPDATABLE_FIELDS,
    CoachReply,
    Profile,
    ProfileUpdate,
    ReplyPart,
    RouterDecision,
)
from .providers import ModelCall, TurnModels

# The key is underscore-prefixed so LangGraph keeps it out of checkpoint metadata.
TURN_CONTEXT_KEY = "__turn"

log = logging.getLogger("nutrigraph.agent.graph")

CHECKPOINTED = ("user_id", "messages", "pending_clarification")

# Below this the Turn cannot be classified, so the Coach asks instead of guessing.
CONFIDENCE_FLOOR = 0.6

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


class TurnState(TypedDict):
    user_id: str
    messages: Annotated[list[dict[str, str]], operator.add]
    # Survives until a Turn is understood. A clarify node replaces the value
    # rather than adding a second one, because this key is a plain overwrite.
    pending_clarification: str | None


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
    # What the rule list caught, if anything, and whether the Turn was refused.
    # A Refusal is this codebase's own words, so the text scan does not read it.
    subject: Subject | None = None
    refused: bool = False
    nodes_run: list[str] = field(default_factory=list)
    # What the node currently running has to report on its metric row.
    call: ModelCall | None = None
    intent: str | None = None

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
    decision, call = await models(ctx).fill(
        RouterDecision, system=ROUTER_SYSTEM + asked, user=ctx.raw_message
    )
    ctx.decision = decision
    ctx.record(call)
    ctx.intent = decision.intents[0] if decision.intents else None
    if decision.confidence >= CONFIDENCE_FLOOR and not decision.out_of_scope:
        # The Turn was understood, so the pending Clarification is answered. A
        # Refusal turn takes neither branch, so a Refusal does not clear it.
        return {"pending_clarification": None}
    return {}


def next_node(state: TurnState, config: RunnableConfig) -> str:
    decision = turn_context(config).decision
    assert decision is not None
    # Out of scope beats an unsure classification: refusing is the safer answer
    # to a request the Coach may not take either way. Both this and the rule
    # list in `guard` are decided before any Intent path is chosen, so an
    # out-of-scope message never reaches one.
    if decision.out_of_scope:
        return "refuse"
    if decision.confidence < CONFIDENCE_FLOOR:
        return "clarify"
    # The first Intent is the one that runs. Chaining a second one onto the
    # first is a later slice; until then the stub says what was decided.
    if decision.intents and decision.intents[0] == "update_profile":
        return "update_profile"
    return "dispatch"


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

    Leaving `ctx.reply` unset sends the Turn to `clarify`: an ambiguous Profile
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
    ctx.reply = CoachReply(
        text=text, parts=[ReplyPart(intent="update_profile", text=text)]
    )
    return {"messages": [{"role": "coach", "text": text}]}


def after_update(state: TurnState, config: RunnableConfig) -> str:
    """A Profile statement the extractor could not pin to one field goes to the
    clarify path rather than to a guess."""
    return END if turn_context(config).reply is not None else "clarify"


async def dispatch(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The stub each remaining Intent path replaces. It says what the router
    decided and stops."""
    ctx = turn_context(config)
    decision = ctx.decision
    assert decision is not None
    ctx.intent = decision.intents[0] if decision.intents else None
    named = ", ".join(decision.intents) or "nothing I handle"
    text = f"{ctx.name}, I read that as: {named}."
    ctx.reply = CoachReply(
        text=text,
        parts=[ReplyPart(intent=intent, text=text) for intent in decision.intents]
        or [ReplyPart(intent="none", text=text)],
    )
    return {"messages": [{"role": "coach", "text": text}]}


def measured(name: str, node: Any) -> Any:
    """Every node writes an `interaction_event` row: node, Intent, model,
    latency, token counts, and cost. The wrapping happens in `build_graph`, so
    a new node cannot be added without one."""

    async def run(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        ctx = turn_context(config)
        ctx.call, ctx.intent = None, None
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
        ("dispatch", dispatch),
        ("refuse", refuse),
    ):
        builder.add_node(name, measured(name, node))
    builder.add_edge(START, "load_profile")
    builder.add_edge("load_profile", "guard")
    builder.add_conditional_edges("guard", after_guard, ["refuse", "route"])
    builder.add_conditional_edges(
        "route", next_node, ["clarify", "update_profile", "dispatch", "refuse"]
    )
    builder.add_conditional_edges("update_profile", after_update, ["clarify", END])
    builder.add_edge("clarify", END)
    builder.add_edge("dispatch", END)
    builder.add_edge("refuse", END)
    return builder.compile(checkpointer=checkpointer)
