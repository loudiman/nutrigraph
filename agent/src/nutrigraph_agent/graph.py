"""The graph. `load_profile` reads the Profile, `route` classifies the message
into Intents with one model call, and the Turn either dispatches or asks one
clarifying question.

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

from .db import InteractionEvent, RetrievedChunk
from .deps import Deps
from .models import (
    INTENTS,
    Answer,
    CoachReply,
    Profile,
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
class TurnContext:
    """Rebuilt every Turn, never checkpointed."""

    deps: Deps
    turn_id: UUID
    raw_message: str
    profile: Profile | None = None
    models: TurnModels | None = None
    decision: RouterDecision | None = None
    reply: CoachReply | None = None
    # What retrieval found this Turn. Rebuilt every Turn, so it never enters the
    # checkpoint — a Corpus passage is not part of the Thread.
    passages: list[RetrievedChunk] = field(default_factory=list)
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
    if decision.confidence >= CONFIDENCE_FLOOR:
        # The Turn was understood, so the pending Clarification is answered. A
        # Refusal turn never reaches here, so a Refusal does not clear it.
        return {"pending_clarification": None}
    return {}


def next_node(state: TurnState, config: RunnableConfig) -> str:
    decision = turn_context(config).decision
    assert decision is not None
    if decision.confidence < CONFIDENCE_FLOOR:
        return "clarify"
    # The first Intent, not any Intent: the order matters, because the second
    # reads what the first produced, and no other Intent path is built yet. A
    # message that logs a Meal and then asks about it still goes to the stub.
    # ponytail: this becomes an ordered walk of `decision.intents` when the
    # second path lands; the router already keeps them in the order to run.
    return "retrieve" if decision.intents[:1] == ["ask_question"] else "dispatch"


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
    embedding call, through the same wrapper every other provider call uses."""
    ctx = turn_context(config)
    ctx.intent = "ask_question"
    vector, call = await models(ctx).embed_query(ctx.raw_message)
    ctx.record(call)
    found = await ctx.deps.db.search_corpus(vector, limit=PASSAGES)
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
    ctx.reply = CoachReply(
        text=answer.text,
        parts=[
            ReplyPart(
                intent="ask_question", text=answer.text, citations=answer.citations
            )
        ],
    )
    return {"messages": [{"role": "coach", "text": answer.text}]}


async def dispatch(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """The stub each Intent path replaces. No Intent path is built yet, so this
    says what the router decided and stops."""
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
        ("route", route),
        ("clarify", clarify),
        ("retrieve", retrieve),
        ("answer_question", answer_question),
        ("dispatch", dispatch),
    ):
        builder.add_node(name, measured(name, node))
    builder.add_edge(START, "load_profile")
    builder.add_edge("load_profile", "route")
    builder.add_conditional_edges("route", next_node, ["clarify", "retrieve", "dispatch"])
    builder.add_edge("clarify", END)
    builder.add_edge("retrieve", "answer_question")
    builder.add_edge("answer_question", END)
    builder.add_edge("dispatch", END)
    return builder.compile(checkpointer=checkpointer)
