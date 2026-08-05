"""Pydantic is the contract. FastAPI publishes these as an OpenAPI document and
a build step generates `gateway/src/generated/agent.ts` from it."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, RootModel


class Profile(BaseModel):
    """The stable facts the Coach holds about a User."""

    user_id: str
    name: str
    sex: str | None = None
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    target_weight_kg: float | None = None
    activity_level: str | None = None
    diet_pattern: str | None = None
    units: str = "metric"
    allergies: list[str] = Field(default_factory=list)
    disliked_foods: list[str] = Field(default_factory=list)


# The Profile fields the User may change by saying so. `user_id` and `name` are
# not among them: one identifies the row, the other is what the Redactor is
# built from. The database seam whitelists against this same tuple, so no field
# name reaches SQL that did not come from here.
UPDATABLE_FIELDS = (
    "sex", "age", "height_cm", "weight_kg", "target_weight_kg",
    "activity_level", "diet_pattern", "units", "allergies", "disliked_foods",
)

ProfileField = Literal[
    "sex", "age", "height_cm", "weight_kg", "target_weight_kg",
    "activity_level", "diet_pattern", "units", "allergies", "disliked_foods",
]

# The two Profile fields that hold a list rather than one value. A statement
# adds to them; it does not replace them.
LIST_FIELDS = ("allergies", "disliked_foods")


class ProfileUpdate(BaseModel):
    """One Profile fact the User changed by saying it.

    `old_value` is what makes the confirmation useful — the User can see the
    Coach changed the right thing — so the extractor is asked for it, and the
    node then reports the Profile's own value, which cannot be hallucinated.
    """

    field: ProfileField | None = Field(
        default=None,
        description="The one Profile field this message changes. Null when the "
        "message does not clearly change exactly one of them.",
    )
    old_value: str = Field(default="", description="What the Profile holds now.")
    new_value: str = Field(
        default="",
        description="The value alone: a number with no unit for a number field, "
        "and for allergies and disliked_foods the single food being added.",
    )


INTENTS = ("log_meal", "ask_question", "review_day", "recommend", "update_profile")

Intent = Literal["log_meal", "ask_question", "review_day", "recommend", "update_profile"]


class RouterDecision(BaseModel):
    """What one router call decides about a User message. The router detects an
    out-of-scope request; it never writes the Refusal, which is the guardrail's
    wording to give."""

    intents: list[Intent] = Field(
        default_factory=list,
        max_length=2,
        description="The Intents the message carries, in the order they must run, "
        "at most two. The second reads what the first produced.",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="How sure the classification is.")
    out_of_scope: bool = Field(
        default=False, description="The request falls outside what a nutrition Coach does."
    )


class ReplyPart(BaseModel):
    """One part of a reply, one for each Intent the Turn ran."""

    intent: str
    text: str


class CoachReply(BaseModel):
    """The Coach's complete answer to one User message."""

    text: str
    parts: list[ReplyPart] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)


class TurnRequest(BaseModel):
    user_id: str
    message: str = Field(min_length=1)


class NodeEvent(BaseModel):
    """A node of the graph finished. These go out while the Turn runs."""

    type: Literal["node"] = "node"
    turn_id: UUID
    node: str


class AnswerEvent(BaseModel):
    """The answer text, held back and sent as one event at the end.

    A later slice inserts the guardrail text scan immediately before this event
    is emitted, and does not have to change this contract.
    """

    type: Literal["answer"] = "answer"
    turn_id: UUID
    reply: CoachReply


class ErrorEvent(BaseModel):
    """A failure mid-Turn. The stream closes after this event."""

    type: Literal["error"] = "error"
    turn_id: UUID
    code: str
    message: str


TurnEvent = Annotated[NodeEvent | AnswerEvent | ErrorEvent, Field(discriminator="type")]


class TurnEventEnvelope(RootModel[TurnEvent]):
    """One line of the `application/x-ndjson` turn stream."""
