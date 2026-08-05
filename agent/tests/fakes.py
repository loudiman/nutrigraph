"""The fakes the agent turn seam stands on: the database, and the provider.

The provider is faked at the chat-model boundary, one layer below the redaction
wrapper, so every test runs the real wrapper and can read exactly what Google
would have seen.
"""

from __future__ import annotations

import random
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from nutrigraph_agent.db import InteractionEvent, RetrievedChunk
from nutrigraph_agent.models import UPDATABLE_FIELDS, Profile, RouterDecision
from nutrigraph_agent.providers import Models

# What `gemini-embedding-001` returns before truncation.
PROVIDER_DIMENSIONS = 3072

# One Corpus passage, so a seam test can assert what a Citation names.
EGGS_CHUNK = RetrievedChunk(
    document="Dietary Guidelines for Americans, 2025-2030",
    source_url="https://cdn.realfood.gov/DGA_508.pdf",
    locator="page 3",
    text="Eat a variety of protein foods, including eggs, seafood, lean meats, "
    "beans, and nuts. Protein foods supply the amino acids the body cannot make.",
    licence_id="us-gov-public-domain",
    attribution="Dietary Guidelines for Americans, 2025-2030. A work of the "
    "United States federal government, public domain under 17 U.S.C. §105.",
    commercial_use=True,
    score=0.81,
)

DEMO_PROFILE = Profile(
    user_id="demo-user-1",
    name="Lou",
    sex="M",
    age=24,
    height_cm=172,
    weight_kg=78,
    target_weight_kg=72,
    activity_level="light",
    diet_pattern="omnivore",
    allergies=["peanut"],
)

NAME_PLACEHOLDER = re.compile(r"\[NAME_\d+\]")


@dataclass
class StoredMessage:
    user_id: str
    turn_id: UUID
    role: str
    raw_text: str


@dataclass
class FakeDatabase:
    profiles: dict[str, Profile] = field(
        default_factory=lambda: {DEMO_PROFILE.user_id: DEMO_PROFILE}
    )
    messages: list[StoredMessage] = field(default_factory=list)
    events: list[InteractionEvent] = field(default_factory=list)
    redaction_maps: dict[UUID, dict[str, str]] = field(default_factory=dict)
    # Every Profile handed to a Turn, in order, so a test can read exactly what
    # a later Turn was given rather than what the store happens to hold now.
    loaded: list[Profile] = field(default_factory=list)
    fail_on_load: bool = False
    # What the Corpus returns for any query vector. Empty means the Corpus does
    # not cover the question.
    corpus: list[RetrievedChunk] = field(default_factory=lambda: [EGGS_CHUNK])
    searched: list[list[float]] = field(default_factory=list)

    async def load_profile(self, user_id: str) -> Profile | None:
        if self.fail_on_load:
            raise ConnectionError("database is gone")
        profile = self.profiles.get(user_id)
        if profile is not None:
            self.loaded.append(profile)
        return profile

    async def update_profile(self, user_id: str, *, field: str, value: Any) -> None:
        if field not in UPDATABLE_FIELDS:
            raise ValueError(f"{field!r} is not a Profile field a User may change")
        self.profiles[user_id] = self.profiles[user_id].model_copy(update={field: value})

    async def store_message(
        self, *, user_id: str, turn_id: UUID, role: str, raw_text: str
    ) -> None:
        self.messages.append(StoredMessage(user_id, turn_id, role, raw_text))

    async def store_interaction_event(self, event: InteractionEvent) -> None:
        self.events.append(event)

    async def store_redaction_map(self, *, turn_id: UUID, mapping: dict[str, str]) -> None:
        self.redaction_maps.setdefault(turn_id, {}).update(mapping)

    async def search_corpus(
        self, embedding: Sequence[float], *, limit: int = 5
    ) -> list[RetrievedChunk]:
        self.searched.append(list(embedding))
        return self.corpus[:limit]


@dataclass
class ProviderCall:
    """One call as the provider received it. Nothing above this line ran."""

    model: str
    texts: list[str]

    @property
    def sent(self) -> str:
        return "\n".join(self.texts)


def _message(content: str) -> SimpleNamespace:
    """Shaped like a real answer: `content` is a list of blocks — Gemini 3 puts
    a thought signature beside the words — and the words are on `text`."""
    return SimpleNamespace(
        content=[{"type": "text", "text": content}, {"type": "reasoning", "signature": "…"}],
        text=content,
        usage_metadata={"input_tokens": 11, "output_tokens": 7},
    )


@dataclass
class FakeProvider:
    """Records every prompt that reached the provider, and answers from a
    script. A `str` in the script is a schema failure, which forces the retry.

    The script is in call order, and a Turn may now make more than one schema
    call: the router first, then the Intent path's own."""

    decisions: deque[BaseModel | str] = field(default_factory=deque)
    default: RouterDecision = field(
        default_factory=lambda: RouterDecision(intents=["log_meal"], confidence=0.92)
    )
    seen: list[ProviderCall] = field(default_factory=list)
    # Every text that reached the embedding model, as it reached it.
    embedded: list[str] = field(default_factory=list)
    # What the prose tier writes back, when a test needs to choose the words.
    prose: str | None = None

    def script(self, *decisions: BaseModel | str) -> FakeProvider:
        self.decisions.extend(decisions)
        return self

    def models(
        self,
        *,
        schema_model: str,
        prose_model: str,
        embedding_model: str = "gemini-embedding-001",
    ) -> Models:
        return Models(
            factory=lambda model: _FakeChat(self, model),
            schema_model=schema_model,
            prose_model=prose_model,
            embedding_factory=lambda model: _FakeEmbeddings(self, model),
            embedding_model=embedding_model,
        )

    def vector(self, text: str) -> list[float]:
        """What the provider returns: 3072 dimensions, and deliberately not unit
        length, so truncation and the hand re-normalization are what make the
        stored vector usable. Deterministic, so a test can repeat a query."""
        self.embedded.append(text)
        rng = random.Random(text)
        return [rng.uniform(-1.0, 1.0) * 7.0 for _ in range(PROVIDER_DIMENSIONS)]

    def _next(self) -> BaseModel | str:
        return self.decisions.popleft() if self.decisions else self.default


@dataclass
class _FakeChat:
    provider: FakeProvider
    model: str

    def _record(self, messages: list[dict[str, str]]) -> ProviderCall:
        call = ProviderCall(self.model, [m["content"] for m in messages])
        self.provider.seen.append(call)
        return call

    def with_structured_output(self, schema: Any, *, include_raw: bool = False) -> Any:
        return _FakeStructured(self, schema)

    async def ainvoke(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        call = self._record(messages)
        if self.provider.prose is not None:
            return _message(self.provider.prose)
        # A Coach answering a redacted prompt writes the placeholder back, which
        # is what lets the reply address the User by name.
        found = NAME_PLACEHOLDER.search(call.sent)
        who = f"{found.group()}, " if found else ""
        return _message(f"{who}what did you mean by that?")


@dataclass
class _FakeEmbeddings:
    """Faked one layer below the redaction wrapper, exactly like the chat model,
    so a test reads what Google would have seen."""

    provider: FakeProvider
    model: str

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.provider.vector(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.provider.vector(text)


@dataclass
class _FakeStructured:
    chat: _FakeChat
    schema: Any

    async def ainvoke(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        self.chat._record(messages)
        answer = self.chat.provider._next()
        if isinstance(answer, str):
            return {"raw": _message(""), "parsed": None, "parsing_error": ValueError(answer)}
        # A real provider cannot answer with the wrong schema, so a script that
        # does says the test is out of step with the calls the Turn makes.
        assert isinstance(answer, self.schema), (
            f"the script's next answer is a {type(answer).__name__}, but the Turn "
            f"asked for a {self.schema.__name__}"
        )
        return {
            "raw": _message(answer.model_dump_json()),
            "parsed": answer,
            "parsing_error": None,
        }
