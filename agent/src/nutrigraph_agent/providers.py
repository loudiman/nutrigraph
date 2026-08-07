"""The one place in the process that talks to a model provider.

**The model routing rule, not a list of nodes.** Work that fills a fixed schema
from text uses the schema tier; work that reasons or writes prose for the User
uses the prose tier. A node inherits its model from the rule by choosing `fill`,
`write` or `compose`, and never names a model.

`compose` is the third method the rule produces rather than a fourth tier: the
composer writes prose for the User, so it takes the prose model, and its output
is schema-validated, so it takes the same structured-output path and the same
one retry as `fill`. What it must not do is take the schema tier because it
happens to fill a schema — the User reads what it writes.

**The redaction wrapper.** `fill` and `write` redact before the call and restore
after it, including on the retry after a schema failure (ADR 0002). A node
cannot opt out: it holds a `TurnModels`, which owns the Redactor, and there is
no way through to the provider that skips it.

**The provider is a configuration string.** `init_chat_model` is called on one
line, here, and nowhere else in the codebase — a test asserts that. Changing
`MODEL_PROVIDER` in `.env` from `google_genai` to `openai` or `anthropic`, and
the two model names beside it, moves every call. There is no code path per
provider to leave behind.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .redaction import Redacted, Redactor

log = logging.getLogger("nutrigraph.agent.models")

Schema = TypeVar("Schema", bound=BaseModel)

# `gemini-embedding-001` returns 3072 dimensions; pgvector's HNSW index accepts
# at most 2000, so the full output cannot be indexed (ADR 0001).
HNSW_MAX_DIMENSIONS = 2000
EMBEDDING_DIMENSIONS = 768


class SchemaFailure(RuntimeError):
    """The provider could not fill the schema, and the retry did not either."""

# US dollars per million tokens, input then output, from
# https://ai.google.dev/gemini-api/docs/pricing. The free tier bills nothing;
# the interaction_event row records what the Turn would have cost.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}


@dataclass(frozen=True)
class ModelCall:
    """What one provider call cost, for the interaction_event row."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        per_in, per_out = PRICES.get(self.model, (0.0, 0.0))
        return (self.input_tokens * per_in + self.output_tokens * per_out) / 1_000_000

    def __add__(self, other: ModelCall) -> ModelCall:
        """Two calls one node made, on one `interaction_event` row."""
        return ModelCall(
            model=self.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ChatModel(Protocol):
    """The slice of a LangChain chat model this codebase uses."""

    def with_structured_output(self, schema: Any, *, include_raw: bool = ...) -> Any: ...

    async def ainvoke(self, input: Any) -> Any: ...


class EmbeddingModel(Protocol):
    """The slice of a LangChain embedding model this codebase uses."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...


# A model name in, a model out. `init_chat_model` and `init_embeddings` are these.
ModelFactory = Callable[[str], ChatModel]
EmbeddingFactory = Callable[[str], EmbeddingModel]


def truncate_and_normalize(
    vector: Sequence[float], dimensions: int = EMBEDDING_DIMENSIONS
) -> list[float]:
    """Matryoshka truncation to what an HNSW index accepts, then unit length by
    hand — version 1 of `gemini-embedding-001` does not re-normalize a truncated
    vector, so cosine distance would be wrong without this (ADR 0001)."""
    if dimensions > HNSW_MAX_DIMENSIONS:
        raise ValueError(f"HNSW accepts at most {HNSW_MAX_DIMENSIONS} dimensions")
    head = list(vector[:dimensions])
    if len(head) < dimensions:
        raise ValueError(f"the embedding has {len(head)} dimensions, fewer than {dimensions}")
    norm = math.sqrt(sum(value * value for value in head))
    if norm == 0.0:
        raise ValueError("the embedding truncated to a zero vector")
    return [value / norm for value in head]


def langchain_factory(provider: str, *, temperature: float = 0.0) -> ModelFactory:
    """The provider is a configuration string, not a code path.

    Temperature 0: the router classifies, it does not invent. Some Gemini models
    have fixed sampling and warn that they ignore it, which is the provider's
    business — the request still asks for the deterministic setting, so the same
    line is correct after a provider swap.
    """
    from langchain.chat_models import init_chat_model

    return lambda model: init_chat_model(
        model, model_provider=provider, temperature=temperature
    )


def langchain_embedding_factory(provider: str) -> EmbeddingFactory:
    """The same rule for the vector half of the system: one line, one provider
    string, no code path per vendor."""
    from langchain.embeddings import init_embeddings

    return lambda model: init_embeddings(model, provider=provider)


def _usage(raw: Any, model: str) -> ModelCall:
    usage = getattr(raw, "usage_metadata", None) or {}
    return ModelCall(
        model=model,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
    )


@dataclass(frozen=True)
class Models:
    """The provider seam. A Turn binds it to a Redactor and gets `TurnModels`."""

    factory: ModelFactory
    schema_model: str
    prose_model: str
    # The vector tier. Ingestion and retrieval both reach it through `TurnModels`,
    # so an embedding call redacts exactly like a chat call does.
    embedding_factory: EmbeddingFactory | None = None
    embedding_model: str = "gemini-embedding-001"

    def for_turn(self, *, known_names: list[str]) -> TurnModels:
        return TurnModels(self, Redactor(known_names=known_names))


@dataclass
class TurnModels:
    """Every provider call one Turn makes. Each one redacts first."""

    models: Models
    redactor: Redactor
    # Every placeholder this Turn handed to a provider, for the private table.
    mapping: dict[str, str] = field(default_factory=dict)

    def _redact(self, *texts: str) -> Redacted:
        redacted = self.redactor.redact(*texts)
        self.mapping.update(redacted.mapping)
        return redacted

    def restore(self, text: str) -> str:
        """Put back every identifier this Turn hid, so an answer the provider
        filled into a schema can still name the User and the document."""
        for placeholder, original in self.mapping.items():
            text = text.replace(placeholder, original)
        return text

    def _embedder(self) -> EmbeddingModel:
        factory = self.models.embedding_factory
        if factory is None:
            raise RuntimeError("no embedding provider is configured")
        return factory(self.models.embedding_model)

    async def fill(
        self, schema: type[Schema], *, system: str, user: str, retries: int = 1
    ) -> tuple[Schema, ModelCall]:
        """Fill a fixed schema from text. The schema tier, by the routing rule."""
        return await self._fill(
            schema, self.models.schema_model, system=system, user=user, retries=retries
        )

    async def compose(
        self, schema: type[Schema], *, system: str, user: str, retries: int = 1
    ) -> tuple[Schema, ModelCall]:
        """Write prose for the User into a fixed schema. The prose tier.

        The one retry is `fill`'s: the failure and the corrected attempt are two
        provider calls, so both appear in the trace rather than the first one
        disappearing behind the second.
        """
        return await self._fill(
            schema, self.models.prose_model, system=system, user=user, retries=retries
        )

    async def _fill(
        self, schema: type[Schema], model_name: str, *, system: str, user: str, retries: int
    ) -> tuple[Schema, ModelCall]:
        chat = self.models.factory(model_name).with_structured_output(
            schema, include_raw=True
        )
        total = ModelCall(model=model_name)
        note = ""
        for attempt in range(retries + 1):
            # Redaction runs on every attempt, so the retry after a schema
            # failure cannot reach the provider unredacted (ADR 0002).
            redacted = self._redact(system + note, user)
            result = await chat.ainvoke(
                [
                    {"role": "system", "content": redacted.texts[0]},
                    {"role": "user", "content": redacted.texts[1]},
                ]
            )
            total = total + _usage(result.get("raw"), model_name)
            parsed, error = result.get("parsed"), result.get("parsing_error")
            if parsed is not None and error is None:
                return parsed, total
            log.warning("schema failure, retrying", extra={"attempt": attempt, "error": str(error)})
            note = f"\n\nThe previous answer did not fit the schema: {error}. Answer again."
        raise SchemaFailure(f"{model_name} did not fill {schema.__name__} in {retries + 1} attempts")

    async def write(self, *, system: str, user: str) -> tuple[str, ModelCall]:
        """Write prose for the User. The prose tier, by the routing rule."""
        model_name = self.models.prose_model
        redacted = self._redact(system, user)
        message = await self.models.factory(model_name).ainvoke(
            [
                {"role": "system", "content": redacted.texts[0]},
                {"role": "user", "content": redacted.texts[1]},
            ]
        )
        # `content` is a list of blocks — Gemini 3 puts a thought signature
        # beside the words — so the text comes off the message, not off content.
        text = getattr(message, "text", None)
        if not isinstance(text, str) and callable(text):  # langchain-core < 1.0
            text = text()
        if not isinstance(text, str):  # pragma: no cover - no provider does this
            text = message.content if isinstance(message.content, str) else ""
        # The answer comes back holding placeholders, so the Coach can address
        # the User by name without the provider ever having seen it.
        return redacted.restore(text).strip(), _usage(message, model_name)

    async def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], ModelCall]:
        """Corpus text, for the index. The Corpus is public guidance rather than
        user data, but the route to the provider is the same one — there is no
        second way out of the process (ADR 0002)."""
        redacted = self._redact(*texts)
        vectors = await self._embedder().aembed_documents(list(redacted.texts))
        return (
            [truncate_and_normalize(vector) for vector in vectors],
            ModelCall(model=self.models.embedding_model),
        )

    async def embed_query(self, text: str) -> tuple[list[float], ModelCall]:
        """The User's question, which is exactly the text redaction is for."""
        redacted = self._redact(text)
        vector = await self._embedder().aembed_query(redacted.text)
        return truncate_and_normalize(vector), ModelCall(model=self.models.embedding_model)
