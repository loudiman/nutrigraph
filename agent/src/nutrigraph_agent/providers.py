"""The one place in the process that talks to a model provider.

**The model routing rule, not a list of nodes.** Work that fills a fixed schema
from text uses the schema tier; work that reasons or writes prose for the User
uses the prose tier. A node inherits its model from the rule by choosing `fill`
or `write`, and never names a model.

**The redaction wrapper.** `fill` and `write` redact before the call and restore
after it, including on the retry after a schema failure (ADR 0002). A node
cannot opt out: it holds a `TurnModels`, which owns the Redactor, and there is
no way through to the provider that skips it.

**The provider is a configuration string.** `init_chat_model` is called on one
line, here, and nowhere else in the codebase — a test asserts that. Changing
`MODEL_PROVIDER` in `.env` from `google_genai` to `openai` or `anthropic`, and
the two model names beside it, moves every call. There is no code path per
provider to leave behind.

**The retry ladder, bounded at four attempts.** On a free tier a rate-limit stop
is normal traffic, not an exception, and the ladder treats it that way: two
retries about a second and then three seconds apart, then the same call on the
weaker tier, because a slightly worse answer beats no answer. A stop that is not
transient — a schema the provider could not fill, a bad key — is raised at once
rather than retried three times for nothing. Because the ladder is bounded, a
Turn can never hang; when it runs out, `ProviderUnavailable` reaches `run_turn`,
which ends the Turn with the fixed fallback message and an `error` event.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .redaction import Redacted, Redactor

log = logging.getLogger("nutrigraph.agent.models")

Schema = TypeVar("Schema", bound=BaseModel)
Result = TypeVar("Result")

# `gemini-embedding-001` returns 3072 dimensions; pgvector's HNSW index accepts
# at most 2000, so the full output cannot be indexed (ADR 0001).
HNSW_MAX_DIMENSIONS = 2000
EMBEDDING_DIMENSIONS = 768


class SchemaFailure(RuntimeError):
    """The provider could not fill the schema, and the retry did not either."""


class ProviderUnavailable(RuntimeError):
    """Every rung of the ladder failed. The Turn ends with the fallback."""


# How long the two retries wait, in seconds. About one, and then about three.
BACKOFF_SECONDS = (1.0, 3.0)

# The hard bound: three on the first model, one on the weaker tier. A Turn can
# never hang, because there is no fifth attempt to make.
MAX_ATTEMPTS = 4

# What is worth retrying, read off the exception rather than off an imported
# vendor type: the provider is a configuration string, and importing
# `google.api_core.exceptions` here would be the code path per vendor that this
# module exists not to have. Every provider names these stops much the same way,
# and an HTTP status settles the rest.
TRANSIENT_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError", "DeadlineExceeded", "InternalError",
    "InternalServerError", "OverloadedError", "RateLimitError", "ReadTimeout",
    "ResourceExhausted", "ServerError", "ServiceUnavailable",
    "ServiceUnavailableError", "TooManyRequests", "UnavailableError",
})
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def is_transient(exc: BaseException) -> bool:
    """A rate-limit stop, a timeout, or a transient provider error."""
    if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
        return True
    for attribute in ("status_code", "http_status", "code"):
        status = getattr(exc, attribute, None)
        if isinstance(status, int) and status in TRANSIENT_STATUS:
            return True
    return any(base.__name__ in TRANSIENT_NAMES for base in type(exc).__mro__)


def ladder(primary: str, fallback: str | None) -> list[tuple[str, float]]:
    """The four rungs: the model to call, and how long to wait first.

    The fallback rung is dropped when it names the model that already failed —
    the schema tier is the weaker tier, so there is nothing below it to fall to,
    and a fourth identical attempt would only spend another second.
    """
    plan = [(primary, 0.0)] + [(primary, wait) for wait in BACKOFF_SECONDS]
    if fallback and fallback != primary:
        plan.append((fallback, 0.0))
    return plan[:MAX_ATTEMPTS]

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
    # What the ladder waits with. A test hands in a recorder, so the two waits
    # are asserted rather than sat through.
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    @property
    def fallback_model(self) -> str:
        """The rung below Flash. The routing rule already names two tiers and
        the schema tier is the cheaper one, so the weaker model is configured
        and there is no third name to keep in step with the other two."""
        return self.schema_model

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

    async def _climb(
        self,
        primary: str,
        fallback: str | None,
        run: Callable[[str], Awaitable[Result]],
    ) -> tuple[Result, str]:
        """Run one call up the ladder. Returns what came back and which model
        answered, because the weaker tier bills at its own price."""
        failure: BaseException | None = None
        for model, wait in ladder(primary, fallback):
            if wait:
                await self.models.sleep(wait)
            try:
                return await run(model), model
            except Exception as exc:
                if not is_transient(exc):
                    # A schema the provider could not fill, or a bad key. Three
                    # more attempts would fail the same way.
                    raise
                failure = exc
                log.warning(
                    "provider stopped, climbing the ladder",
                    extra={"model": model, "error": f"{type(exc).__name__}: {exc}"},
                )
        raise ProviderUnavailable(
            f"{primary} did not answer in {len(ladder(primary, fallback))} attempts"
        ) from failure

    async def fill(
        self, schema: type[Schema], *, system: str, user: str, retries: int = 1
    ) -> tuple[Schema, ModelCall]:
        """Fill a fixed schema from text. The schema tier, by the routing rule."""
        primary = self.models.schema_model
        model_name, tokens_in, tokens_out = primary, 0, 0
        note = ""
        for attempt in range(retries + 1):
            # Redaction runs on every attempt, so the retry after a schema
            # failure cannot reach the provider unredacted (ADR 0002). The
            # ladder's own retries reuse this redacted copy: it is the same call
            # made again, and there is no unredacted text in scope to send.
            redacted = self._redact(system + note, user)
            messages = [
                {"role": "system", "content": redacted.texts[0]},
                {"role": "user", "content": redacted.texts[1]},
            ]

            async def call(model: str) -> Any:
                chat = self.models.factory(model).with_structured_output(
                    schema, include_raw=True
                )
                return await chat.ainvoke(messages)

            result, model_name = await self._climb(
                primary, self.models.fallback_model, call
            )
            usage = _usage(result.get("raw"), model_name)
            tokens_in += usage.input_tokens
            tokens_out += usage.output_tokens
            parsed, error = result.get("parsed"), result.get("parsing_error")
            if parsed is not None and error is None:
                return parsed, ModelCall(model_name, tokens_in, tokens_out)
            log.warning("schema failure, retrying", extra={"attempt": attempt, "error": str(error)})
            note = f"\n\nThe previous answer did not fit the schema: {error}. Answer again."
        raise SchemaFailure(f"{primary} did not fill {schema.__name__} in {retries + 1} attempts")

    async def write(self, *, system: str, user: str) -> tuple[str, ModelCall]:
        """Write prose for the User. The prose tier, by the routing rule."""
        redacted = self._redact(system, user)
        messages = [
            {"role": "system", "content": redacted.texts[0]},
            {"role": "user", "content": redacted.texts[1]},
        ]
        message, model_name = await self._climb(
            self.models.prose_model,
            self.models.fallback_model,
            lambda model: self.models.factory(model).ainvoke(messages),
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
        # No fallback rung: the vector tier has one model, and answering with a
        # vector from a different one would poison the index.
        vectors, _ = await self._climb(
            self.models.embedding_model,
            None,
            lambda _model: self._embedder().aembed_documents(list(redacted.texts)),
        )
        return (
            [truncate_and_normalize(vector) for vector in vectors],
            ModelCall(model=self.models.embedding_model),
        )

    async def embed_query(self, text: str) -> tuple[list[float], str, ModelCall]:
        """The User's question, which is exactly the text redaction is for.

        The redacted question comes back beside the vector because it is what
        the lookup cache is keyed on in `key_text`: a cache entry outlives the
        90-day purge of `message.raw_text`, so the copy kept there is the one
        the provider saw and not the one the User typed.
        """
        redacted = self._redact(text)
        vector, _ = await self._climb(
            self.models.embedding_model,
            None,
            lambda _model: self._embedder().aembed_query(redacted.text),
        )
        return (
            truncate_and_normalize(vector),
            redacted.text,
            ModelCall(model=self.models.embedding_model),
        )
