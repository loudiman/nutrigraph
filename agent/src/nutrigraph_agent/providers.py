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
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .redaction import Redacted, Redactor

log = logging.getLogger("nutrigraph.agent.models")

Schema = TypeVar("Schema", bound=BaseModel)


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


class ChatModel(Protocol):
    """The slice of a LangChain chat model this codebase uses."""

    def with_structured_output(self, schema: Any, *, include_raw: bool = ...) -> Any: ...

    async def ainvoke(self, input: Any) -> Any: ...


# A model name in, a chat model out. `init_chat_model` is one of these.
ModelFactory = Callable[[str], ChatModel]


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

    async def fill(
        self, schema: type[Schema], *, system: str, user: str, retries: int = 1
    ) -> tuple[Schema, ModelCall]:
        """Fill a fixed schema from text. The schema tier, by the routing rule."""
        model_name = self.models.schema_model
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
            call = _usage(result.get("raw"), model_name)
            total = ModelCall(
                model=model_name,
                input_tokens=total.input_tokens + call.input_tokens,
                output_tokens=total.output_tokens + call.output_tokens,
            )
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
