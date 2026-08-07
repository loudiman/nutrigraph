"""The fakes the agent turn seam stands on: the database, and the provider.

The provider is faked at the chat-model boundary, one layer below the redaction
wrapper, so every test runs the real wrapper and can read exactly what Google
would have seen.
"""

from __future__ import annotations

import json
import random
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel

from nutrigraph_agent.db import (
    DayTotal,
    InteractionEvent,
    LocalFood,
    MealItemRow,
    RetrievedChunk,
    UnmatchedItem,
)
from nutrigraph_agent.food import CANDIDATES, COLUMNS, FoodCandidate, candidate
from nutrigraph_agent.models import UPDATABLE_FIELDS, Profile, RouterDecision
from nutrigraph_agent.providers import Models
from nutrigraph_agent.seed import DISHES_FILE, aliases_for

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
class StoredMeal:
    meal_id: UUID
    user_id: str
    turn_id: UUID
    eaten_at: datetime
    meal_type: str


@dataclass
class StoredItem:
    meal_item_id: UUID
    meal_id: UUID
    user_id: str
    eaten_at: datetime
    row: MealItemRow


def local_table() -> dict[str, LocalFood]:
    """The real dish table, keyed exactly as the loader keys it.

    The seed file is the fixture, so a seam test reads the values the Coach will
    actually give a User — the proxy rows, the calculated rows, and the nulls
    among them — rather than numbers invented for a test.
    """
    dishes = json.loads(DISHES_FILE.read_text(encoding="utf-8"))["dishes"]
    table: dict[str, LocalFood] = {}
    for dish in dishes:
        food = LocalFood(
            local_food_id=uuid4(),
            name=dish["name"],
            value_kind=dish["value_kind"],
            source_note=dish["source_note"],
            per_100g={c: float(dish[c]) for c in COLUMNS if dish.get(c) is not None},
            source=dish,
        )
        for alias in aliases_for(dish):
            table[alias] = food
    return table


# One `foods[]` entry as FoodData Central returns it, so the nutrient-number
# mapping is exercised rather than assumed. An egg is not in the local dish
# table, so this is the food that takes the search-then-choose path.
EGG_FOOD = {
    "fdcId": 748967,
    "description": "Egg, whole, raw, fresh",
    "dataType": "Foundation",
    "foodNutrients": [
        {"nutrientNumber": "208", "nutrientName": "Energy", "value": 143.0},
        {"nutrientNumber": "203", "nutrientName": "Protein", "value": 12.6},
        {"nutrientNumber": "204", "nutrientName": "Total lipid (fat)", "value": 9.51},
        {"nutrientNumber": "205", "nutrientName": "Carbohydrate", "value": 0.72},
        {"nutrientNumber": "291", "nutrientName": "Fiber", "value": 0.0},
        {"nutrientNumber": "307", "nutrientName": "Sodium", "value": 142.0},
    ],
}
EGG = candidate(EGG_FOOD)


@dataclass
class FakeFoodSearch:
    """FoodData Central, faked at the web request. `results` is keyed by the
    food word; anything not in it is a food FoodData Central does not know."""

    results: dict[str, list[FoodCandidate]] = field(default_factory=dict)
    searched: list[str] = field(default_factory=list)

    async def search(
        self, query: str, *, limit: int = CANDIDATES
    ) -> list[FoodCandidate]:
        self.searched.append(query)
        return self.results.get(query.lower(), [])[:limit]


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
    # The food log. The dish table is the real seed file.
    local_foods: dict[str, LocalFood] = field(default_factory=local_table)
    looked_up: list[str] = field(default_factory=list)
    meals: list[StoredMeal] = field(default_factory=list)
    items: list[StoredItem] = field(default_factory=list)

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

    # --- the food log ---------------------------------------------------------

    async def match_local_food(self, name: str) -> LocalFood | None:
        self.looked_up.append(name)
        if name in self.local_foods:
            return self.local_foods[name]
        # The prefix half of the real query, shortest alias first.
        near = sorted((a for a in self.local_foods if a.startswith(name)), key=len)
        return self.local_foods[near[0]] if near else None

    async def store_meal(
        self,
        *,
        user_id: str,
        turn_id: UUID,
        eaten_at: datetime,
        meal_type: str,
        items: Sequence[MealItemRow],
    ) -> UUID:
        meal = StoredMeal(uuid4(), user_id, turn_id, eaten_at, meal_type)
        self.meals.append(meal)
        self.items += [
            StoredItem(uuid4(), meal.meal_id, user_id, eaten_at, row) for row in items
        ]
        return meal.meal_id

    async def open_unmatched_items(
        self, user_id: str, *, since: datetime
    ) -> list[UnmatchedItem]:
        return [
            UnmatchedItem(item.meal_item_id, item.row.said_as)
            for item in self.items
            if item.user_id == user_id
            and item.row.status == "unmatched"
            and item.eaten_at >= since
        ]

    async def correct_meal_item(self, meal_item_id: UUID, item: MealItemRow) -> None:
        stored = next(i for i in self.items if i.meal_item_id == meal_item_id)
        # The Meal it belongs to does not move, so the day it counts against is
        # still the day it was eaten on.
        stored.row = item

    async def day_total(
        self, user_id: str, *, start: datetime, end: datetime
    ) -> DayTotal:
        rows = [
            i.row
            for i in self.items
            if i.user_id == user_id and start <= i.eaten_at < end
        ]
        matched = [r for r in rows if r.status == "matched"]
        return DayTotal(
            counted=len(matched),
            not_counted=len(rows) - len(matched),
            values={
                c: sum(r.values[c] for r in matched if c in r.values)
                for c in COLUMNS
                if any(c in r.values for r in matched)
            },
            missing={
                c: sum(1 for r in matched if c not in r.values) for c in COLUMNS
            },
        )


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
    # An Intent with no path of its own, so an unscripted Turn ends at the stub
    # and makes exactly one call. A scripted Turn names the Intent it wants.
    default: RouterDecision = field(
        default_factory=lambda: RouterDecision(intents=["review_day"], confidence=0.92)
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
