"""The database seam. `Database` is what a node is allowed to know about
PostgreSQL; the turn seam swaps a fake in for it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .food import COLUMNS
from .models import UPDATABLE_FIELDS, Profile

PROFILE_COLUMNS = """
    user_id, name, sex, age, height_cm, weight_kg, target_weight_kg,
    activity_level, diet_pattern, units, allergies, disliked_foods
"""

# One predicate excludes every non-commercial chunk, and it needs no join,
# because `commercial_use` is written onto the chunk row at ingestion time
# beside the licence identifier and the attribution string. A commercial review
# runs `delete from corpus_chunk where not commercial_use` and is finished.
COMMERCIAL_ONLY = "commercial_use"

SEARCH_CORPUS = f"""
select d.title as document, d.source_url, c.locator, c.text,
       c.licence_id, c.attribution, c.{COMMERCIAL_ONLY} as commercial_use,
       1 - (c.embedding <=> %(query)s::vector) as score
from corpus_chunk c join corpus_document d using (document_id)
order by c.embedding <=> %(query)s::vector
limit %(limit)s
"""


LOCAL_FOOD_COLUMNS = f"""
    f.local_food_id, f.name, f.value_kind, f.source_note, f.source,
    {", ".join("f." + c for c in COLUMNS)}
"""

# The local table, first and cheap. An exact alias wins at once; a prefix is the
# fallback, and `text_pattern_ops` is what lets it use the index under any
# collation. One indexed query, which is what makes it affordable to run before
# every FoodData Central call.
MATCH_LOCAL_FOOD = f"""
select {LOCAL_FOOD_COLUMNS}
from local_food_alias a join local_food f using (local_food_id)
where a.alias = %(exact)s or a.alias like %(prefix)s
order by (a.alias = %(exact)s) desc, length(a.alias)
limit 1
"""

MEAL_ITEM_COLUMNS = (
    "meal_id", "ordinal", "said_as", "quantity", "unit", "grams",
    "portion_assumed", "status", "source", "local_food_id", "fdc_id",
    "food_name", "value_kind", "match_note", "nutrients", *COLUMNS,
)

INSERT_MEAL_ITEM = f"""
insert into meal_item ({", ".join(MEAL_ITEM_COLUMNS)})
values ({", ".join("%s" for _ in MEAL_ITEM_COLUMNS)})
"""

# A correction fills in the Item that was already written. The Meal it belongs
# to does not move, so the day it was eaten on stays the day it counts against.
CORRECT_MEAL_ITEM = f"""
update meal_item set
    said_as = %s, quantity = %s, unit = %s, grams = %s, portion_assumed = %s,
    status = %s, source = %s, local_food_id = %s, fdc_id = %s, food_name = %s,
    value_kind = %s, match_note = %s, nutrients = %s,
    {", ".join(f"{c} = %s" for c in COLUMNS)},
    updated_at = now()
where meal_item_id = %s
"""

OPEN_UNMATCHED = """
select mi.meal_item_id, mi.said_as
from meal_item mi join meal m using (meal_id)
where m.user_id = %s and mi.status = 'unmatched' and mi.created_at >= %s
order by mi.created_at
"""

# A day total is one SQL sum over the six numeric columns. The `filter` counters
# beside it are what stop the total reading as complete when a null contributed:
# a column PostgreSQL sums over nulls, and the count says how many Items had
# nothing there to add.
DAY_TOTAL = f"""
select count(*) filter (where mi.status = 'matched') as counted,
       count(*) filter (where mi.status = 'unmatched') as not_counted,
       {", ".join(f"sum(mi.{c}) as {c}" for c in COLUMNS)},
       {", ".join(
           f"count(*) filter (where mi.status = 'matched' and mi.{c} is null) "
           f"as {c}_missing" for c in COLUMNS
       )}
from meal_item mi join meal m using (meal_id)
where m.user_id = %(user_id)s and m.eaten_at >= %(start)s and m.eaten_at < %(end)s
"""


def vector_literal(values: Sequence[float]) -> str:
    """pgvector's text input. No adapter to register, and no dependency beyond
    psycopg — the cast in the query does the rest."""
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


@dataclass(frozen=True)
class RetrievedChunk:
    """One Corpus chunk, holding both halves of a Citation and the licence terms
    it was ingested under."""

    document: str
    source_url: str
    locator: str
    text: str
    licence_id: str
    attribution: str
    commercial_use: bool
    score: float


def _floats(row: dict[str, Any]) -> dict[str, float]:
    """The six numeric columns of one row, as plain numbers.

    A column PostgreSQL hands back as null is left out rather than turned into
    a zero, and it stays left out all the way to the next row it is written to.
    Fibre and sodium are null wherever the source prints nothing, and a null is
    the truth there.
    """
    return {c: float(row[c]) for c in COLUMNS if row.get(c) is not None}


@dataclass(frozen=True)
class LocalFood:
    """One row of the hand-entered Filipino dish table.

    `value_kind` and `source_note` travel with every match because the Coach has
    to say what the numbers are: a `proxy` row is a canned or commercial product
    standing in for the home-cooked dish, and a `calculated` row is computed
    from components with a stated understatement.
    """

    local_food_id: UUID
    name: str
    value_kind: str
    source_note: str
    per_100g: dict[str, float] = field(default_factory=dict)
    # The whole seed row, for `meal_item.nutrients`.
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class MealItemRow:
    """One Item on its way into PostgreSQL.

    An unmatched Item carries `said_as` and nothing else, which is the point:
    nothing the User said is lost, and the Meal is written either way.
    """

    ordinal: int
    said_as: str
    status: str
    quantity: float | None = None
    unit: str | None = None
    grams: float | None = None
    portion_assumed: bool = False
    source: str | None = None
    local_food_id: UUID | None = None
    fdc_id: str | None = None
    food_name: str | None = None
    value_kind: str | None = None
    match_note: str | None = None
    nutrients: dict[str, Any] | None = None
    # The six, scaled to `grams`. A nutrient the source did not state is absent
    # here and null in the column.
    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class UnmatchedItem:
    """An Item the Coach could not count, which a later message may correct."""

    meal_item_id: UUID
    said_as: str


@dataclass(frozen=True)
class DayTotal:
    """One day, summed.

    `missing` is what stops the total reading as complete when it is not: it
    counts the counted Items that carried no value for each column, so a day
    holding one dish whose sodium the source does not print says so instead of
    reporting a sodium total that silently omits it.
    """

    counted: int = 0
    not_counted: int = 0
    values: dict[str, float] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)

    def complete(self, column: str) -> bool:
        return self.missing.get(column, 0) == 0


class Database(Protocol):
    async def load_profile(self, user_id: str) -> Profile | None: ...

    async def update_profile(self, user_id: str, *, field: str, value: Any) -> None: ...

    async def store_message(
        self, *, user_id: str, turn_id: UUID, role: str, raw_text: str
    ) -> None: ...

    async def store_interaction_event(self, event: InteractionEvent) -> None: ...

    async def store_redaction_map(
        self, *, turn_id: UUID, mapping: dict[str, str]
    ) -> None: ...

    async def search_corpus(
        self, embedding: Sequence[float], *, limit: int = 5
    ) -> list[RetrievedChunk]: ...

    async def match_local_food(self, name: str) -> LocalFood | None: ...

    async def store_meal(
        self,
        *,
        user_id: str,
        turn_id: UUID,
        eaten_at: datetime,
        meal_type: str,
        items: Sequence[MealItemRow],
    ) -> UUID: ...

    async def open_unmatched_items(
        self, user_id: str, *, since: datetime
    ) -> list[UnmatchedItem]: ...

    async def correct_meal_item(self, meal_item_id: UUID, item: MealItemRow) -> None: ...

    async def day_total(
        self, user_id: str, *, start: datetime, end: datetime
    ) -> DayTotal: ...


@dataclass(frozen=True)
class InteractionEvent:
    """One node of one Turn, measured. LangSmith is the reading tool; this is
    the record, because the free tier keeps traces for 14 days."""

    turn_id: UUID
    user_id: str
    node: str
    latency_ms: int
    intent: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class PostgresDatabase:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def load_profile(self, user_id: str) -> Profile | None:
        async with self._pool.connection() as conn:
            cur = await conn.cursor(row_factory=dict_row).execute(
                f"select {PROFILE_COLUMNS} from user_profile where user_id = %s",
                (user_id,),
            )
            row = await cur.fetchone()
        return Profile.model_validate(row) if row else None

    async def update_profile(self, user_id: str, *, field: str, value: Any) -> None:
        """The Profile lives in PostgreSQL alone, so a change is written here and
        nowhere else. The next Turn reads it back, in this Session or another.

        A column name cannot be a query parameter, so it is whitelisted against
        the same tuple the extractor's schema is built from — a value that did
        not come from there never reaches this string.
        """
        if field not in UPDATABLE_FIELDS:
            raise ValueError(f"{field!r} is not a Profile field a User may change")
        async with self._pool.connection() as conn:
            await conn.execute(
                f"update user_profile set {field} = %s, updated_at = now() "
                f"where user_id = %s",
                (value, user_id),
            )

    async def store_message(
        self, *, user_id: str, turn_id: UUID, role: str, raw_text: str
    ) -> None:
        # The raw text is stored unchanged; redaction happens at the provider
        # call, not at the database write (ADR 0002).
        async with self._pool.connection() as conn:
            await conn.execute(
                "insert into message (user_id, turn_id, role, raw_text) "
                "values (%s, %s, %s, %s)",
                (user_id, str(turn_id), role, raw_text),
            )

    async def store_interaction_event(self, event: InteractionEvent) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "insert into interaction_event (turn_id, user_id, node, intent, model, "
                "latency_ms, input_tokens, output_tokens, cost_usd) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    str(event.turn_id), event.user_id, event.node, event.intent,
                    event.model, event.latency_ms, event.input_tokens,
                    event.output_tokens, event.cost_usd,
                ),
            )

    async def store_redaction_map(
        self, *, turn_id: UUID, mapping: dict[str, str]
    ) -> None:
        """The private table that maps a placeholder back to what the User wrote."""
        if not mapping:
            return
        async with self._pool.connection() as conn:
            await conn.cursor().executemany(
                "insert into redaction_placeholder (turn_id, placeholder, original) "
                "values (%s, %s, %s) on conflict do nothing",
                [(str(turn_id), k, v) for k, v in mapping.items()],
            )

    async def search_corpus(
        self, embedding: Sequence[float], *, limit: int = 5
    ) -> list[RetrievedChunk]:
        """Nearest chunks by cosine distance, through the HNSW index."""
        async with self._pool.connection() as conn:
            cur = await conn.cursor(row_factory=dict_row).execute(
                SEARCH_CORPUS,
                {"query": vector_literal(embedding), "limit": limit},
            )
            rows = await cur.fetchall()
        return [RetrievedChunk(**row) for row in rows]

    # --- the food log ---------------------------------------------------------

    async def match_local_food(self, name: str) -> LocalFood | None:
        """The local table, before FoodData Central is reached for at all."""
        async with self._pool.connection() as conn:
            cur = await conn.cursor(row_factory=dict_row).execute(
                MATCH_LOCAL_FOOD, {"exact": name, "prefix": f"{name}%"}
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return LocalFood(
            local_food_id=row["local_food_id"],
            name=row["name"],
            value_kind=row["value_kind"],
            source_note=row["source_note"],
            per_100g=_floats(row),
            source=row["source"],
        )

    async def store_meal(
        self,
        *,
        user_id: str,
        turn_id: UUID,
        eaten_at: datetime,
        meal_type: str,
        items: Sequence[MealItemRow],
    ) -> UUID:
        """The Meal and its Items, in one transaction. An Item the Coach could
        not match is written with the rest; the Meal is never dropped for it."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "insert into meal (user_id, turn_id, eaten_at, meal_type) "
                "values (%s, %s, %s, %s) returning meal_id",
                (user_id, str(turn_id), eaten_at, meal_type),
            )
            meal_id = (await cur.fetchone())[0]
            await conn.cursor().executemany(
                INSERT_MEAL_ITEM, [(meal_id, *_item_values(item)) for item in items]
            )
        return meal_id

    async def open_unmatched_items(
        self, user_id: str, *, since: datetime
    ) -> list[UnmatchedItem]:
        async with self._pool.connection() as conn:
            cur = await conn.cursor(row_factory=dict_row).execute(
                OPEN_UNMATCHED, (user_id, since)
            )
            rows = await cur.fetchall()
        return [UnmatchedItem(**row) for row in rows]

    async def correct_meal_item(self, meal_item_id: UUID, item: MealItemRow) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                CORRECT_MEAL_ITEM, (*_item_values(item)[1:], str(meal_item_id))
            )

    async def day_total(
        self, user_id: str, *, start: datetime, end: datetime
    ) -> DayTotal:
        async with self._pool.connection() as conn:
            cur = await conn.cursor(row_factory=dict_row).execute(
                DAY_TOTAL, {"user_id": user_id, "start": start, "end": end}
            )
            row = await cur.fetchone()
        return DayTotal(
            counted=row["counted"],
            not_counted=row["not_counted"],
            values=_floats(row),
            missing={c: row[f"{c}_missing"] for c in COLUMNS},
        )


def _item_values(item: MealItemRow) -> tuple[Any, ...]:
    """One Item as its columns, in the order both statements name them. The
    `ordinal` leads, so a correction — which updates an Item already numbered —
    drops it and keeps the rest."""
    return (
        item.ordinal, item.said_as, item.quantity, item.unit, item.grams,
        item.portion_assumed, item.status, item.source,
        str(item.local_food_id) if item.local_food_id else None,
        item.fdc_id, item.food_name, item.value_kind, item.match_note,
        Jsonb(item.nutrients) if item.nutrients is not None else None,
        # A nutrient the source did not state is not in `values`, so it is
        # written as null. Never a zero.
        *(item.values.get(c) for c in COLUMNS),
    )
