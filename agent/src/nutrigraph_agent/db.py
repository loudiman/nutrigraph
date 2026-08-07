"""The database seam. `Database` is what a node is allowed to know about
PostgreSQL; the turn seam swaps a fake in for it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
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


# --- the lookup cache ---------------------------------------------------------
#
# Two kinds, and both are independent of the User: a Corpus retrieval keyed on
# the question's embedding, and the food a parsed name matched. Nothing that
# read the Profile or today's Meals is written here, and no whole answer is
# either. These four statements are the only writers and readers there are.

# A hit needs this much cosine similarity. Below it the questions are two
# questions, and the second one searches the Corpus for itself.
RETRIEVAL_SIMILARITY = 0.95

# pgvector stores a `float4`, so a vector written at exactly the floor reads
# back a fraction of a millionth either side of it and an exact `>= 0.95` would
# turn away a question that is, to every digit anyone means by it, at the floor.
# The comparison allows the rounding and nothing more.
FLOAT4_TOLERANCE = 1e-6

# How long a food match stands. The FoodData Central catalogue moves slowly,
# and a month-old choice of `fdcId` for 'pandesal' is still the right choice.
FOOD_MATCH_DAYS = 30

# What the Corpus looks like now. A retrieval entry written against an older
# Corpus is not read, so a re-ingest invalidates every one of them without a
# row having to be found first — the ingest deletes them too, but this is what
# makes the invalidation true even between the two statements.
CORPUS_VERSION = "select coalesce(max(ingested_at)::text, '') from corpus_document"

# `update ... returning` rather than a select and then an update: one round
# trip, and `hits` cannot drift from the number of times the entry was served.
READ_RETRIEVAL = f"""
update lookup_cache set hits = hits + 1
where lookup_cache_id = (
    select lookup_cache_id
      from lookup_cache
     where kind = 'retrieval'
       and corpus_version = ({CORPUS_VERSION})
       and 1 - (key_embedding <=> %(query)s::vector) >= %(floor)s
     order by key_embedding <=> %(query)s::vector
     limit 1
)
returning value
"""

WRITE_RETRIEVAL = f"""
insert into lookup_cache (kind, key_text, key_embedding, value, corpus_version)
values ('retrieval', %(key_text)s, %(embedding)s::vector, %(value)s, ({CORPUS_VERSION}))
"""

READ_FOOD_MATCH = """
update lookup_cache set hits = hits + 1
where kind = 'food_match'
  and key_text = %(name)s
  and created_at > now() - make_interval(days => %(days)s)
returning value
"""

# An expired row is replaced rather than left to block the key, so a food whose
# entry has aged out is looked up again and written back under the same key.
WRITE_FOOD_MATCH = """
insert into lookup_cache (kind, key_text, value) values ('food_match', %(name)s, %(value)s)
on conflict (key_text) where kind = 'food_match'
do update set value = excluded.value, created_at = now(), hits = 0
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
       array_agg(mi.said_as order by m.eaten_at, mi.ordinal)
           filter (where mi.status = 'unmatched') as unmatched,
       {", ".join(f"sum(mi.{c}) as {c}" for c in COLUMNS)},
       {", ".join(
           f"count(*) filter (where mi.status = 'matched' and mi.{c} is null) "
           f"as {c}_missing" for c in COLUMNS
       )}
from meal_item mi join meal m using (meal_id)
where m.user_id = %(user_id)s and m.eaten_at >= %(start)s and m.eaten_at < %(end)s
"""


# --- the recommend path -------------------------------------------------------
#
# Code finds; the model ranks. Everything below runs before any provider call on
# this path, so the surviving rows are the only foods a model is ever shown.

# The vector of the foods this User actually ate or accepted. `avg` over
# unit-length vectors is the centroid of their taste, and cosine distance
# ignores magnitude, so it needs no re-normalizing. A User with neither returns
# one row holding null, which is the cold start: the ordering then rests on the
# gap alone.
TASTE = """
select avg(e.embedding) as v
  from food_embedding e
 where exists (
           select 1
             from meal_item mi join meal m using (meal_id)
            where m.user_id = %(user_id)s
              and mi.status = 'matched'
              and ((e.source = 'local' and e.source_id = mi.local_food_id::text)
                or (e.source = 'fdc' and e.source_id = mi.fdc_id))
       )
    or exists (
           select 1
             from recommendation r
            where r.user_id = %(user_id)s and r.accepted
              and lower(e.name) in (select lower(f) from unnest(r.foods) f)
       )
"""

# The two sources of candidates: the local dish table, and the FoodData Central
# foods this User has already logged. A logged Item holds the values for the
# portion that was eaten, so `grams` puts it back on the per-100 g basis both
# sources are read on.
CANDIDATE_ROWS = f"""
select 'local' as source, f.local_food_id::text as source_id, f.name,
       f.tags, f.value_kind, f.source_note, null::text as category,
       {", ".join(f"f.{c}::float8 as {c}" for c in COLUMNS)}
  from local_food f
union all
select * from (
    select distinct on (lower(mi.food_name))
           'fdc' as source, mi.fdc_id as source_id, mi.food_name as name,
           '{{}}'::text[] as tags, mi.value_kind, null::text as source_note,
           -- A FoodData Central item carries no tags, so its source category is
           -- what the diet-pattern filter reads instead.
           mi.nutrients->>'foodCategory' as category,
           {", ".join(f"(mi.{c} * 100.0 / mi.grams)::float8 as {c}" for c in COLUMNS)}
      from meal_item mi join meal m using (meal_id)
     where m.user_id = %(user_id)s and mi.status = 'matched'
       and mi.source = 'fdc' and mi.food_name is not null
       and mi.grams is not null and mi.grams > 0
     order by lower(mi.food_name), m.eaten_at desc
) logged
"""

# **The hard filters, in SQL and not in a prompt.**
#
# An allergen or a disliked food is struck on the name, on the dish table's tags
# and on the FoodData Central category, because any of the three may be the only
# structured place that knows: a User allergic to peanut who is offered
# kare-kare is offered a dish whose name says no peanut at all.
#
# A diet-pattern conflict is struck on the tags and the category alone, never on
# the name. 'egg' inside 'eggplant' is why: a substring match on the name would
# take a vegetable off a vegan's list. An allergy is filtered the other way
# round on purpose — there the false positive is the safe direction.
FILTERS = """
   and not exists (
           select 1 from unnest(%(blocked)s::text[]) w
            where c.name ilike ('%%' || w || '%%')
               or w = any (c.tags)
               or coalesce(c.category, '') ilike ('%%' || w || '%%')
       )
   and not (c.tags && %(conflicts)s::text[])
   and not exists (
           select 1 from unnest(%(conflicts)s::text[]) w
            where coalesce(c.category, '') ilike ('%%' || w || '%%')
       )
"""

# How much of the largest gap a 100 g serving of this food closes, capped at
# one whole gap, plus how near it is to what this User already eats. The two
# are added rather than ranked one after the other: a strict ordering on the
# gap would make the similarity a tie-break that continuous numbers never
# reach, and the personalisation would be decoration.
#
# ponytail: one weight each. A weight worth tuning is one the eval can measure,
# and the eval set is issue #39.
SCORED = "least(coalesce({column}, 0) / %(gap)s, 1.0) + coalesce(similarity, 0)"

# Nothing is short today, or no target could be worked out. The nearest food to
# what this User eats, and the lightest of those, which is the honest ordering
# when there is no gap to close.
UNSCORED = "coalesce(similarity, 0) desc, kcal asc nulls last"


def candidate_query(nutrient: str | None) -> str:
    """The candidate query, ordered by the nutrient the gap is largest on.

    A column name cannot be a query parameter, so it is whitelisted against the
    same six columns every other statement here is built from.
    """
    if nutrient is not None and nutrient not in COLUMNS:
        raise ValueError(f"{nutrient!r} is not a nutrient column")
    order = f"({SCORED.format(column=nutrient)}) desc" if nutrient else UNSCORED
    # The similarity is named in its own step, because a select-list alias
    # cannot be used inside an expression in the same statement's `order by`.
    return f"""
with taste as ({TASTE}), candidate as ({CANDIDATE_ROWS}),
surviving as (
    select c.*,
           case when t.v is null then null
                else 1 - (e.embedding <=> t.v) end as similarity
      from candidate c
      cross join taste t
      left join food_embedding e
             on e.source = c.source and e.source_id = c.source_id
     where true {FILTERS}
)
select * from surviving
 order by {order}
 limit %(limit)s
"""


INSERT_RECOMMENDATION = """
insert into recommendation
    (user_id, turn_id, gap_nutrient, gap_amount, suggestion, reason, foods)
values (%(user_id)s, %(turn_id)s, %(gap_nutrient)s, %(gap_amount)s,
        %(suggestion)s, %(reason)s, %(foods)s)
returning recommendation_id
"""

# The acceptance signal: null becomes true or false, once. `where accepted is
# null` is what makes a second answer to the same suggestion a no-op rather than
# a rewritten history.
ANSWER_RECOMMENDATION = """
update recommendation set accepted = %(accepted)s, responded_at = now()
 where recommendation_id = %(recommendation_id)s and accepted is null
returning recommendation_id
"""

# **Both measurement signals, and no new column.** `accepted` is the first.
# The second is a Meal holding one of the recommended foods inside the window,
# which is this join over `recommendation.foods` and `meal_item` — acceptance
# alone cannot tell a polite yes from a real change, which is why it is here.
RECOMMENDATION_OUTCOMES = """
select r.recommendation_id, r.created_at, r.foods, r.accepted, r.gap_nutrient,
       exists (
           select 1
             from meal_item mi join meal m using (meal_id)
            where m.user_id = r.user_id
              and m.eaten_at >= r.created_at
              and m.eaten_at < r.created_at + %(within)s
              and exists (
                      select 1 from unnest(r.foods) f
                       where lower(mi.food_name) = lower(f)
                          or lower(mi.said_as) = lower(f)
                  )
       ) as followed
  from recommendation r
 where r.user_id = %(user_id)s
 order by r.created_at desc
"""

# One row for each food, so the seed is safe to run twice and a food this system
# has already seen is never embedded a second time.
INSERT_FOOD_EMBEDDING = """
insert into food_embedding (source, source_id, name, embedding)
values (%(source)s, %(source_id)s, %(name)s, %(embedding)s::vector)
on conflict (source, source_id) do nothing
"""

UNEMBEDDED_LOCAL_FOODS = """
select f.local_food_id::text as source_id, f.name
  from local_food f
 where not exists (
           select 1 from food_embedding e
            where e.source = 'local' and e.source_id = f.local_food_id::text
       )
 order by f.name
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
    # The foods that contributed nothing at all, by the words the User used. An
    # unmatched Item is not in the sum, so the review names these and says the
    # total is short by an unknown amount rather than presenting it as complete.
    unmatched: list[str] = field(default_factory=list)

    def complete(self, column: str) -> bool:
        return self.missing.get(column, 0) == 0


@dataclass(frozen=True)
class Candidate:
    """One food that survived the filters, on the per-100 g basis both sources
    are read on.

    `value_kind` and `source_note` travel here for the same reason they travel
    with a match: a proxy row is a stand-in and a calculated row is computed
    rather than measured, and the suggestion has to say so.

    `similarity` is how near this food is to what this User actually eats, and
    it is null when there is nothing to be near — a User with no Meal and no
    accepted suggestion, or a food this system has not embedded yet.
    """

    source: str
    source_id: str
    name: str
    per_100g: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    value_kind: str | None = None
    source_note: str | None = None
    category: str | None = None
    similarity: float | None = None


@dataclass(frozen=True)
class RecommendationOutcome:
    """One suggestion, measured by both signals. `accepted` is null until the
    User says; `followed` is a Meal holding one of the foods inside the window,
    which is a query and not a column."""

    recommendation_id: UUID
    created_at: datetime
    foods: list[str]
    accepted: bool | None
    gap_nutrient: str | None
    followed: bool


# How long a Meal has to appear in for it to count as following the suggestion.
FOLLOWED_WITHIN = timedelta(hours=24)


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

    async def cached_retrieval(
        self, embedding: Sequence[float]
    ) -> list[RetrievedChunk] | None: ...

    async def store_cached_retrieval(
        self, *, key_text: str, embedding: Sequence[float], chunks: Sequence[RetrievedChunk]
    ) -> None: ...

    async def cached_food_match(self, name: str) -> dict[str, Any] | None: ...

    async def store_cached_food_match(self, name: str, value: dict[str, Any]) -> None: ...

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

    async def candidate_foods(
        self,
        user_id: str,
        *,
        blocked: Sequence[str],
        conflicts: Sequence[str],
        nutrient: str | None,
        gap: float,
        limit: int,
    ) -> list[Candidate]: ...

    async def store_recommendation(
        self,
        *,
        user_id: str,
        turn_id: UUID,
        gap_nutrient: str | None,
        gap_amount: float | None,
        suggestion: str,
        reason: str,
        foods: Sequence[str],
    ) -> UUID: ...

    async def answer_recommendation(
        self, recommendation_id: UUID, *, accepted: bool
    ) -> bool: ...

    async def recommendation_outcomes(
        self, user_id: str, *, within: timedelta = FOLLOWED_WITHIN
    ) -> list[RecommendationOutcome]: ...

    async def has_food_embedding(self, *, source: str, source_id: str) -> bool: ...

    async def store_food_embedding(
        self, *, source: str, source_id: str, name: str, embedding: Sequence[float]
    ) -> None: ...


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
    # The Turn was still over the token budget after trimming, ran anyway, and
    # said so here — so the eval reads the overrun rather than inferring it.
    over_budget: bool = False


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
                "latency_ms, input_tokens, output_tokens, cost_usd, over_budget) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    str(event.turn_id), event.user_id, event.node, event.intent,
                    event.model, event.latency_ms, event.input_tokens,
                    event.output_tokens, event.cost_usd, event.over_budget,
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

    # --- the lookup cache -----------------------------------------------------

    async def cached_retrieval(
        self, embedding: Sequence[float]
    ) -> list[RetrievedChunk] | None:
        """The passages a near-enough question already found, or nothing.

        The scores come back as the first question measured them. They are the
        scores of a question 0.95 or more similar, which is what the threshold
        is for, and `RELEVANCE_FLOOR` still reads them.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                READ_RETRIEVAL,
                {
                    "query": vector_literal(embedding),
                    "floor": RETRIEVAL_SIMILARITY - FLOAT4_TOLERANCE,
                },
            )
            row = await cur.fetchone()
        return [RetrievedChunk(**chunk) for chunk in row[0]] if row else None

    async def store_cached_retrieval(
        self, *, key_text: str, embedding: Sequence[float], chunks: Sequence[RetrievedChunk]
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                WRITE_RETRIEVAL,
                {
                    "key_text": key_text,
                    "embedding": vector_literal(embedding),
                    "value": Jsonb([asdict(chunk) for chunk in chunks]),
                },
            )

    async def cached_food_match(self, name: str) -> dict[str, Any] | None:
        """The food a name matched last time, if the entry has not expired."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                READ_FOOD_MATCH, {"name": name, "days": FOOD_MATCH_DAYS}
            )
            row = await cur.fetchone()
        return row[0] if row else None

    async def store_cached_food_match(self, name: str, value: dict[str, Any]) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(WRITE_FOOD_MATCH, {"name": name, "value": Jsonb(value)})

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
            # `array_agg` over no rows is null, not an empty array.
            unmatched=list(row["unmatched"] or []),
        )

    # --- the recommend path ---------------------------------------------------

    async def candidate_foods(
        self,
        user_id: str,
        *,
        blocked: Sequence[str],
        conflicts: Sequence[str],
        nutrient: str | None,
        gap: float,
        limit: int,
    ) -> list[Candidate]:
        """The foods this User may be offered, best first. One query, and every
        filter inside it, so no allergen and no disliked food is ever in the
        list a model is shown."""
        async with self._pool.connection() as conn:
            cur = await conn.cursor(row_factory=dict_row).execute(
                candidate_query(nutrient),
                {
                    "user_id": user_id,
                    "blocked": [w.strip().lower() for w in blocked if w.strip()],
                    "conflicts": [w.strip().lower() for w in conflicts if w.strip()],
                    # Never zero: `candidate_query` only names the gap when there
                    # is one to close, and this divides by it.
                    "gap": gap or 1.0,
                    "limit": limit,
                },
            )
            rows = await cur.fetchall()
        return [
            Candidate(
                source=row["source"],
                source_id=row["source_id"],
                name=row["name"],
                per_100g=_floats(row),
                tags=list(row["tags"] or []),
                value_kind=row["value_kind"],
                source_note=row["source_note"],
                category=row["category"],
                similarity=(
                    None if row["similarity"] is None else float(row["similarity"])
                ),
            )
            for row in rows
        ]

    async def store_recommendation(
        self,
        *,
        user_id: str,
        turn_id: UUID,
        gap_nutrient: str | None,
        gap_amount: float | None,
        suggestion: str,
        reason: str,
        foods: Sequence[str],
    ) -> UUID:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                INSERT_RECOMMENDATION,
                {
                    "user_id": user_id,
                    "turn_id": str(turn_id),
                    "gap_nutrient": gap_nutrient,
                    "gap_amount": gap_amount,
                    "suggestion": suggestion,
                    "reason": reason,
                    "foods": list(foods),
                },
            )
            return (await cur.fetchone())[0]

    async def answer_recommendation(
        self, recommendation_id: UUID, *, accepted: bool
    ) -> bool:
        """The acceptance signal. False when there is no such suggestion, or
        when it was already answered — a second answer does not rewrite the
        first."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                ANSWER_RECOMMENDATION,
                {"recommendation_id": str(recommendation_id), "accepted": accepted},
            )
            return await cur.fetchone() is not None

    async def recommendation_outcomes(
        self, user_id: str, *, within: timedelta = FOLLOWED_WITHIN
    ) -> list[RecommendationOutcome]:
        async with self._pool.connection() as conn:
            cur = await conn.cursor(row_factory=dict_row).execute(
                RECOMMENDATION_OUTCOMES, {"user_id": user_id, "within": within}
            )
            rows = await cur.fetchall()
        return [
            RecommendationOutcome(
                recommendation_id=row["recommendation_id"],
                created_at=row["created_at"],
                foods=list(row["foods"] or []),
                accepted=row["accepted"],
                gap_nutrient=row["gap_nutrient"],
                followed=row["followed"],
            )
            for row in rows
        ]

    async def has_food_embedding(self, *, source: str, source_id: str) -> bool:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select 1 from food_embedding where source = %s and source_id = %s",
                (source, source_id),
            )
            return await cur.fetchone() is not None

    async def store_food_embedding(
        self, *, source: str, source_id: str, name: str, embedding: Sequence[float]
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                INSERT_FOOD_EMBEDDING,
                {
                    "source": source,
                    "source_id": source_id,
                    "name": name,
                    "embedding": vector_literal(embedding),
                },
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
