"""The database seam. `Database` is what a node is allowed to know about
PostgreSQL; the turn seam swaps a fake in for it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

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
