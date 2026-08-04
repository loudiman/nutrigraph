"""The database seam. `Database` is what a node is allowed to know about
PostgreSQL; the turn seam swaps a fake in for it."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .models import Profile

PROFILE_COLUMNS = """
    user_id, name, sex, age, height_cm, weight_kg, target_weight_kg,
    activity_level, diet_pattern, units, allergies, disliked_foods
"""


class Database(Protocol):
    async def load_profile(self, user_id: str) -> Profile | None: ...

    async def store_message(
        self, *, user_id: str, turn_id: UUID, role: str, raw_text: str
    ) -> None: ...


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
