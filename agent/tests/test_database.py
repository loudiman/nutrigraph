"""The migration runner, the seed command, and the checkpointer's schema.

These run against a real PostgreSQL. Set NUTRIGRAPH_TEST_DATABASE_URL to a
database you are happy to have emptied.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from nutrigraph_agent.app import open_checkpointer
from nutrigraph_agent.config import CHECKPOINT_SCHEMA
from nutrigraph_agent.graph import build_graph
from nutrigraph_agent.migrate import MIGRATIONS_DIR, migrate, pending
from nutrigraph_agent.seed import seed_profiles

TABLES = (
    "schema_migration",
    "user_profile",
    "message",
    "interaction_event",
    "redaction_placeholder",
)


@pytest.fixture
def empty_database(integration_database_url: str) -> str:
    with psycopg.connect(integration_database_url, autocommit=True) as conn:
        conn.execute("drop schema if exists public cascade")
        conn.execute("create schema public")
        conn.execute(f"drop schema if exists {CHECKPOINT_SCHEMA} cascade")
    return integration_database_url


def test_no_migration_file_references_the_langgraph_schema():
    for path in Path(MIGRATIONS_DIR).glob("*.sql"):
        assert CHECKPOINT_SCHEMA not in path.read_text(encoding="utf-8").lower()


def test_the_first_migration_creates_the_version_table_and_the_two_tables(empty_database):
    applied = migrate(empty_database)

    assert applied == ["001_init.sql", "002_router.sql"]
    with psycopg.connect(empty_database) as conn:
        for table in TABLES:
            assert conn.execute("select to_regclass(%s)", (f"public.{table}",)).fetchone()[0]
        recorded = conn.execute("select filename from schema_migration").fetchall()
    assert recorded == [("001_init.sql",), ("002_router.sql",)]


def test_re_running_the_migration_applies_nothing_and_fails_nothing(empty_database):
    migrate(empty_database)

    assert migrate(empty_database) == []
    with psycopg.connect(empty_database) as conn:
        assert pending(conn) == []


def test_the_seed_is_safe_to_run_twice(empty_database):
    migrate(empty_database)

    seeded = seed_profiles(empty_database)
    seed_profiles(empty_database)

    with psycopg.connect(empty_database) as conn:
        count = conn.execute("select count(*) from user_profile").fetchone()[0]
    assert count == len(seeded)


async def test_the_checkpointer_writes_to_the_langgraph_schema(empty_database):
    migrate(empty_database)
    seed_profiles(empty_database)
    saver, pool = await open_checkpointer(empty_database)
    try:
        graph = build_graph(saver)
        await graph.aupdate_state(
            {"configurable": {"thread_id": "demo-user-1"}},
            {"user_id": "demo-user-1", "messages": [], "pending_clarification": None},
        )
    finally:
        await pool.close()

    with psycopg.connect(empty_database) as conn:
        rows = conn.execute(
            "select table_name from information_schema.tables where table_schema = %s",
            (CHECKPOINT_SCHEMA,),
        ).fetchall()
        public_tables = {
            r[0]
            for r in conn.execute(
                "select table_name from information_schema.tables "
                "where table_schema = 'public'"
            ).fetchall()
        }
    assert "checkpoints" in {r[0] for r in rows}
    assert public_tables == set(TABLES)
