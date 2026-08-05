"""The 90-day purge, and the warning ADR 0002 rests on.

The purge runs against a real PostgreSQL: it is one SQL statement, and faking a
database would only test the fake. Set NUTRIGRAPH_TEST_DATABASE_URL to a
database you are happy to have emptied.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from nutrigraph_agent.migrate import migrate
from nutrigraph_agent.retention import DEMO_WARNING, RETENTION_DAYS, purge_raw_text
from nutrigraph_agent.seed import seed_profiles

from .test_database import empty_database  # noqa: F401 - a fixture, used by name

REPO = Path(__file__).resolve().parents[2]

USER = "demo-user-1"


def message(conn: psycopg.Connection, *, text: str, age_days: int) -> str:
    """One `message` row, written `age_days` ago."""
    return conn.execute(
        "insert into message (user_id, turn_id, role, raw_text, created_at) "
        "values (%s, %s, 'user', %s, now() - make_interval(days => %s)) "
        "returning message_id",
        (USER, str(uuid4()), text, age_days),
    ).fetchone()[0]


def rows(conn: psycopg.Connection, table: str) -> list[tuple]:
    return conn.execute(f"select * from {table} order by 1").fetchall()


@pytest.fixture
def database(empty_database: str) -> str:  # noqa: F811 - the imported fixture
    migrate(empty_database)
    seed_profiles(empty_database)
    return empty_database


def test_raw_text_older_than_ninety_days_is_nulled_and_stamped(database):
    with psycopg.connect(database, autocommit=True) as conn:
        old = message(conn, text="I ate two eggs", age_days=RETENTION_DAYS + 1)
        recent = message(conn, text="I ate pandesal", age_days=RETENTION_DAYS - 1)

    assert purge_raw_text(database) == 1

    with psycopg.connect(database) as conn:
        purged = conn.execute(
            "select raw_text, purged_at from message where message_id = %s", (old,)
        ).fetchone()
        kept = conn.execute(
            "select raw_text, purged_at from message where message_id = %s", (recent,)
        ).fetchone()
    assert purged[0] is None and purged[1] is not None
    assert kept == ("I ate pandesal", None)


def test_a_purged_row_keeps_its_identifiers_its_role_and_its_timestamps(database):
    with psycopg.connect(database, autocommit=True) as conn:
        before = conn.execute(
            "insert into message (user_id, turn_id, role, raw_text, created_at) "
            "values (%s, %s, 'user', 'I ate two eggs', now() - make_interval(days => %s)) "
            "returning message_id, user_id, turn_id, role, created_at",
            (USER, str(uuid4()), RETENTION_DAYS + 1),
        ).fetchone()

    purge_raw_text(database)

    with psycopg.connect(database) as conn:
        after = conn.execute(
            "select message_id, user_id, turn_id, role, created_at from message"
        ).fetchone()
    assert after == before


def test_the_job_is_safe_to_run_twice(database):
    with psycopg.connect(database, autocommit=True) as conn:
        message(conn, text="I ate two eggs", age_days=RETENTION_DAYS + 1)
    purge_raw_text(database)
    with psycopg.connect(database) as conn:
        first = rows(conn, "message")

    assert purge_raw_text(database) == 0

    with psycopg.connect(database) as conn:
        assert rows(conn, "message") == first


def test_the_purge_writes_two_columns_of_one_table_and_nothing_else(database):
    """Meals, Items, Recommendations and `interaction_event` rows are untouched.
    Read from `information_schema`, so a table a later slice adds is covered by
    this test the day it is created, without anyone remembering to come back."""
    turn = uuid4()
    with psycopg.connect(database, autocommit=True) as conn:
        message(conn, text="I ate two eggs", age_days=RETENTION_DAYS + 1)
        conn.execute(
            "insert into interaction_event (turn_id, user_id, node, latency_ms) "
            "values (%s, %s, 'route', 12)",
            (str(turn), USER),
        )
        conn.execute(
            "insert into redaction_placeholder (turn_id, placeholder, original) "
            "values (%s, '[NAME_1]', 'Lou')",
            (str(turn),),
        )
        tables = [
            r[0]
            for r in conn.execute(
                "select table_name from information_schema.tables "
                "where table_schema = 'public' and table_name <> 'message'"
            ).fetchall()
        ]
        before = {table: rows(conn, table) for table in tables}

    purge_raw_text(database)

    with psycopg.connect(database) as conn:
        assert {table: rows(conn, table) for table in tables} == before


def test_a_day_review_over_a_purged_period_still_reports_correct_totals(database):
    """The day review sums the structured Items, never the free text, so nulling
    the free text cannot change a total.

    ponytail: `meal` and `meal_item` are created here with the columns the
    specification fixes, because the log_meal slice (#33) has not landed. When it
    does, `create table if not exists` finds the real tables and this test reads
    them instead — the assertion is the same either way.
    """
    with psycopg.connect(database, autocommit=True) as conn:
        conn.execute(
            "create table if not exists meal ("
            "  meal_id uuid primary key default gen_random_uuid(),"
            "  user_id text not null references user_profile (user_id),"
            "  logged_at timestamptz not null default now(),"
            "  meal_type text,"
            "  message_id uuid references message (message_id))"
        )
        conn.execute(
            "create table if not exists meal_item ("
            "  item_id uuid primary key default gen_random_uuid(),"
            "  meal_id uuid not null references meal (meal_id),"
            "  name_as_said text not null,"
            "  kcal numeric,"
            "  protein_g numeric)"
        )
        said = message(conn, text="I ate two eggs and pandesal", age_days=RETENTION_DAYS + 1)
        meal = conn.execute(
            "insert into meal (user_id, logged_at, meal_type, message_id) "
            "values (%s, now() - make_interval(days => %s), 'breakfast', %s) "
            "returning meal_id",
            (USER, RETENTION_DAYS + 1, said),
        ).fetchone()[0]
        conn.execute(
            "insert into meal_item (meal_id, name_as_said, kcal, protein_g) values "
            "(%s, 'two eggs', 143, 12.6), (%s, 'pandesal', 147, 4.2)",
            (meal, meal),
        )

    purge_raw_text(database)

    with psycopg.connect(database) as conn:
        totals = conn.execute(
            "select sum(i.kcal), sum(i.protein_g) from meal_item i "
            "join meal m on m.meal_id = i.meal_id "
            "where m.user_id = %s and m.logged_at::date = "
            "      (now() - make_interval(days => %s))::date",
            (USER, RETENTION_DAYS + 1),
        ).fetchone()
        raw_text = conn.execute("select raw_text from message").fetchone()[0]
    assert totals == (Decimal("290"), Decimal("16.8"))
    assert raw_text is None, "the free text the totals were parsed from is gone"


def test_the_demo_warning_is_where_a_developer_seeds_or_connects():
    """One sentence, in every place it has to be, and no copy drifting from
    another. ADR 0002 rests on this warning rather than on the schema."""
    from nutrigraph_agent import migrate as migrate_module
    from nutrigraph_agent import seed as seed_module

    assert migrate_module.DEMO_WARNING is DEMO_WARNING
    assert seed_module.DEMO_WARNING is DEMO_WARNING

    for path in (REPO / "README.md", REPO / "gateway" / "public" / "index.html"):
        text = " ".join(path.read_text(encoding="utf-8").split())
        assert DEMO_WARNING in text, f"the warning is not in {path.name}"
