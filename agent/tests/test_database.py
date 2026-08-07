"""The migration runner, the seed command, and the checkpointer's schema.

These run against a real PostgreSQL. Set NUTRIGRAPH_TEST_DATABASE_URL to a
database you are happy to have emptied.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from nutrigraph_agent.app import open_checkpointer
from nutrigraph_agent.config import CHECKPOINT_TABLES
from nutrigraph_agent.corpus import CorpusEntry
from nutrigraph_agent.db import COMMERCIAL_ONLY, MealItemRow, PostgresDatabase
from nutrigraph_agent.graph import build_graph
from nutrigraph_agent.ingest import ingest
from nutrigraph_agent.meal import MANILA, day_bounds
from nutrigraph_agent.migrate import MIGRATIONS_DIR, migrate, pending
from nutrigraph_agent.providers import EMBEDDING_DIMENSIONS
from nutrigraph_agent.seed import seed_local_foods, seed_profiles

from .conftest import PROSE_MODEL, SCHEMA_MODEL
from .fakes import FakeProvider
from .test_filipino_dishes import DISHES

TABLES = (
    "schema_migration",
    "user_profile",
    "message",
    "interaction_event",
    "redaction_placeholder",
    "corpus_document",
    "corpus_chunk",
    "local_food",
    "local_food_alias",
    "meal",
    "meal_item",
)
OUR_TABLES = set(TABLES)

# 004, 005 and 006 were reserved by tickets that were never written. A number is
# never reused: the runner records the file name, so a reused number would
# silently skip a file that had already been applied under it.
MIGRATION_FILES = [
    "001_init.sql", "002_router.sql", "003_corpus.sql", "007_meal.sql"
]

# Two documents on different licences, so one predicate has something to exclude.
ENTRIES = [
    CorpusEntry(
        slug="dga-2025-2030",
        title="Dietary Guidelines for Americans, 2025-2030",
        source_url="https://cdn.realfood.gov/DGA_508.pdf",
        publisher="U.S. Departments of Agriculture and Health and Human Services",
        published_on="2026-01-07",
        licence_id="us-gov-public-domain",
    ),
    CorpusEntry(
        slug="who-healthy-diet-factsheet",
        title="Healthy diet fact sheet",
        source_url="https://www.who.int/publications/m/item/healthy-diet-factsheet394",
        publisher="World Health Organization",
        published_on="2020-04-29",
        licence_id="cc-by-nc-sa-3.0-igo",
    ),
]


def stub_fetcher(entry: CorpusEntry) -> list[tuple[str, str]]:
    """The network, replaced. Ingestion is fetch, chunk, embed, store; this test
    supplies the first step so the other three can be checked offline."""
    return [
        ("page 1", (f"{entry.title}: eat a variety of protein foods. " * 6).strip()),
        ("page 2", (f"{entry.title}: limit added sugars and sodium. " * 6).strip()),
    ]


def fake_models() -> object:
    return FakeProvider().models(schema_model=SCHEMA_MODEL, prose_model=PROSE_MODEL)


@pytest.fixture
def empty_database(integration_database_url: str) -> str:
    with psycopg.connect(integration_database_url, autocommit=True) as conn:
        conn.execute("drop schema if exists public cascade")
        conn.execute("create schema public")
    return integration_database_url


def test_no_migration_file_names_a_table_the_checkpointer_owns():
    for path in Path(MIGRATIONS_DIR).glob("*.sql"):
        sql = path.read_text(encoding="utf-8").lower()
        assert not [table for table in CHECKPOINT_TABLES if table in sql]


def test_the_first_migration_creates_the_version_table_and_the_two_tables(empty_database):
    applied = migrate(empty_database)

    assert applied == MIGRATION_FILES
    with psycopg.connect(empty_database) as conn:
        for table in TABLES:
            assert conn.execute("select to_regclass(%s)", (f"public.{table}",)).fetchone()[0]
        recorded = conn.execute("select filename from schema_migration").fetchall()
    assert recorded == [(name,) for name in MIGRATION_FILES]


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


async def test_a_profile_change_is_written_to_postgresql_and_read_back(empty_database):
    """The Profile lives in PostgreSQL alone, so this is where the change is
    proved: written through the seam, read back through the seam."""
    migrate(empty_database)
    seed_profiles(empty_database)
    pool = AsyncConnectionPool(empty_database, open=False)
    await pool.open()
    db = PostgresDatabase(pool)
    try:
        before = await db.load_profile("demo-user-1")
        await db.update_profile(
            "demo-user-1", field="allergies", value=[*before.allergies, "shrimp"]
        )
        await db.update_profile("demo-user-1", field="target_weight_kg", value=70.0)
        after = await db.load_profile("demo-user-1")
    finally:
        await pool.close()

    assert after.allergies == [*before.allergies, "shrimp"]
    assert after.target_weight_kg == 70.0

    with psycopg.connect(empty_database) as conn:
        touched = conn.execute(
            "select updated_at > created_at from user_profile where user_id = %s",
            ("demo-user-1",),
        ).fetchone()[0]
    assert touched


async def test_the_checkpointer_makes_its_own_tables_and_leaves_ours_alone(empty_database):
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
        public_tables = {
            r[0]
            for r in conn.execute(
                "select table_name from information_schema.tables "
                "where table_schema = 'public'"
            ).fetchall()
        }
    assert "checkpoints" in public_tables
    assert OUR_TABLES <= public_tables
    assert public_tables - OUR_TABLES <= CHECKPOINT_TABLES


# --- the Corpus ---------------------------------------------------------------


@pytest.fixture
def corpus(empty_database):
    """A migrated database with two documents ingested through the real command,
    with the network stubbed and the embedding model faked."""
    migrate(empty_database)
    asyncio.run(ingest(empty_database, fake_models(), ENTRIES, fetcher=stub_fetcher))
    return empty_database


def test_the_embedding_column_is_vector_768_with_an_hnsw_cosine_index(corpus):
    with psycopg.connect(corpus) as conn:
        column = conn.execute(
            "select format_type(atttypid, atttypmod) from pg_attribute "
            "where attrelid = 'corpus_chunk'::regclass and attname = 'embedding'"
        ).fetchone()[0]
        index = conn.execute(
            "select indexdef from pg_indexes where indexname = 'corpus_chunk_embedding_idx'"
        ).fetchone()[0].lower()

    assert column == f"vector({EMBEDDING_DIMENSIONS})"
    assert "using hnsw" in index and "vector_cosine_ops" in index


def test_the_stored_vectors_have_unit_norm(corpus):
    with psycopg.connect(corpus) as conn:
        stored = conn.execute("select embedding::text from corpus_chunk").fetchall()

    assert stored
    for (literal,) in stored:
        values = [float(v) for v in literal.strip("[]").split(",")]
        assert len(values) == EMBEDDING_DIMENSIONS
        assert math.isclose(math.sqrt(sum(v * v for v in values)), 1.0, rel_tol=1e-5)


def test_every_chunk_carries_a_licence_identifier_and_an_attribution_string(corpus):
    with psycopg.connect(corpus) as conn:
        rows = conn.execute(
            "select licence_id, attribution from corpus_chunk"
        ).fetchall()

    assert rows
    assert all(licence and attribution for licence, attribution in rows)
    assert any("CC BY-NC-SA 3.0 IGO" in attribution for _, attribution in rows)


def test_one_predicate_excludes_every_non_commercial_chunk_with_no_join(corpus):
    """A commercial review runs this and is finished: the licence columns are on
    the chunk row, so nothing has to be joined to find them."""
    statement = f"select count(*) from corpus_chunk where not {COMMERCIAL_ONLY}"

    assert "join" not in statement.lower()
    with psycopg.connect(corpus) as conn:
        excluded = conn.execute(statement).fetchone()[0]
        kept = conn.execute(
            f"select count(*) from corpus_chunk where {COMMERCIAL_ONLY}"
        ).fetchone()[0]
        who_is_all_of_it = conn.execute(
            f"select count(*) from corpus_chunk where not {COMMERCIAL_ONLY} "
            "and licence_id <> 'cc-by-nc-sa-3.0-igo'"
        ).fetchone()[0]

    assert excluded == 2 and kept == 2
    assert who_is_all_of_it == 0


def test_the_ingest_is_safe_to_run_twice(corpus):
    with psycopg.connect(corpus) as conn:
        before = conn.execute("select count(*) from corpus_chunk").fetchone()[0]

    report = asyncio.run(ingest(corpus, fake_models(), ENTRIES, fetcher=stub_fetcher))

    with psycopg.connect(corpus) as conn:
        after = conn.execute("select count(*) from corpus_chunk").fetchone()[0]
        documents = conn.execute("select count(*) from corpus_document").fetchone()[0]
    assert (before, after, documents) == (4, 4, 2)
    # Unchanged text is not embedded again, which is what makes it cheap as well
    # as safe.
    assert sorted(report.unchanged) == ["dga-2025-2030", "who-healthy-diet-factsheet"]
    assert report.ingested == {} and report.failed == {}


def test_a_document_that_cannot_be_fetched_does_not_lose_the_others(corpus):
    def half_broken(entry: CorpusEntry) -> list[tuple[str, str]]:
        if entry.slug == "dga-2025-2030":
            raise ConnectionError("the web server did not answer")
        return [("page 1", f"{entry.title}: new text about fibre and whole grains. " * 6)]

    report = asyncio.run(ingest(corpus, fake_models(), ENTRIES, fetcher=half_broken))

    assert list(report.failed) == ["dga-2025-2030"]
    assert report.ingested == {"who-healthy-diet-factsheet": 1}


# --- the food log -------------------------------------------------------------


@pytest.fixture
def seeded(empty_database):
    """A migrated database holding the demo Profiles and the real dish table."""
    migrate(empty_database)
    seed_profiles(empty_database)
    seed_local_foods(empty_database)
    return empty_database


@pytest.fixture
async def food_log(seeded):
    pool = AsyncConnectionPool(seeded, open=False)
    await pool.open()
    try:
        yield PostgresDatabase(pool)
    finally:
        await pool.close()


def test_the_dish_table_seed_is_safe_to_run_twice(seeded):
    seed_local_foods(seeded)

    with psycopg.connect(seeded) as conn:
        dishes = conn.execute("select count(*) from local_food").fetchone()[0]
        aliases = conn.execute("select count(*) from local_food_alias").fetchone()[0]
        kinds = dict(
            conn.execute(
                "select value_kind, count(*) from local_food group by value_kind"
            ).fetchall()
        )
    assert dishes == len(DISHES)
    assert aliases > dishes, "a dish answers to more than its own name"
    assert kinds == {"direct": 5, "proxy": 10, "calculated": 6}


def test_the_lookup_the_matcher_runs_first_is_a_text_pattern_index(seeded):
    """`text_pattern_ops` is what lets `alias like 'kare kare%'` use an index
    under any collation, which is what makes the local table cheap enough to
    run before every FoodData Central call."""
    with psycopg.connect(seeded) as conn:
        definitions = {
            name: definition
            for name, definition in conn.execute(
                "select indexname, indexdef from pg_indexes "
                "where tablename in ('local_food', 'local_food_alias')"
            ).fetchall()
        }
        plan = "\n".join(
            row[0]
            for row in conn.execute(
                "explain select * from local_food_alias where alias like 'kare kare%'"
            ).fetchall()
        )

    assert "text_pattern_ops" in definitions["local_food_alias_pattern_idx"]
    assert "text_pattern_ops" in definitions["local_food_name_pattern_idx"]
    assert "Seq Scan" not in plan


async def test_a_dish_is_matched_from_the_local_table_with_its_value_kind(food_log):
    adobo = await food_log.match_local_food("pork adobo")

    assert adobo.name == "Adobo (pork)"
    assert adobo.value_kind == "proxy"
    assert adobo.source_note.startswith("PROXY:")
    # The whole seed row travelled with it, so nothing transcribed is lost.
    assert adobo.source["philfct_food_id"] == "R050"


async def test_a_prefix_finds_the_dish_a_user_named_without_its_qualifier(food_log):
    assert (await food_log.match_local_food("kare kare")).name == "Kare-kare (beef)"
    assert await food_log.match_local_food("not a food at all") is None


async def test_a_value_the_source_does_not_print_comes_back_absent_never_zero(food_log):
    dinuguan = await food_log.match_local_food("dinuguan")

    assert "sodium_mg" not in dinuguan.per_100g
    assert dinuguan.per_100g["kcal"] > 0


async def test_a_meal_is_written_with_its_items_and_a_day_total_is_one_sum(food_log):
    now = datetime.now(MANILA)
    await food_log.store_meal(
        user_id="demo-user-1",
        turn_id=uuid4(),
        eaten_at=now,
        meal_type="breakfast",
        items=[
            MealItemRow(
                ordinal=0, said_as="pandesal", status="matched", quantity=2,
                unit="piece", grams=200.0, portion_assumed=True, source="local",
                food_name="Pandesal", value_kind="direct", match_note="the local table",
                nutrients={"panel": "kept whole"},
                values={"kcal": 618.0, "protein_g": 18.0},
            ),
            MealItemRow(ordinal=1, said_as="kwek kwek", status="unmatched", quantity=3),
        ],
    )

    start, end = day_bounds(now)
    total = await food_log.day_total("demo-user-1", start=start, end=end)

    assert (total.counted, total.not_counted) == (1, 1)
    assert total.values["kcal"] == 618.0
    # The six columns are populated where values exist and null where they do
    # not, so the total says it is short rather than reading as complete.
    assert total.missing["fibre_g"] == 1
    assert total.complete("kcal") and not total.complete("fibre_g")
    # An unmatched Item added nothing to that sum, so the same query names it
    # and the review can say the total is short by an unknown amount.
    assert total.unmatched == ["kwek kwek"]


async def test_the_whole_source_response_is_kept_as_jsonb(seeded, food_log):
    await food_log.store_meal(
        user_id="demo-user-1", turn_id=uuid4(), eaten_at=datetime.now(MANILA),
        meal_type="lunch",
        items=[
            MealItemRow(
                ordinal=0, said_as="adobo", status="matched", grams=100.0,
                source="local", food_name="Adobo (pork)", value_kind="proxy",
                nutrients={"per_100g": {"proximates": {"Protein (g)": "9.9"}}},
                values={"kcal": 199.0},
            )
        ],
    )

    with psycopg.connect(seeded) as conn:
        row = conn.execute(
            "select nutrients, kcal, fibre_g from meal_item"
        ).fetchone()

    assert row[0]["per_100g"]["proximates"]["Protein (g)"] == "9.9"
    assert row[1] == 199
    assert row[2] is None, "a value the source did not state is null, not zero"


async def test_a_correction_fills_in_the_item_that_was_already_written(food_log):
    now = datetime.now(MANILA)
    await food_log.store_meal(
        user_id="demo-user-1", turn_id=uuid4(), eaten_at=now, meal_type="snack",
        items=[MealItemRow(ordinal=0, said_as="kwek kwek", status="unmatched")],
    )
    open_items = await food_log.open_unmatched_items(
        "demo-user-1", since=now - timedelta(hours=24)
    )

    assert [o.said_as for o in open_items] == ["kwek kwek"]
    await food_log.correct_meal_item(
        open_items[0].meal_item_id,
        MealItemRow(
            ordinal=0, said_as="quail egg", status="matched", grams=100.0,
            source="fdc", fdc_id="172194", food_name="Egg, quail, whole, fresh, raw",
            match_note="chosen from 10 candidates", nutrients={"fdcId": 172194},
            values={"kcal": 158.0},
        ),
    )

    start, end = day_bounds(now)
    total = await food_log.day_total("demo-user-1", start=start, end=end)
    assert (total.counted, total.not_counted) == (1, 0)
    assert total.values["kcal"] == 158.0
    # `array_agg` over no unmatched rows is null, and that is an empty list here.
    assert total.unmatched == []
    assert await food_log.open_unmatched_items("demo-user-1", since=start) == []


async def test_retrieval_finds_the_nearest_chunk_and_names_its_document(corpus):
    from psycopg_pool import AsyncConnectionPool

    from nutrigraph_agent.db import PostgresDatabase

    provider = FakeProvider()
    turn = provider.models(
        schema_model=SCHEMA_MODEL, prose_model=PROSE_MODEL
    ).for_turn(known_names=[])
    # The same text that was ingested, so the nearest chunk is a known one.
    query, _ = await turn.embed_query(stub_fetcher(ENTRIES[0])[0][1])

    pool = AsyncConnectionPool(corpus, open=False)
    await pool.open()
    try:
        found = await PostgresDatabase(pool).search_corpus(query, limit=3)
    finally:
        await pool.close()

    assert found
    assert found[0].document == ENTRIES[0].title
    assert found[0].locator == "page 1"
    assert found[0].score > 0.99
