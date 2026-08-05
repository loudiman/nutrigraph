"""The migration runner, the seed command, and the checkpointer's schema.

These run against a real PostgreSQL. Set NUTRIGRAPH_TEST_DATABASE_URL to a
database you are happy to have emptied.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import psycopg
import pytest

from nutrigraph_agent.app import open_checkpointer
from nutrigraph_agent.config import CHECKPOINT_SCHEMA
from nutrigraph_agent.corpus import CorpusEntry
from nutrigraph_agent.db import COMMERCIAL_ONLY
from nutrigraph_agent.graph import build_graph
from nutrigraph_agent.ingest import ingest
from nutrigraph_agent.migrate import MIGRATIONS_DIR, migrate, pending
from nutrigraph_agent.providers import EMBEDDING_DIMENSIONS
from nutrigraph_agent.seed import seed_profiles

from .conftest import PROSE_MODEL, SCHEMA_MODEL
from .fakes import FakeProvider

TABLES = (
    "schema_migration",
    "user_profile",
    "message",
    "interaction_event",
    "redaction_placeholder",
    "corpus_document",
    "corpus_chunk",
)

MIGRATION_FILES = ["001_init.sql", "002_router.sql", "003_corpus.sql"]

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
        conn.execute(f"drop schema if exists {CHECKPOINT_SCHEMA} cascade")
    return integration_database_url


def test_no_migration_file_references_the_langgraph_schema():
    for path in Path(MIGRATIONS_DIR).glob("*.sql"):
        assert CHECKPOINT_SCHEMA not in path.read_text(encoding="utf-8").lower()


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
