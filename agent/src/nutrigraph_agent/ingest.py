"""Fetch, chunk, embed, store.

Its own command, and not part of `nutrigraph-seed`, because it is the slow step:
it talks to forty web servers and to the embedding model. Tying it to the demo
Profiles would make a two-second command a two-minute one.

**Safe to run twice.** A document is keyed by its slug and its chunks are
replaced wholesale, so a second run cannot double the Corpus. A document whose
extracted text has not changed is skipped before any embedding call, so a second
run is also nearly free.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from ._windows import use_selector_event_loop
from .config import Settings
from .corpus import (
    MANIFEST,
    CorpusEntry,
    content_hash,
    fetch_chunks,
    load_manifest,
)
from .db import vector_literal
from .providers import Models, langchain_embedding_factory, langchain_factory

log = logging.getLogger("nutrigraph.agent.ingest")

# One embedding request per batch of chunks, to stay inside the free tier's
# request-per-minute limit without a rate limiter.
BATCH = 16

UPSERT_DOCUMENT = """
insert into corpus_document
    (slug, title, source_url, publisher, published_on,
     licence_id, attribution, commercial_use, content_hash)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
on conflict (slug) do update set
    title = excluded.title,
    source_url = excluded.source_url,
    publisher = excluded.publisher,
    published_on = excluded.published_on,
    licence_id = excluded.licence_id,
    attribution = excluded.attribution,
    commercial_use = excluded.commercial_use,
    content_hash = excluded.content_hash,
    ingested_at = now()
returning document_id
"""

INSERT_CHUNK = """
insert into corpus_chunk
    (document_id, ordinal, locator, text,
     licence_id, attribution, commercial_use, embedding)
values (%s, %s, %s, %s, %s, %s, %s, %s::vector)
"""

Fetcher = Callable[[CorpusEntry], list[tuple[str, str]]]


@dataclass
class Report:
    ingested: dict[str, int] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def chunks(self) -> int:
        return sum(self.ingested.values())

    def __str__(self) -> str:
        lines = [
            f"ingested {len(self.ingested)} documents, {self.chunks} chunks",
            f"unchanged {len(self.unchanged)}",
        ]
        lines += [f"FAILED {slug}: {why}" for slug, why in self.failed.items()]
        return "\n".join(lines)


def _stored_hash(conn: psycopg.Connection, slug: str) -> str | None:
    row = conn.execute(
        "select content_hash from corpus_document where slug = %s", (slug,)
    ).fetchone()
    return row[0] if row else None


async def ingest(
    database_url: str,
    models: Models,
    entries: Sequence[CorpusEntry] | None = None,
    *,
    manifest: Path = MANIFEST,
    fetcher: Fetcher = fetch_chunks,
    force: bool = False,
) -> Report:
    entries = list(entries if entries is not None else load_manifest(manifest))
    report = Report()
    # No Profile and no known names: the Corpus is public guidance. The wrapper
    # still runs, because there is no route to the provider that skips it.
    turn = models.for_turn(known_names=[])

    with psycopg.connect(database_url) as conn:
        for entry in entries:
            try:
                passages = fetcher(entry)
                if not passages:
                    raise ValueError("no text could be extracted")
                digest = content_hash(passages)
                if not force and _stored_hash(conn, entry.slug) == digest:
                    report.unchanged.append(entry.slug)
                    continue

                vectors: list[list[float]] = []
                for start in range(0, len(passages), BATCH):
                    batch = [text for _, text in passages[start : start + BATCH]]
                    embedded, _ = await turn.embed_documents(batch)
                    vectors.extend(embedded)

                licence = entry.licence
                document_id = conn.execute(
                    UPSERT_DOCUMENT,
                    (
                        entry.slug, entry.title, entry.source_url, entry.publisher,
                        entry.published_on, licence.licence_id, entry.attribution,
                        licence.commercial_use, digest,
                    ),
                ).fetchone()[0]
                # Replaced wholesale, so a re-run cannot double the Corpus and a
                # shortened document cannot leave orphan chunks behind.
                conn.execute("delete from corpus_chunk where document_id = %s", (document_id,))
                conn.cursor().executemany(
                    INSERT_CHUNK,
                    [
                        (
                            document_id, ordinal, locator, text,
                            licence.licence_id, entry.attribution,
                            licence.commercial_use, vector_literal(vector),
                        )
                        for ordinal, ((locator, text), vector) in enumerate(
                            zip(passages, vectors, strict=True)
                        )
                    ],
                )
                conn.commit()
                report.ingested[entry.slug] = len(passages)
                log.info("ingested %s (%d chunks)", entry.slug, len(passages))
            except Exception as exc:
                # One unreachable web server is not a reason to lose the other
                # thirty-nine documents. The run reports it and exits non-zero.
                conn.rollback()
                report.failed[entry.slug] = f"{type(exc).__name__}: {exc}"
                log.warning("could not ingest %s: %s", entry.slug, exc)
    return report


def main() -> int:
    use_selector_event_loop()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings.from_env()
    models = Models(
        factory=langchain_factory(settings.model_provider),
        schema_model=settings.schema_model,
        prose_model=settings.prose_model,
        embedding_factory=langchain_embedding_factory(settings.model_provider),
        embedding_model=settings.embedding_model,
    )
    report = asyncio.run(ingest(settings.database_url, models, force="--force" in sys.argv))
    print(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
