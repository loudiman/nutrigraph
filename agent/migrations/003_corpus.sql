-- The Corpus: the curated set of public nutrition documents the Coach cites.
-- Public guidance, not user data, so nothing in here is redacted or purged.
create extension if not exists vector;

create table if not exists corpus_document (
    document_id    uuid        primary key default gen_random_uuid(),
    slug           text        not null unique,
    title          text        not null,
    source_url     text        not null,
    publisher      text        not null,
    published_on   date,
    licence_id     text        not null,
    attribution    text        not null,
    -- Licences are data, not documentation. False for WHO (CC BY-NC-SA 3.0 IGO)
    -- and for FNRI (RA 8293 s.176, which needs prior approval for use for profit).
    commercial_use boolean     not null,
    -- What was ingested last time. An unchanged document is not embedded again,
    -- which is what makes the ingest command safe to run twice.
    content_hash   text        not null,
    ingested_at    timestamptz not null default now()
);

create table if not exists corpus_chunk (
    chunk_id       uuid        primary key default gen_random_uuid(),
    document_id    uuid        not null references corpus_document (document_id) on delete cascade,
    ordinal        integer     not null,
    -- The section heading or the page number, which is the half of a Citation
    -- that names the place within the document.
    locator        text        not null,
    text           text        not null,
    -- Repeated from the document on purpose, so a licence filter never needs a
    -- join. A commercial review excludes every non-commercial chunk in one
    -- predicate:  delete from corpus_chunk where not commercial_use;
    -- and the attribution string travels with the chunk, so WHO's attribution
    -- requirement is satisfied wherever a chunk is used.
    licence_id     text        not null,
    attribution    text        not null,
    commercial_use boolean     not null,
    -- gemini-embedding-001 returns 3072 dimensions, truncated to 768 and
    -- re-normalized by hand, because version 1 of the model does not
    -- re-normalize a truncated vector. HNSW accepts at most 2000 dimensions,
    -- which is why 768 and not 3072 (ADR 0001). Changing this number forces a
    -- re-index of the whole Corpus.
    embedding      vector(768) not null,
    ingested_at    timestamptz not null default now(),
    unique (document_id, ordinal)
);

-- Cosine, because the vectors are unit length after the hand re-normalization.
create index if not exists corpus_chunk_embedding_idx
    on corpus_chunk using hnsw (embedding vector_cosine_ops);

create index if not exists corpus_chunk_licence_idx
    on corpus_chunk (commercial_use);
