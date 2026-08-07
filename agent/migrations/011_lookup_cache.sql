-- The lookup cache, and the one column the token budget writes.
--
-- Migration number 011. 004, 005 and 006 were reserved by tickets that were
-- never written; a number is never reused, because the runner records the file
-- name and a reused number would silently skip.

-- **The cache holds lookups, never answers.** Both kinds are independent of the
-- User: a Corpus retrieval, and the food a parsed name matched. Nothing that
-- read the Profile or today's Meals is ever written here, and no whole answer
-- is either — judging a question to be impersonal is a judgement, and a wrong
-- judgement serves one User's answer to another. That rule is what makes a
-- stale nutrition answer impossible, and it is enforced by the writers in
-- `db.py` being the only two there are, and by `kind` accepting nothing else.
create table if not exists lookup_cache (
    lookup_cache_id uuid        primary key default gen_random_uuid(),
    kind            text        not null check (kind in ('retrieval', 'food_match')),
    -- The text the entry was keyed on. For a retrieval it is the **redacted**
    -- question, the same string that was embedded: an entry outlives the
    -- 90-day purge of `message.raw_text`, so nothing that the purge would have
    -- removed may be written here. For a food match it is the lowercased,
    -- punctuation-flattened food name, which is not user data at all.
    key_text        text        not null,
    -- Null for a food match, which is keyed on the exact name and not on a
    -- vector. The check below is what makes that structural rather than a habit.
    key_embedding   vector(768),
    value           jsonb       not null,
    -- What the Corpus looked like when the entry was written. A retrieval entry
    -- whose version is not the current one is not read, so a re-ingest
    -- invalidates every one of them without a row having to be found and
    -- deleted first. Null for a food match, which no re-ingest affects.
    corpus_version  text,
    created_at      timestamptz not null default now(),
    hits            integer     not null default 0,
    constraint lookup_cache_embedding_matches_kind
        check ((kind = 'retrieval') = (key_embedding is not null))
);

-- Cosine, like the Corpus index, because the vectors are unit length after the
-- hand re-normalization (ADR 0001). A hit needs a similarity of 0.95 or higher,
-- which the query applies; this index is what makes finding the nearest cheap.
create index if not exists lookup_cache_embedding_idx
    on lookup_cache using hnsw (key_embedding vector_cosine_ops);

-- One row for each food name, so a repeat updates rather than accumulates. The
-- index is partial because a retrieval's `key_text` is not unique and must not
-- be: two differently worded questions can be near enough to share an answer
-- without being the same string.
create unique index if not exists lookup_cache_food_key_idx
    on lookup_cache (key_text) where kind = 'food_match';

create index if not exists lookup_cache_created_idx on lookup_cache (created_at);

-- The token budget is about a measurable cost for each Turn, not a technical
-- limit. When a Turn is still over it after trimming, it proceeds with the
-- trimmed context and says so here, so the eval can see the overrun rather
-- than infer it from a token count. False on every row that fitted.
alter table interaction_event
    add column if not exists over_budget boolean not null default false;
