-- The recommend path: the food vectors that make personalisation honest, and
-- the row that makes a suggestion measurable.
--
-- Migration number 012. 004, 005, 006, 008, 009 and 010 were reserved by
-- tickets that needed no schema; a number is never reused, because the runner
-- records the file name and a reused number would silently skip.
--
-- `local_food.tags` is not added here: 007 already created it, and the
-- candidate filter reads it as it stands.

-- **The second vector table.** It holds the local dish names and every food
-- this system has matched — hundreds of rows, not millions — so a similarity
-- query over the foods one User actually ate is cheap and exact.
--
-- The dimension is 768 and the index is HNSW cosine, as ADR 0001 fixes and as
-- `corpus_chunk` already is: `gemini-embedding-001` returns 3072, truncated to
-- 768 and re-normalized by hand, because version 1 of the model does not
-- re-normalize a truncated vector. The one helper that does both is
-- `providers.truncate_and_normalize`, and there is no second copy of it.
create table if not exists food_embedding (
    food_embedding_id uuid        primary key default gen_random_uuid(),
    -- Which table the food lives in. 'local' is a `local_food_id`; 'fdc' is a
    -- FoodData Central `fdcId`. Both are text here, so one column keys both.
    source            text        not null check (source in ('local', 'fdc')),
    source_id         text        not null,
    name              text        not null,
    embedding         vector(768) not null,
    first_seen_at     timestamptz not null default now(),
    -- One row for each food, so the seed is safe to run twice and a food this
    -- system matches a second time is not embedded a second time.
    unique (source, source_id)
);

create index if not exists food_embedding_embedding_idx
    on food_embedding using hnsw (embedding vector_cosine_ops);

-- **One suggestion, and what became of it.** Both measurement signals read this
-- table and neither needs a column beyond it:
--
--   acceptance -- `accepted` turns from null to true or false;
--   following  -- a Meal holding one of `foods` appears within 24 hours, which
--                is a join to `meal_item` over this array and nothing more.
--
-- `foods` is the names the model was allowed to say, copied from the candidate
-- rows the SQL filter left standing. A suggestion naming anything else is
-- rejected before it is written, so every name in here traces to a database
-- row.
create table if not exists recommendation (
    recommendation_id uuid        primary key default gen_random_uuid(),
    user_id           text        not null references user_profile (user_id),
    -- The Turn that produced it, so a suggestion joins to its message and its
    -- metric rows exactly as a Meal does.
    turn_id           uuid        not null,
    -- The nutrient the gap was largest on, and by how much, so the eval can ask
    -- whether the suggestion closed what it said it would.
    gap_nutrient      text,
    gap_amount        numeric,
    suggestion        text        not null,
    reason            text        not null,
    foods             text[]      not null default '{}',
    -- Null until the User says. True or false is the acceptance signal.
    accepted          boolean,
    responded_at      timestamptz,
    created_at        timestamptz not null default now()
);

create index if not exists recommendation_user_created_idx
    on recommendation (user_id, created_at);
create index if not exists recommendation_turn_idx on recommendation (turn_id);
-- The following signal reads the recent unanswered ones first.
create index if not exists recommendation_accepted_idx on recommendation (accepted);
