-- The food log, and the local table the matcher reads before it reaches out.
--
-- Migration number 007. 004, 005 and 006 were reserved by tickets that were
-- never written; a number is never reused, because the runner records the file
-- name and a reused number would silently skip.

-- The hand-entered Filipino dish table, transcribed from DOST-FNRI PhilFCT
-- (issue #26). Loaded by `nutrigraph-seed` from agent/seeds/filipino_dishes.json.
--
-- `value_kind` is not decoration. A `proxy` row is a canned or commercial
-- product standing in for the home-cooked dish, and a `calculated` row is
-- computed from component foods with a stated understatement. The Coach must
-- say so, which is why the column and `source_note` travel with every match.
--
-- The six numeric columns are all nullable, because PhilFCT prints '-' for
-- fibre and sodium on part of its items and a null is the truth there. Nothing
-- in this schema, and nothing that reads it, may coerce one to zero.
create table if not exists local_food (
    local_food_id   uuid        primary key default gen_random_uuid(),
    name            text        not null,
    aliases         text[]      not null default '{}',
    tags            text[]      not null default '{}',
    value_kind      text        not null check (value_kind in ('direct', 'proxy', 'calculated')),
    -- What the value actually is. A proxy row's note begins 'PROXY:' and names
    -- the product that stood in; a calculated row's note carries its assumption
    -- and the direction of the error.
    source_note     text        not null,
    philfct_food_id text,
    kcal            numeric,
    protein_g       numeric,
    fat_g           numeric,
    carb_g          numeric,
    fibre_g         numeric,
    sodium_mg       numeric,
    -- The whole seed row, so nothing transcribed from PhilFCT is lost to the
    -- six columns: the full nutrient panel, and a calculated row's components,
    -- gram weights and arithmetic.
    source          jsonb       not null,
    seeded_at       timestamptz not null default now()
);

create unique index if not exists local_food_name_key on local_food (lower(name));

-- Every name a dish answers to, the dish's own name among them, lowercased and
-- normalized. One table, so the matcher's lookup is one indexed query, and the
-- primary key is what makes 'lechon manok' unable to name two different dishes.
create table if not exists local_food_alias (
    alias         text not null primary key,
    local_food_id uuid not null references local_food (local_food_id) on delete cascade
);

-- The text-pattern indexes. `text_pattern_ops` is what lets `alias like 'kare
-- kare%'` use an index under any collation, which is what makes the local table
-- cheap enough to run before every FoodData Central call.
create index if not exists local_food_alias_pattern_idx
    on local_food_alias (alias text_pattern_ops);
create index if not exists local_food_name_pattern_idx
    on local_food (lower(name) text_pattern_ops);
create index if not exists local_food_alias_food_idx on local_food_alias (local_food_id);

-- One eating occasion. The Meal Type comes from the time and from the User's
-- words, not from a list they pick.
create table if not exists meal (
    meal_id    uuid        primary key default gen_random_uuid(),
    user_id    text        not null references user_profile (user_id),
    -- The Turn that logged it, so a Meal joins to its message and its metric rows.
    turn_id    uuid        not null,
    eaten_at   timestamptz not null default now(),
    meal_type  text        not null check (meal_type in ('breakfast', 'lunch', 'dinner', 'snack')),
    created_at timestamptz not null default now()
);

create index if not exists meal_user_eaten_idx on meal (user_id, eaten_at);
create index if not exists meal_turn_idx on meal (turn_id);

-- One food inside a Meal. An Item names what the User said, and is kept whether
-- or not the Coach could attach it to a Food: `status` is 'unmatched' with every
-- nutrient column null when nothing matched, and the Meal is written anyway.
--
-- `said_as` holds the food word the User used, which is what a correction is
-- matched against and what the answer names back. It is a food, not an
-- identifier: the redaction wrapper sends food to the provider unchanged for
-- the same reason (ADR 0002).
create table if not exists meal_item (
    meal_item_id   uuid        primary key default gen_random_uuid(),
    meal_id        uuid        not null references meal (meal_id) on delete cascade,
    ordinal        integer     not null,
    said_as        text        not null,
    quantity       numeric,
    unit           text,
    -- What the nutrient columns were scaled from. Null on an unmatched Item.
    grams          numeric,
    -- True when the gram weight is this codebase's declared serving and not
    -- something the User stated, so the answer can say so.
    portion_assumed boolean    not null default false,
    status         text        not null check (status in ('matched', 'unmatched')),
    -- 'local' or 'fdc'. Null on an unmatched Item.
    source         text        check (source in ('local', 'fdc')),
    local_food_id  uuid        references local_food (local_food_id),
    fdc_id         text,
    food_name      text,
    value_kind     text        check (value_kind in ('direct', 'proxy', 'calculated')),
    -- Why this candidate was chosen, and by what. The same string goes to the
    -- trace, so a match is auditable from either end.
    match_note     text,
    -- The six plain numbers a day total sums, all nullable, because the source
    -- lacks fibre and sodium for part of its items and a null is the truth.
    kcal           numeric,
    protein_g      numeric,
    fat_g          numeric,
    carb_g         numeric,
    fibre_g        numeric,
    sodium_mg      numeric,
    -- The whole source response, so nothing the source said is lost to the six
    -- columns: the full PhilFCT panel, or the FoodData Central food object.
    nutrients      jsonb,
    created_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now(),
    unique (meal_id, ordinal)
);

create index if not exists meal_item_meal_idx on meal_item (meal_id);
create index if not exists meal_item_status_idx on meal_item (status);
