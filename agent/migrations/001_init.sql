-- The version table the runner records applied files in.
create table if not exists schema_migration (
    filename   text primary key,
    applied_at timestamptz not null default now()
);

create table if not exists user_profile (
    user_id          text primary key,
    name             text        not null,
    sex              text,
    age              integer,
    height_cm        numeric,
    weight_kg        numeric,
    target_weight_kg numeric,
    activity_level   text,
    diet_pattern     text,
    units            text        not null default 'metric',
    allergies        text[]      not null default '{}',
    disliked_foods   text[]      not null default '{}',
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now()
);

create table if not exists message (
    message_id uuid        primary key default gen_random_uuid(),
    user_id    text        not null references user_profile (user_id),
    turn_id    uuid        not null,
    role       text        not null check (role in ('user', 'coach')),
    raw_text   text,
    purged_at  timestamptz,
    created_at timestamptz not null default now()
);

create index if not exists message_user_created_idx on message (user_id, created_at);
create index if not exists message_turn_idx on message (turn_id);
