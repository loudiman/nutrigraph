-- The metric record. LangSmith is the reading tool; this table is the record,
-- because the free tier keeps traces for 14 days. One row per node per Turn.
create table if not exists interaction_event (
    event_id      uuid        primary key default gen_random_uuid(),
    turn_id       uuid        not null,
    user_id       text        not null references user_profile (user_id),
    node          text        not null,
    intent        text,
    model         text,
    latency_ms    integer     not null,
    input_tokens  integer     not null default 0,
    output_tokens integer     not null default 0,
    cost_usd      numeric(14, 8) not null default 0,
    created_at    timestamptz not null default now()
);

create index if not exists interaction_event_turn_idx on interaction_event (turn_id);
create index if not exists interaction_event_created_idx on interaction_event (created_at);

-- The private table that maps a placeholder back to what the User wrote, so an
-- answer can address the User by name although the provider never saw it. The
-- `message` row still holds the raw text: redaction happens at the provider
-- call and nowhere else (ADR 0002).
create table if not exists redaction_placeholder (
    turn_id     uuid        not null,
    placeholder text        not null,
    original    text        not null,
    created_at  timestamptz not null default now(),
    primary key (turn_id, placeholder)
);
