-- Supabase / Postgres + pgvector schema.
-- NO raw photos anywhere: embeddings and metadata only.
-- Lock every row to the owner's key with row-level security in production.

create extension if not exists vector;

-- Labelled embeddings from calibration + every real swipe (continuous learning).
create table if not exists swipe_labels (
    id            bigserial primary key,
    app           text not null,
    embedding     vector(512) not null,
    label         text not null check (label in ('like', 'pass')),
    primary_flag  boolean not null default false,
    created_at    timestamptz not null default now()
);

-- Audit log of what the engine did and why. `reason` captures red-flag hits etc.
create table if not exists profiles_seen (
    id            bigserial primary key,
    app           text not null,
    external_ref  text not null,
    score         double precision,
    action        text not null check (action in ('like', 'pass', 'review', 'superlike')),
    reason        text,
    created_at    timestamptz not null default now()
);

-- Match threads (message text only; used for replies + shadowban stats).
create table if not exists conversations (
    match_id      text primary key,
    app           text not null,
    messages      jsonb not null default '[]'::jsonb,
    last_activity timestamptz
);

-- The owner's own past sent messages, used as the voice/style corpus.
create table if not exists style_corpus (
    id            bigserial primary key,
    text          text not null,
    created_at    timestamptz not null default now()
);

-- Tunable settings (thresholds, caps, active-hours, red-flag phrase list).
create table if not exists settings (
    key           text primary key,
    value         jsonb not null,
    updated_at    timestamptz not null default now()
);

-- Daily counters feeding the shadowban monitor + dashboard.
create table if not exists stats (
    day           date primary key,
    swipes        integer not null default 0,
    likes         integer not null default 0,
    matches       integer not null default 0
);
