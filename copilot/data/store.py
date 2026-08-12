"""SQLite store for local development.

Mirrors schema.sql but with SQLite types (vectors stored as JSON text). The
production target is Supabase/Postgres + pgvector; this keeps dev dependency-free.
No raw photos are stored — vectors and metadata only.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date

_SCHEMA = """
create table if not exists swipe_labels (
    id integer primary key autoincrement,
    app text not null,
    embedding text not null,          -- JSON array of floats
    label text not null check (label in ('like','pass')),
    primary_flag integer not null default 0,
    source text,                      -- capture pipeline: my_likes | recs | folder
    created_at text not null default (datetime('now'))
);
create table if not exists profiles_seen (
    id integer primary key autoincrement,
    app text not null,
    external_ref text not null,
    score real,
    action text not null,
    reason text,
    created_at text not null default (datetime('now'))
);
create table if not exists conversations (
    match_id text primary key,
    app text not null,
    messages text not null default '[]',
    last_activity text
);
create table if not exists style_corpus (
    id integer primary key autoincrement,
    text text not null,
    created_at text not null default (datetime('now'))
);
create table if not exists settings (
    key text primary key,
    value text not null,
    updated_at text not null default (datetime('now'))
);
create table if not exists stats (
    day text primary key,
    swipes integer not null default 0,
    likes integer not null default 0,
    matches integer not null default 0
);
"""


class SQLiteStore:
    def __init__(self, path: str = "copilot_dev.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        # Add the `source` column to pre-existing databases created before it.
        cols = [r[1] for r in self.conn.execute("pragma table_info(swipe_labels)")]
        if "source" not in cols:
            self.conn.execute("alter table swipe_labels add column source text")

    def close(self) -> None:
        self.conn.close()

    # --- swipe labels / continuous learning -----------------------------
    def add_swipe_label(self, app: str, embedding, label: str,
                        primary: bool = False, source: str | None = None) -> None:
        self.conn.execute(
            "insert into swipe_labels (app, embedding, label, primary_flag, source) "
            "values (?, ?, ?, ?, ?)",
            (app, json.dumps(list(embedding)), label, int(primary), source),
        )
        self.conn.commit()

    def get_swipe_labels(self, app: str | None = None,
                         source: str | None = None) -> list[tuple[list[float], str]]:
        clauses, params = [], []
        if app:
            clauses.append("app = ?")
            params.append(app)
        if source:
            clauses.append("source = ?")
            params.append(source)
        where = (" where " + " and ".join(clauses)) if clauses else ""
        rows = self.conn.execute(
            f"select embedding, label from swipe_labels{where} order by id", params
        )
        return [(json.loads(r["embedding"]), r["label"]) for r in rows]

    # --- audit log -------------------------------------------------------
    def record_profile_seen(self, app: str, external_ref: str, action: str,
                            score: float | None = None, reason: str | None = None) -> None:
        self.conn.execute(
            "insert into profiles_seen (app, external_ref, score, action, reason) "
            "values (?, ?, ?, ?, ?)",
            (app, external_ref, score, action, reason),
        )
        self.conn.commit()

    # --- conversations ---------------------------------------------------
    def upsert_conversation(self, match_id: str, app: str, messages: list[dict],
                            last_activity: str | None = None) -> None:
        self.conn.execute(
            "insert into conversations (match_id, app, messages, last_activity) "
            "values (?, ?, ?, ?) "
            "on conflict(match_id) do update set messages=excluded.messages, "
            "last_activity=excluded.last_activity",
            (match_id, app, json.dumps(messages), last_activity),
        )
        self.conn.commit()

    # --- style corpus ----------------------------------------------------
    def add_style_message(self, text: str) -> None:
        self.conn.execute("insert into style_corpus (text) values (?)", (text,))
        self.conn.commit()

    def get_style_corpus(self, limit: int = 200) -> list[str]:
        rows = self.conn.execute(
            "select text from style_corpus order by id desc limit ?", (limit,)
        )
        return [r["text"] for r in rows][::-1]

    # --- settings --------------------------------------------------------
    def set_setting(self, key: str, value) -> None:
        self.conn.execute(
            "insert into settings (key, value) values (?, ?) "
            "on conflict(key) do update set value=excluded.value, "
            "updated_at=datetime('now')",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_setting(self, key: str, default=None):
        row = self.conn.execute(
            "select value from settings where key = ?", (key,)
        ).fetchone()
        return json.loads(row["value"]) if row else default

    # --- stats / shadowban ----------------------------------------------
    def bump_stats(self, day: str | None = None, swipes: int = 0, likes: int = 0,
                   matches: int = 0) -> None:
        day = day or date.today().isoformat()
        self.conn.execute(
            "insert into stats (day, swipes, likes, matches) values (?, ?, ?, ?) "
            "on conflict(day) do update set "
            "swipes=swipes+excluded.swipes, likes=likes+excluded.likes, "
            "matches=matches+excluded.matches",
            (day, swipes, likes, matches),
        )
        self.conn.commit()

    def get_stats(self, days: int = 14) -> list[dict]:
        rows = self.conn.execute(
            "select day, swipes, likes, matches from stats order by day desc limit ?",
            (days,),
        )
        return [dict(r) for r in rows][::-1]
