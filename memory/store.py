"""Persistent memory database foundation for the coding agent.

Opens a SQLite database at ``.coding_agent/memory.db`` within the given
project root, loads the optional *sqlite-vec* extension, and runs a single
idempotent schema migration that creates ALL tables for every memory layer
(episodes, facts, graph_nodes / graph_edges) plus their FTS5 and vec0
companion virtual tables.

Later phases add code that USES these tables but will add NO further
migrations.
"""

from __future__ import annotations

import os
import sys
import sqlite3

SCHEMA_VERSION = 2


def open_store(project_root: str) -> "MemoryStore":
    """Open (create if absent) the persistent-memory store for *project_root*.

    Returns a new :class:`MemoryStore` backed by an SQLite connection that has
    already had its pragmas applied, *sqlite-vec* loaded (if available), and
    schema migration executed.
    """
    db_dir = os.path.join(project_root, ".coding_agent")
    os.makedirs(db_dir, exist_ok=True)

    db_path = os.path.join(db_dir, "memory.db")
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row

    # ── pragmas ───────────────────────────────────────────────────────────
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")

    # ── sqlite-vec attempt ────────────────────────────────────────────────
    vec_enabled = False
    try:
        conn.enable_load_extension(True)
        import sqlite_vec  # noqa: F811

        sqlite_vec.load(conn)
        vec_enabled = True
    except Exception as exc:
        print(
            f"[memory] sqlite-vec unavailable, running in FTS-only mode: {exc}",
            file=sys.stderr,
        )

    # ── idempotent schema migration ───────────────────────────────────────
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < 1:
        _run_migration(conn, vec_enabled)
    if current < 2:
        _migrate_v1_to_v2(conn)
    if current < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

    return MemoryStore(conn=conn, db_path=db_path, vec_enabled=vec_enabled)


# ── internal migration helpers ────────────────────────────────────────────────


def _run_migration(conn: sqlite3.Connection, vec_enabled: bool) -> None:
    """Create all tables and virtual tables."""

    # Base-table DDL (always executed).
    conn.executescript("""
        -- 7.1 episodes ────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS episodes (
            id              TEXT PRIMARY KEY,
            gist            TEXT NOT NULL,
            salience        INTEGER NOT NULL CHECK (salience BETWEEN 1 AND 10),
            created_at      TEXT NOT NULL,
            last_relevant_at TEXT,
            last_accessed_at TEXT,
            transcript_ids  TEXT,
            has_open_loop   INTEGER NOT NULL DEFAULT 0,
            facts_extracted_at TEXT,
            deleted_at      TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
            USING fts5(gist, content='episodes', content_rowid='rowid');

        -- 7.2 facts ────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS facts (
            id                 INTEGER PRIMARY KEY,
            kind               TEXT NOT NULL,
            key                TEXT NOT NULL,
            value              TEXT NOT NULL,
            salience_floor     REAL    NOT NULL DEFAULT 0.1,
            d_base             REAL    NOT NULL DEFAULT 0.3,
            retrieval_weight   REAL    NOT NULL DEFAULT 1.0,
            first_seen_at      TEXT NOT NULL,
            last_confirmed_at  TEXT,
            last_accessed_at   TEXT,
            valid_from         TEXT NOT NULL,
            valid_to           TEXT,
            active             INTEGER NOT NULL DEFAULT 1,
            deleted_at         TEXT,
            anchor_path        TEXT,
            anchor_symbol      TEXT,
            anchor_hash        TEXT,
            learned_commit     TEXT,
            confidence         REAL NOT NULL DEFAULT 0.5,
            source             TEXT NOT NULL DEFAULT 'legacy'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
            USING fts5(key, value, kind, content='facts',
                       content_rowid='id', tokenize='porter');

        -- 7.3 graph_nodes ──────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id              INTEGER PRIMARY KEY,
            type            TEXT NOT NULL CHECK (type IN ('rule','decision','pivot','spec')),
            title           TEXT NOT NULL,
            body            TEXT NOT NULL,
            extra           TEXT,
            created_at      TEXT NOT NULL,
            superseded_at   TEXT,
            active          INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS graph_edges (
            id           INTEGER PRIMARY KEY,
            from_id      INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
            to_id        INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
            edge_type    TEXT NOT NULL CHECK (edge_type IN ('supersedes','implements','constrains','refines','relates_to')),
            created_at   TEXT NOT NULL,
            UNIQUE (from_id, to_id, edge_type)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS graph_nodes_fts
            USING fts5(title, body, content='graph_nodes', content_rowid='id');
    """)

    # Vector-table DDL (only when sqlite-vec is available).
    if vec_enabled:
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vec
                USING vec0(embedding float[768]);

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_key_vec
                USING vec0(embedding float[768]);

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_value_vec
                USING vec0(embedding float[768]);

            CREATE VIRTUAL TABLE IF NOT EXISTS graph_nodes_vec
                USING vec0(embedding float[768]);
        """)


# Six anchor columns added to `facts` in schema v2 (code-anchoring, schema v2).
_V2_ANCHOR_COLUMNS = [
    ("anchor_path", "TEXT"),
    ("anchor_symbol", "TEXT"),
    ("anchor_hash", "TEXT"),
    ("learned_commit", "TEXT"),
    ("confidence", "REAL NOT NULL DEFAULT 0.5"),
    ("source", "TEXT NOT NULL DEFAULT 'legacy'"),
]


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Additively backfill the anchor columns onto an existing v1 ``facts`` table.

    Introspects ``PRAGMA table_info(facts)`` and ``ALTER TABLE ... ADD COLUMN``s
    only the columns that are missing -- a no-op on a fresh DB (whose base DDL
    already declares them) and additive, data-preserving on a real v1 DB.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
    for name, decl in _V2_ANCHOR_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {decl}")


# ── public class ────────────────────────────────────────────────────────────────


class MemoryStore:
    """Lightweight handle on the persistent-memory SQLite connection."""

    def __init__(self, conn: sqlite3.Connection, db_path: str, vec_enabled: bool):
        self.conn = conn
        self.db_path = db_path
        self.vec_enabled = vec_enabled

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()
