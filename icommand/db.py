"""SQLite database operations for icommand."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from icommand.config import get_icommand_dir

logger = logging.getLogger(__name__)

_AUTO_VACUUM_INCREMENTAL = 2


def _get_db_path() -> Path:
    """Return the path to ~/.icommand/history.db, ensuring the directory exists."""
    return get_icommand_dir() / "history.db"


def _get_connection() -> sqlite3.Connection:
    """Create and return a database connection with row factory."""
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _fts_exists(conn: sqlite3.Connection) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='commands_fts'"
    )
    return cursor.fetchone() is not None


def init_db() -> None:
    """Create the database schema if it doesn't exist."""
    db_path = _get_db_path()
    is_new_db = not db_path.exists()

    conn = _get_connection()
    try:
        if is_new_db:
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                directory TEXT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                embedded_at TEXT,
                embedding BLOB,
                exit_code INTEGER,
                embedding_model TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        # Add columns for older databases.
        for sql in (
            "ALTER TABLE commands ADD COLUMN embedding BLOB",
            "ALTER TABLE commands ADD COLUMN exit_code INTEGER",
            "ALTER TABLE commands ADD COLUMN embedding_model TEXT",
        ):
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass

        for sql in (
            "CREATE INDEX IF NOT EXISTS idx_commands_timestamp ON commands(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_commands_unembedded ON commands(id) WHERE embedded_at IS NULL",
            "CREATE INDEX IF NOT EXISTS idx_commands_directory ON commands(directory)",
        ):
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass

        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS commands_fts USING fts5(
                    command_text,
                    content='commands',
                    content_rowid='id'
                )
                """
            )
        except sqlite3.OperationalError:
            pass

        for sql in (
            """
            CREATE TRIGGER IF NOT EXISTS commands_fts_insert
            AFTER INSERT ON commands
            BEGIN
                INSERT INTO commands_fts(rowid, command_text)
                VALUES (new.id, new.command);
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS commands_fts_delete
            AFTER DELETE ON commands
            BEGIN
                INSERT INTO commands_fts(commands_fts, rowid, command_text)
                VALUES ('delete', old.id, old.command);
            END
            """,
        ):
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass

        conn.commit()
    finally:
        conn.close()


def ensure_incremental_auto_vacuum() -> bool:
    """Enable incremental auto-vacuum for existing databases if needed."""
    conn = _get_connection()
    try:
        current = conn.execute("PRAGMA auto_vacuum").fetchone()[0]
        if current == _AUTO_VACUUM_INCREMENTAL:
            return False

        conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
        return True
    finally:
        conn.close()


def insert_command(command: str, directory: str, exit_code: Optional[int] = None) -> int:
    """Insert a new command record into the database."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO commands (command, directory, exit_code) VALUES (?, ?, ?)",
            (command, directory, exit_code),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_unembedded_commands_for_hot_window(min_id: int, limit: int) -> list[dict]:
    """Return the newest unembedded commands in the active hot window."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, command, directory, timestamp
            FROM commands
            WHERE embedded_at IS NULL AND id >= ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (min_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def clear_stale_embeddings(current_model: str) -> int:
    """Clear embeddings created by a different embedding model."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM commands
            WHERE embedding IS NOT NULL
              AND (embedding_model IS NULL OR embedding_model != ?)
            """,
            (current_model,),
        )
        stale_count = cursor.fetchone()[0]
        if stale_count == 0:
            return 0

        conn.execute(
            """
            UPDATE commands
            SET embedding = NULL,
                embedded_at = NULL,
                embedding_model = NULL
            WHERE embedding IS NOT NULL
              AND (embedding_model IS NULL OR embedding_model != ?)
            """,
            (current_model,),
        )
        conn.commit()
        return stale_count
    finally:
        conn.close()


def mark_embedded(
    ids: list[int],
    embeddings: list[np.ndarray],
    model_name: str = "arctic-xs",
) -> None:
    """Store embeddings for the given command IDs."""
    if not ids:
        return

    rows = [
        (
            embedding.astype(np.float32).tobytes(),
            model_name,
            cmd_id,
        )
        for cmd_id, embedding in zip(ids, embeddings)
    ]

    conn = _get_connection()
    try:
        conn.executemany(
            """
            UPDATE commands
            SET embedded_at = datetime('now', 'localtime'),
                embedding = ?,
                embedding_model = ?
            WHERE id = ?
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def iter_embedded_commands(
    min_id: int = 1,
    batch_size: int = 5000,
) -> Iterator[list[dict]]:
    """Yield embedded commands in ascending ID order."""
    last_id = min_id - 1

    while True:
        conn = _get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, command, directory, timestamp, embedding
                FROM commands
                WHERE embedding IS NOT NULL
                  AND id >= ?
                  AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (min_id, last_id, batch_size),
            )
            fetched = cursor.fetchall()
        finally:
            conn.close()

        if not fetched:
            return

        rows = []
        for row in fetched:
            data = dict(row)
            if data["embedding"]:
                data["embedding"] = np.frombuffer(data["embedding"], dtype=np.float32)
            rows.append(data)

        last_id = rows[-1]["id"]
        yield rows


def get_embedded_command_count(min_id: int = 1) -> int:
    """Return the number of embedded commands within the active hot window."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM commands WHERE embedding IS NOT NULL AND id >= ?",
            (min_id,),
        )
        return cursor.fetchone()[0]
    finally:
        conn.close()


def get_all_commands() -> list[dict]:
    """Return all command records (without embedding blobs)."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, command, directory, timestamp, embedded_at
            FROM commands
            ORDER BY timestamp DESC
            """
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_commands_by_ids(ids: list[int]) -> list[dict]:
    """Return command records for the given IDs without embedding blobs."""
    if not ids:
        return []

    placeholders = ",".join("?" * len(ids))
    conn = _get_connection()
    try:
        cursor = conn.execute(
            f"""
            SELECT id, command, directory, timestamp, embedded_at
            FROM commands
            WHERE id IN ({placeholders})
            """,
            ids,
        )
        rows_by_id = {row["id"]: dict(row) for row in cursor.fetchall()}
        return [rows_by_id[i] for i in ids if i in rows_by_id]
    finally:
        conn.close()


def get_recent_commands(limit: int = 20, offset: int = 0) -> list[dict]:
    """Return recent command records with pagination."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, command, directory, timestamp, embedded_at
            FROM commands
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_command_count() -> int:
    """Return the total number of retained commands."""
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM commands")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def get_max_command_id() -> int:
    """Return the largest command ID currently stored."""
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT COALESCE(MAX(id), 0) FROM commands")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def delete_oldest_commands(limit: int) -> int:
    """Delete up to `limit` oldest commands by ID."""
    if limit <= 0:
        return 0

    conn = _get_connection()
    try:
        conn.execute(
            """
            DELETE FROM commands
            WHERE id IN (
                SELECT id FROM commands
                ORDER BY id ASC
                LIMIT ?
            )
            """,
            (limit,),
        )
        conn.commit()
        return conn.execute("SELECT changes()").fetchone()[0]
    finally:
        conn.close()


def clear_embeddings_before_id(floor_id: int) -> int:
    """Drop embeddings for rows that are older than the semantic hot window."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            UPDATE commands
            SET embedding = NULL,
                embedded_at = NULL,
                embedding_model = NULL
            WHERE id < ?
              AND embedding IS NOT NULL
            """,
            (floor_id,),
        )
        conn.commit()
        return conn.execute("SELECT changes()").fetchone()[0]
    finally:
        conn.close()


def keyword_search(query: str, limit: int = 1000) -> list[int]:
    """Search commands by keyword using FTS5."""
    conn = _get_connection()
    try:
        if not _fts_exists(conn):
            return []

        terms = query.strip().split()
        if not terms:
            return []

        fts_terms = []
        for term in terms:
            escaped = term.replace('"', '""')
            fts_terms.append(f'"{escaped}"*')
        fts_query = " AND ".join(fts_terms)

        cursor = conn.execute(
            """
            SELECT rowid FROM commands_fts
            WHERE commands_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        )
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError as exc:
        logger.debug("FTS search error: %s", exc)
        return []
    finally:
        conn.close()


def get_embedded_commands_by_ids(ids: list[int]) -> list[dict]:
    """Return embedded command records for the given IDs."""
    if not ids:
        return []

    placeholders = ",".join("?" * len(ids))
    conn = _get_connection()
    try:
        cursor = conn.execute(
            f"""
            SELECT id, command, directory, timestamp, embedding
            FROM commands
            WHERE id IN ({placeholders}) AND embedding IS NOT NULL
            """,
            ids,
        )
        rows = []
        for row in cursor.fetchall():
            data = dict(row)
            if data["embedding"]:
                data["embedding"] = np.frombuffer(data["embedding"], dtype=np.float32)
            rows.append(data)
        return rows
    finally:
        conn.close()


def rebuild_fts_index() -> int:
    """Rebuild the FTS5 index from scratch if it is empty."""
    conn = _get_connection()
    try:
        if not _fts_exists(conn):
            return 0

        try:
            cursor = conn.execute("SELECT COUNT(*) FROM commands_fts")
            count = cursor.fetchone()[0]
            if count > 0:
                return count
        except sqlite3.OperationalError:
            pass

        conn.execute("DELETE FROM commands_fts")
        cursor = conn.execute("SELECT id, command FROM commands")
        commands = cursor.fetchall()
        for cmd_id, command in commands:
            conn.execute(
                "INSERT INTO commands_fts(rowid, command_text) VALUES (?, ?)",
                (cmd_id, command),
            )
        conn.commit()
        return len(commands)
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def get_app_state(key: str) -> Optional[str]:
    """Read a persisted application-level setting."""
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_app_state(key: str, value: str) -> None:
    """Persist an application-level setting."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO app_state(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_db_page_stats() -> dict[str, int]:
    """Return low-level SQLite page metrics for compaction decisions."""
    conn = _get_connection()
    try:
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
        return {
            "page_size": page_size,
            "page_count": page_count,
            "freelist_count": freelist_count,
        }
    finally:
        conn.close()


def optimize_database() -> None:
    """Run lightweight SQLite/FTS optimization."""
    conn = _get_connection()
    try:
        conn.execute("PRAGMA optimize")
        try:
            if _fts_exists(conn):
                conn.execute("INSERT INTO commands_fts(commands_fts) VALUES ('optimize')")
        except sqlite3.OperationalError:
            pass
        conn.execute("PRAGMA incremental_vacuum")
        conn.commit()
    finally:
        conn.close()


def vacuum_database() -> None:
    """Run a full SQLite VACUUM."""
    conn = _get_connection()
    try:
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()
