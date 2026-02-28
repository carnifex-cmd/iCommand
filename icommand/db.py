"""SQLite database operations for icommand.

All command history is stored in ~/.icommand/history.db.
Embeddings are stored as binary BLOBs directly in the commands table.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from icommand.config import get_icommand_dir


def _get_db_path() -> Path:
    """Return the path to ~/.icommand/history.db, ensuring the directory exists."""
    return get_icommand_dir() / "history.db"


def _get_connection() -> sqlite3.Connection:
    """Create and return a database connection with row factory."""
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the commands table if it doesn't exist."""
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                directory TEXT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                embedded_at TEXT,
                embedding BLOB
            )
        """)
        # Add embedding column to existing databases that don't have it
        try:
            conn.execute("ALTER TABLE commands ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
    finally:
        conn.close()


def insert_command(command: str, directory: str) -> None:
    """Insert a new command record into the database."""
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO commands (command, directory) VALUES (?, ?)",
            (command, directory),
        )
        conn.commit()
    finally:
        conn.close()


def get_unembedded_commands() -> list[dict]:
    """Return all commands that haven't been embedded yet."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT id, command, directory, timestamp FROM commands WHERE embedded_at IS NULL"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def mark_embedded(ids: list[int], embeddings: list[np.ndarray]) -> None:
    """Mark the given command IDs as embedded and store their embedding vectors."""
    if not ids:
        return

    conn = _get_connection()
    try:
        for cmd_id, embedding in zip(ids, embeddings):
            conn.execute(
                """UPDATE commands
                   SET embedded_at = datetime('now', 'localtime'),
                       embedding = ?
                   WHERE id = ?""",
                (embedding.astype(np.float32).tobytes(), cmd_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_all_embedded_commands() -> list[dict]:
    """Return all commands that have embeddings stored."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """SELECT id, command, directory, timestamp, embedding
               FROM commands
               WHERE embedding IS NOT NULL
               ORDER BY timestamp DESC"""
        )
        rows = []
        for row in cursor.fetchall():
            d = dict(row)
            if d["embedding"]:
                d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            rows.append(d)
        return rows
    finally:
        conn.close()


def get_all_commands() -> list[dict]:
    """Return all command records (without embedding blobs)."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT id, command, directory, timestamp, embedded_at FROM commands ORDER BY timestamp DESC"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
