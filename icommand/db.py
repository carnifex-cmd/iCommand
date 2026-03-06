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
        # Add exit_code column to existing databases that don't have it
        try:
            conn.execute("ALTER TABLE commands ADD COLUMN exit_code INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Add embedding_model column for model migration tracking
        try:
            conn.execute("ALTER TABLE commands ADD COLUMN embedding_model TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
    finally:
        conn.close()


def insert_command(command: str, directory: str, exit_code: Optional[int] = None) -> None:
    """Insert a new command record into the database."""
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO commands (command, directory, exit_code) VALUES (?, ?, ?)",
            (command, directory, exit_code),
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


def clear_stale_embeddings(current_model: str) -> int:
    """Clear all embeddings if the model has changed.

    If any existing embeddings were created by a different model (or have no
    model tag — i.e. legacy MiniLM embeddings), wipe all embeddings so they
    get re-embedded with the current model on next sync.

    Args:
        current_model: Name of the active embedding model.

    Returns:
        Number of rows cleared.
    """
    conn = _get_connection()
    try:
        # Check if any embedded rows have a different (or NULL) model
        cursor = conn.execute(
            """SELECT COUNT(*) FROM commands
               WHERE embedding IS NOT NULL
                 AND (embedding_model IS NULL OR embedding_model != ?)""",
            (current_model,),
        )
        stale_count = cursor.fetchone()[0]

        if stale_count == 0:
            return 0

        # Clear all embeddings for a clean re-embed
        conn.execute(
            """UPDATE commands
               SET embedding = NULL,
                   embedded_at = NULL,
                   embedding_model = NULL
               WHERE embedding IS NOT NULL"""
        )
        conn.commit()
        return stale_count
    finally:
        conn.close()


def mark_embedded(ids: list[int], embeddings: list[np.ndarray], model_name: str = "arctic-xs") -> None:
    """Mark the given command IDs as embedded and store their embedding vectors."""
    if not ids:
        return

    conn = _get_connection()
    try:
        for cmd_id, embedding in zip(ids, embeddings):
            conn.execute(
                """UPDATE commands
                   SET embedded_at = datetime('now', 'localtime'),
                       embedding = ?,
                       embedding_model = ?
                   WHERE id = ?""",
                (embedding.astype(np.float32).tobytes(), model_name, cmd_id),
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


def get_commands_by_ids(ids: list[int]) -> list[dict]:
    """Return command records for given IDs (without embedding blobs).
    
    Args:
        ids: List of command IDs to fetch
        
    Returns:
        List of command dicts in the same order as input IDs
    """
    if not ids:
        return []
        
    conn = _get_connection()
    try:
        # Use parameterized query with placeholders
        placeholders = ','.join('?' * len(ids))
        cursor = conn.execute(
            f"""SELECT id, command, directory, timestamp, embedded_at 
                FROM commands 
                WHERE id IN ({placeholders})""",
            ids
        )
        # Build lookup by ID
        rows_by_id = {row['id']: dict(row) for row in cursor.fetchall()}
        # Return in order of input IDs
        return [rows_by_id[i] for i in ids if i in rows_by_id]
    finally:
        conn.close()


def get_recent_commands(limit: int = 20, offset: int = 0) -> list[dict]:
    """Return recent command records with pagination.
    
    Args:
        limit: Maximum number of commands to return
        offset: Number of commands to skip
        
    Returns:
        List of command dicts ordered by timestamp DESC
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """SELECT id, command, directory, timestamp, embedded_at 
               FROM commands 
               ORDER BY timestamp DESC 
               LIMIT ? OFFSET ?""",
            (limit, offset)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_command_count() -> int:
    """Return the total number of commands in the database.
    
    Returns:
        Total count of commands
    """
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM commands")
        return cursor.fetchone()[0]
    finally:
        conn.close()
