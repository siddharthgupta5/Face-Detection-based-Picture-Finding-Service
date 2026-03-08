"""
share_store.py

SQLite-backed store for shareable result links.

Table: share_links
  
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import List, Optional

from app.config import DB_PATH, LINK_TTL_SECONDS


# Schema

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS share_links (
    id         TEXT PRIMARY KEY,
    photo_ids  TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_created_at ON share_links (created_at);
"""


@contextmanager
def _get_conn():
    """Yield a connected SQLite connection with row_factory set."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        conn.commit()
        yield conn
    finally:
        conn.close()


# Public API 

def create_share_link(photo_ids: List[str]) -> str:
    """
    Persist a list of photo_ids and return a new UUID share token.

    Parameters
    ----------
    photo_ids :
        Ordered list of photo_id strings to include in the shared gallery.

    Returns
    -------
    str
        UUID string that can be used in ``/share/{link_id}``.
    """
    link_id   = str(uuid.uuid4())
    timestamp = time.time()

    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO share_links (id, photo_ids, created_at) VALUES (?, ?, ?)",
            (link_id, json.dumps(photo_ids), timestamp),
        )
        conn.commit()

    return link_id


def get_share_link(link_id: str) -> Optional[List[str]]:
    """
    Retrieve the photo_ids for a share link, or None if expired / not found.

    Parameters
    ----------
    link_id :
        UUID token from the URL.

    Returns
    -------
    list[str] | None
        List of photo_ids if the link exists and is still valid.
        ``None`` if it does not exist or has exceeded LINK_TTL_SECONDS.
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT photo_ids, created_at FROM share_links WHERE id = ?",
            (link_id,),
        ).fetchone()

    if row is None:
        return None

    age = time.time() - row["created_at"]
    if age > LINK_TTL_SECONDS:
        return None  # expired

    return json.loads(row["photo_ids"])


def purge_expired() -> int:
    """
    Delete all expired share links from the database.

    Returns
    -------
    int
        Number of rows deleted.
    """
    cutoff = time.time() - LINK_TTL_SECONDS
    with _get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM share_links WHERE created_at < ?", (cutoff,)
        )
        conn.commit()
        return cursor.rowcount
