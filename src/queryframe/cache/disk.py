"""SQLite-backed persistent disk cache."""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

from queryframe.cache.memory import CachedEntry
from queryframe.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CACHE_DIR = os.path.expanduser("~/.queryframe")
_DEFAULT_CACHE_DB = os.path.join(_DEFAULT_CACHE_DIR, "cache.db")


class DiskCache:
    """SQLite-backed persistent cache for cross-session result reuse."""

    def __init__(
        self,
        db_path: str | None = None,
        default_ttl: float = 86400.0,  # 24 hours
    ) -> None:
        self._db_path = db_path or _DEFAULT_CACHE_DB
        self._default_ttl = default_ttl
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the database and table if they don't exist."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    chart_type TEXT,
                    explanation TEXT,
                    data_pickle BLOB,
                    created_at REAL NOT NULL,
                    ttl_seconds REAL NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=5.0)

    def get(self, key: str) -> CachedEntry | None:
        """Get a cached entry from disk."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT code, chart_type, explanation, data_pickle, created_at, ttl_seconds "
                "FROM cache WHERE key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        code, chart_type, explanation, data_pickle, created_at, ttl_seconds = row

        # Check TTL
        if time.time() - created_at > ttl_seconds:
            self.delete(key)
            return None

        try:
            data = pickle.loads(data_pickle) if data_pickle else None
        except Exception:
            self.delete(key)
            return None

        return CachedEntry(
            data=data,
            code=code,
            chart_type=chart_type,
            explanation=explanation or "",
            created_at=created_at,
            ttl_seconds=ttl_seconds,
        )

    def set(
        self,
        key: str,
        data: Any,
        code: str,
        chart_type: str | None = None,
        explanation: str = "",
        ttl: float | None = None,
    ) -> None:
        """Store a result on disk."""
        try:
            data_pickle = pickle.dumps(data, protocol=5)
        except Exception:
            logger.warning("Cannot pickle data for cache, skipping disk cache")
            return

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(key, code, chart_type, explanation, data_pickle, created_at, ttl_seconds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, code, chart_type, explanation, data_pickle, time.time(), ttl or self._default_ttl),
            )

    def delete(self, key: str) -> None:
        """Delete a specific cache entry."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cache")

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM cache WHERE (? - created_at) > ttl_seconds",
                (now,),
            )
            return cursor.rowcount

    @property
    def size(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return row[0] if row else 0
