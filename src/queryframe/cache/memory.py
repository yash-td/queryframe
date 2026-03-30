"""In-memory LRU cache."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CachedEntry:
    """A cached query result."""

    data: Any
    code: str
    chart_type: str | None
    explanation: str
    created_at: float
    ttl_seconds: float


class MemoryCache:
    """Thread-safe in-memory LRU cache."""

    def __init__(self, max_size: int = 100, default_ttl: float = 3600.0) -> None:
        self._cache: OrderedDict[str, CachedEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> CachedEntry | None:
        """Get a cached entry by key. Returns None on miss or expiry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if time.time() - entry.created_at > entry.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry

    def set(
        self,
        key: str,
        data: Any,
        code: str,
        chart_type: str | None = None,
        explanation: str = "",
        ttl: float | None = None,
    ) -> None:
        """Store a result in the cache."""
        with self._lock:
            entry = CachedEntry(
                data=data,
                code=code,
                chart_type=chart_type,
                explanation=explanation,
                created_at=time.time(),
                ttl_seconds=ttl or self._default_ttl,
            )
            self._cache[key] = entry
            self._cache.move_to_end(key)

            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }
