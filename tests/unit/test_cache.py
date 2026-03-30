"""Tests for caching subsystem."""

import time

import pytest

from queryframe.cache.hasher import hash_query, normalize_query
from queryframe.cache.memory import MemoryCache


class TestNormalizeQuery:
    def test_lowercase(self):
        assert normalize_query("SHOW ME Sales") == "sales"

    def test_strip_whitespace(self):
        assert normalize_query("  average  sales  ") == "average sales"

    def test_remove_fillers(self):
        assert normalize_query("please show me the average") == "the average"
        assert normalize_query("can you tell me the total") == "the total"

    def test_idempotent(self):
        q = "average sales by region"
        assert normalize_query(normalize_query(q)) == normalize_query(q)


class TestHashQuery:
    def test_consistent(self):
        h1 = hash_query("average sales", "abc123")
        h2 = hash_query("average sales", "abc123")
        assert h1 == h2

    def test_different_queries(self):
        h1 = hash_query("average sales", "abc123")
        h2 = hash_query("total revenue", "abc123")
        assert h1 != h2

    def test_different_schemas(self):
        h1 = hash_query("average sales", "abc123")
        h2 = hash_query("average sales", "def456")
        assert h1 != h2

    def test_normalized_same(self):
        h1 = hash_query("show me average sales", "abc")
        h2 = hash_query("SHOW ME average sales", "abc")
        assert h1 == h2


class TestMemoryCache:
    def test_set_and_get(self):
        cache = MemoryCache(max_size=10)
        cache.set("key1", data=42, code="result = 42")
        entry = cache.get("key1")
        assert entry is not None
        assert entry.data == 42
        assert entry.code == "result = 42"

    def test_miss_returns_none(self):
        cache = MemoryCache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = MemoryCache(max_size=2)
        cache.set("k1", data=1, code="1")
        cache.set("k2", data=2, code="2")
        cache.set("k3", data=3, code="3")

        assert cache.get("k1") is None  # evicted
        assert cache.get("k2") is not None
        assert cache.get("k3") is not None

    def test_ttl_expiry(self):
        cache = MemoryCache(max_size=10)
        cache.set("key1", data=42, code="42", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_clear(self):
        cache = MemoryCache()
        cache.set("k1", data=1, code="1")
        cache.set("k2", data=2, code="2")
        cache.clear()
        assert cache.size == 0
        assert cache.get("k1") is None

    def test_stats(self):
        cache = MemoryCache()
        cache.set("k1", data=1, code="1")
        cache.get("k1")  # hit
        cache.get("k2")  # miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_thread_safety(self):
        """Verify cache works with concurrent access."""
        import threading

        cache = MemoryCache(max_size=100)
        errors = []

        def writer(start: int):
            try:
                for i in range(start, start + 50):
                    cache.set(f"k{i}", data=i, code=str(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cache.size <= 100
