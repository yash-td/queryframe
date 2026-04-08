"""Tests for DiskCache."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from queryframe.cache.disk import DiskCache


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    """A DiskCache instance using a temporary directory."""
    db_path = str(tmp_path / "test_cache.db")
    return DiskCache(db_path=db_path, default_ttl=3600.0)


class TestSetGet:
    def test_round_trip(self, cache: DiskCache) -> None:
        cache.set("k1", data={"a": 1}, code="result = 1", explanation="test")
        entry = cache.get("k1")
        assert entry is not None
        assert entry.data == {"a": 1}
        assert entry.code == "result = 1"
        assert entry.explanation == "test"
        assert entry.chart_type is None

    def test_round_trip_with_chart_type(self, cache: DiskCache) -> None:
        cache.set("k2", data=[1, 2, 3], code="c", chart_type="bar", explanation="e")
        entry = cache.get("k2")
        assert entry is not None
        assert entry.chart_type == "bar"
        assert entry.data == [1, 2, 3]

    def test_get_missing_key(self, cache: DiskCache) -> None:
        assert cache.get("nonexistent") is None

    def test_overwrite_key(self, cache: DiskCache) -> None:
        cache.set("k", data=1, code="a")
        cache.set("k", data=2, code="b")
        entry = cache.get("k")
        assert entry is not None
        assert entry.data == 2
        assert entry.code == "b"


class TestTTL:
    def test_expired_entry_returns_none(self, cache: DiskCache) -> None:
        cache.set("k", data=1, code="c", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("k") is None

    def test_non_expired_entry_returns_data(self, cache: DiskCache) -> None:
        cache.set("k", data=42, code="c", ttl=3600.0)
        entry = cache.get("k")
        assert entry is not None
        assert entry.data == 42


class TestDelete:
    def test_delete_existing(self, cache: DiskCache) -> None:
        cache.set("k", data=1, code="c")
        cache.delete("k")
        assert cache.get("k") is None

    def test_delete_nonexistent_no_error(self, cache: DiskCache) -> None:
        cache.delete("nope")  # should not raise


class TestClear:
    def test_clear_removes_all(self, cache: DiskCache) -> None:
        cache.set("a", data=1, code="c")
        cache.set("b", data=2, code="c")
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None
        assert cache.get("b") is None


class TestCleanupExpired:
    def test_cleanup_removes_expired(self, cache: DiskCache) -> None:
        cache.set("expired", data=1, code="c", ttl=0.01)
        cache.set("fresh", data=2, code="c", ttl=3600.0)
        time.sleep(0.02)

        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.get("expired") is None
        assert cache.get("fresh") is not None

    def test_cleanup_nothing_expired(self, cache: DiskCache) -> None:
        cache.set("a", data=1, code="c", ttl=3600.0)
        removed = cache.cleanup_expired()
        assert removed == 0


class TestSize:
    def test_empty_cache(self, cache: DiskCache) -> None:
        assert cache.size == 0

    def test_size_after_inserts(self, cache: DiskCache) -> None:
        cache.set("a", data=1, code="c")
        cache.set("b", data=2, code="c")
        assert cache.size == 2

    def test_size_after_delete(self, cache: DiskCache) -> None:
        cache.set("a", data=1, code="c")
        cache.delete("a")
        assert cache.size == 0


class TestUnpicklableData:
    def test_unpicklable_set_skips_gracefully(self, cache: DiskCache) -> None:
        """Data that cannot be pickled should be skipped without raising."""
        unpicklable = lambda: None  # noqa: E731
        cache.set("k", data=unpicklable, code="c")
        # Entry should not be stored
        assert cache.get("k") is None

    def test_corrupt_pickle_returns_none(self, cache: DiskCache) -> None:
        """If stored pickle is corrupt, get should return None and delete entry."""
        cache.set("k", data=42, code="c")
        # Corrupt the pickle data directly
        with cache._connect() as conn:
            conn.execute(
                "UPDATE cache SET data_pickle = ? WHERE key = ?",
                (b"corrupt_data", "k"),
            )
        assert cache.get("k") is None
        # Entry should be cleaned up
        assert cache.size == 0
