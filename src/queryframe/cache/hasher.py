"""Query and schema fingerprinting for cache keys."""

from __future__ import annotations

import re

import xxhash


def normalize_query(query: str) -> str:
    """Normalize a query for consistent cache key generation.

    - Lowercase
    - Strip extra whitespace
    - Remove filler words
    """
    q = query.lower().strip()
    q = re.sub(r"\s+", " ", q)

    # Remove common filler words that don't change query meaning
    fillers = {"please", "can you", "could you", "show me", "give me", "tell me",
               "i want to", "i'd like to", "i would like to", "let me see"}
    for filler in fillers:
        q = q.replace(filler, "")

    return q.strip()


def hash_query(query: str, schema_fingerprint: str) -> str:
    """Generate a fast hash key from a normalized query + schema fingerprint."""
    normalized = normalize_query(query)
    raw = f"{normalized}|{schema_fingerprint}"
    return xxhash.xxh64(raw.encode()).hexdigest()
