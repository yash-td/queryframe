"""Conversation history manager for follow-up queries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in the conversation."""

    query: str
    code: str
    result_summary: str
    timestamp: float


class ConversationMemory:
    """Stores conversation history scoped by DataFrame schema fingerprint."""

    def __init__(self, max_turns: int = 20) -> None:
        self._history: dict[str, list[ConversationTurn]] = {}
        self._max_turns = max_turns

    def add_turn(
        self,
        schema_fingerprint: str,
        query: str,
        code: str,
        result_summary: str,
    ) -> None:
        """Add a conversation turn for a specific DataFrame."""
        if schema_fingerprint not in self._history:
            self._history[schema_fingerprint] = []

        turn = ConversationTurn(
            query=query,
            code=code,
            result_summary=result_summary,
            timestamp=time.time(),
        )

        turns = self._history[schema_fingerprint]
        turns.append(turn)

        # Trim to max_turns
        if len(turns) > self._max_turns:
            self._history[schema_fingerprint] = turns[-self._max_turns :]

    def get_recent(
        self,
        schema_fingerprint: str,
        max_turns: int = 3,
    ) -> list[ConversationTurn]:
        """Get the most recent conversation turns for a DataFrame."""
        turns = self._history.get(schema_fingerprint, [])
        return turns[-max_turns:]

    def clear(self, schema_fingerprint: str | None = None) -> None:
        """Clear conversation history."""
        if schema_fingerprint:
            self._history.pop(schema_fingerprint, None)
        else:
            self._history.clear()

    @property
    def active_conversations(self) -> int:
        """Number of active DataFrame conversations."""
        return len(self._history)
