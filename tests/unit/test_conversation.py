"""Tests for conversation memory."""

import pytest

from queryframe.memory.conversation import ConversationMemory


class TestConversationMemory:
    def test_add_and_get(self):
        mem = ConversationMemory()
        mem.add_turn("fp1", "what is avg?", "result = df.mean()", "42")
        turns = mem.get_recent("fp1")
        assert len(turns) == 1
        assert turns[0].query == "what is avg?"

    def test_multiple_turns(self):
        mem = ConversationMemory()
        mem.add_turn("fp1", "q1", "c1", "s1")
        mem.add_turn("fp1", "q2", "c2", "s2")
        mem.add_turn("fp1", "q3", "c3", "s3")
        turns = mem.get_recent("fp1", max_turns=2)
        assert len(turns) == 2
        assert turns[0].query == "q2"
        assert turns[1].query == "q3"

    def test_separate_fingerprints(self):
        mem = ConversationMemory()
        mem.add_turn("fp1", "q1", "c1", "s1")
        mem.add_turn("fp2", "q2", "c2", "s2")
        assert len(mem.get_recent("fp1")) == 1
        assert len(mem.get_recent("fp2")) == 1

    def test_max_turns_limit(self):
        mem = ConversationMemory(max_turns=3)
        for i in range(10):
            mem.add_turn("fp1", f"q{i}", f"c{i}", f"s{i}")
        turns = mem.get_recent("fp1", max_turns=10)
        assert len(turns) == 3

    def test_clear_specific(self):
        mem = ConversationMemory()
        mem.add_turn("fp1", "q1", "c1", "s1")
        mem.add_turn("fp2", "q2", "c2", "s2")
        mem.clear("fp1")
        assert len(mem.get_recent("fp1")) == 0
        assert len(mem.get_recent("fp2")) == 1

    def test_clear_all(self):
        mem = ConversationMemory()
        mem.add_turn("fp1", "q1", "c1", "s1")
        mem.add_turn("fp2", "q2", "c2", "s2")
        mem.clear()
        assert mem.active_conversations == 0

    def test_empty_get(self):
        mem = ConversationMemory()
        assert mem.get_recent("nonexistent") == []
