"""Context window builder for conversation-aware prompts."""

from __future__ import annotations

from queryframe.llm.prompt.builder import ConversationTurn as PromptTurn
from queryframe.memory.conversation import ConversationMemory, ConversationTurn


def build_context(
    memory: ConversationMemory,
    schema_fingerprint: str,
    max_turns: int = 3,
) -> list[PromptTurn]:
    """Build conversation context for the prompt builder.

    Converts ConversationMemory turns into the format expected by
    the prompt builder. The most recent turn is kept in full detail,
    older turns are summarized.
    """
    recent = memory.get_recent(schema_fingerprint, max_turns=max_turns)

    if not recent:
        return []

    prompt_turns: list[PromptTurn] = []
    for i, turn in enumerate(recent):
        is_last = i == len(recent) - 1

        if is_last:
            # Most recent turn: include code snippet
            summary = f"{turn.result_summary} (code: {turn.code[:80]}...)"
        else:
            # Older turns: just the summary
            summary = turn.result_summary

        prompt_turns.append(PromptTurn(query=turn.query, summary=summary))

    return prompt_turns
