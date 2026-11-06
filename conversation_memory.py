"""
Conversation memory strategies for the generic RAG chatbot.

Mirrors the Strategy pattern already used for document loaders in
``vector_db_populator.py``: a small abstract base plus interchangeable concrete
implementations, selected by the consumer via a factory. A Null Object variant
lets callers treat "memory disabled" uniformly, without scattered conditionals.

The strategies are deliberately decoupled from the LLM SDK: ``SummaryBufferMemory``
receives a ``summarize_fn`` callable, so it can be unit-tested with a stub.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class ConversationMemoryStrategy(ABC):
    """Stores conversation turns and renders them for prompt injection."""

    @abstractmethod
    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Record one user/assistant exchange."""

    @abstractmethod
    def render(self) -> str:
        """Return a text block to prepend to the prompt ("" when there is nothing)."""

    @abstractmethod
    def clear(self) -> None:
        """Discard all retained state."""

    @property
    @abstractmethod
    def history(self) -> list[dict[str, str]]:
        """Return a copy of the retained recent turns."""


class NullMemory(ConversationMemoryStrategy):
    """Null Object: used when memory is disabled. Every operation is a no-op."""

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        return None

    def render(self) -> str:
        return ""

    def clear(self) -> None:
        return None

    @property
    def history(self) -> list[dict[str, str]]:
        return []


def _format_turns(entries: list[dict[str, str]]) -> str:
    """Render stored turn entries as labelled lines."""
    lines = []
    for entry in entries:
        role_label = "User" if entry["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {entry['content']}")
    return "\n".join(lines)


class BufferMemory(ConversationMemoryStrategy):
    """
    Sliding-window buffer (LangChain ``ConversationBufferMemory`` equivalent).

    Retains the most recent ``max_history_turns`` turns verbatim. Older turns
    fall off entirely ("context cliffing"). ``max_history_turns=None`` keeps an
    unbounded buffer.
    """

    def __init__(self, max_history_turns: int | None = None):
        self.max_history_turns = max_history_turns
        self._history: list[dict[str, str]] = []

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "model", "content": assistant_message})
        self._apply_cap()

    def _apply_cap(self) -> None:
        if self.max_history_turns is None:
            return
        max_entries = self.max_history_turns * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

    def render(self) -> str:
        if not self._history:
            return ""
        return "Previous conversation:\n" + _format_turns(self._history) + "\n"

    def clear(self) -> None:
        self._history = []

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)


class SummaryBufferMemory(ConversationMemoryStrategy):
    """
    Summary + buffer (LangChain ``ConversationSummaryBufferMemory`` equivalent).

    Keeps the most recent ``max_history_turns`` turns in a buffer; when a new
    turn pushes an older one out, the evicted turn is folded into a running
    summary via ``summarize_fn(existing_summary, new_lines) -> str``. This
    preserves long-term context across arbitrarily long conversations at the
    cost of an extra LLM call whenever the buffer overflows.

    ``summarize_fn`` is injected so this class stays decoupled from the LLM SDK
    and can be unit-tested with a stub. ``max_history_turns=None`` disables
    summarization (the buffer never overflows), making it behave like
    ``BufferMemory``.
    """

    def __init__(
        self,
        summarize_fn: Callable[[str, str], str],
        max_history_turns: int | None = None,
    ):
        self.summarize_fn = summarize_fn
        self.max_history_turns = max_history_turns
        self._summary: str = ""
        self._history: list[dict[str, str]] = []

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "model", "content": assistant_message})
        self._evict_overflow()

    def _evict_overflow(self) -> None:
        if self.max_history_turns is None:
            return
        max_entries = self.max_history_turns * 2
        if len(self._history) <= max_entries:
            return
        # Evict the oldest half of the buffer at once rather than one turn at a time.
        # This means summarization fires every max_history_turns // 2 new turns
        # instead of every turn, at the cost of fewer verbatim turns between evictions.
        keep_entries = max(2, max_entries // 2)  # always keep at least 1 turn
        overflow = self._history[:-keep_entries]
        self._history = self._history[-keep_entries:]
        new_lines = _format_turns(overflow)
        self._summary = self.summarize_fn(self._summary, new_lines)

    def render(self) -> str:
        parts = []
        if self._summary:
            parts.append(f"Summary of earlier conversation: {self._summary}")
        if self._history:
            parts.append("Previous conversation:\n" + _format_turns(self._history))
        if not parts:
            return ""
        return "\n\n".join(parts) + "\n"

    def clear(self) -> None:
        self._summary = ""
        self._history = []

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)
