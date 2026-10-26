"""Deterministic LLM for unit tests and offline link checks.

Echoes the last message's content by default; an optional ``responses`` table
keyed on the last content overrides the reply (miss -> echo). Records every
call in ``calls``/``call_count`` so fallback/retry assertions read them
directly instead of wrapping with unittest.mock. Pure determinism: no network,
no random state. Accepts and ignores extra kwargs (aligned with FakeEmbedding).
"""

from __future__ import annotations

from rag_mcp_server.libs.llm.base import BaseLLM


class FakeLLM(BaseLLM):
    """Deterministic LLM; echoes last content unless responses table matches."""

    def __init__(self, responses: dict[str, str] | None = None, **kwargs) -> None:
        self.responses = responses or {}
        self.calls: list[list[dict]] = []
        self.call_count = 0

    def _chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        self.call_count += 1
        last_content = messages[-1]["content"]
        if last_content in self.responses:
            return self.responses[last_content]
        return last_content
