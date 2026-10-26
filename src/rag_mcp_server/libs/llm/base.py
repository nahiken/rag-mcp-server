"""Abstract contract for LLM providers.

The single test seam for LLM behaviour: Transform enhancement (P10.4), LLM
Reranker (P10.3) and evaluation (P10.7) all talk to this interface, never to a
concrete provider. See docs/spec-p10-llm.md and ADR 0008.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["BaseLLM"]


class BaseLLM(ABC):
    """Chat with a model: messages -> text.

    The input contract is enforced once here for every provider: messages must
    be a non-empty list of dicts, each with str ``role`` and str ``content``.
    Empty messages raise ValueError (a call with no messages is a bug, unlike
    embedding where empty input is valid); type/structure problems raise
    TypeError. Extra keys are allowed (vision ``images`` extension point) and
    role values are NOT whitelisted — the backend owns role naming, typos
    surface as its 4xx -> provider error. Subclasses implement the abstract
    ``_chat`` hook.
    """

    def chat(self, messages: list[dict]) -> str:
        """Return the model's reply text for the given messages."""
        if not isinstance(messages, list):
            raise TypeError(f"messages 必须是 list[dict]，得到 {type(messages).__name__}")
        if not messages:
            raise ValueError("messages 不能为空")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(f"messages 第 {i} 个元素必须是 dict，得到 {type(msg).__name__}")
            for key in ("role", "content"):
                value = msg.get(key)
                if not isinstance(value, str):
                    raise TypeError(f"messages 第 {i} 个元素的 {key} 必须是 str，得到 {type(value).__name__}")
        return self._chat(messages)

    @abstractmethod
    def _chat(self, messages: list[dict]) -> str:
        """Produce the reply for a validated, non-empty list of messages."""
