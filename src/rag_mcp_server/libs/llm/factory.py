"""LLM provider factory — instantiate by config provider string."""

from __future__ import annotations

from rag_mcp_server.libs.llm.base import BaseLLM
from rag_mcp_server.libs.llm.fake import FakeLLM

KNOWN_PROVIDERS = ("fake", "ollama", "openai", "deepseek")


def create_llm(provider: str, **kwargs) -> BaseLLM:
    """Build an LLM provider by name.

    `fake` returns a FakeLLM (unit tests / offline); `ollama` returns an
    OllamaLLM (real local backend); `openai` / `deepseek` both return OpenAILLM
    (ADR 0018 — DeepSeek is OpenAI-compatible, the caller passes its base_url,
    zero new classes). Unknown providers raise ValueError (fail fast). Real
    provider imports are lazy so unit tests stay free of httpx.
    """
    if provider == "fake":
        return FakeLLM(**kwargs)
    if provider == "ollama":
        # imported lazily to keep unit tests free of the ollama dependency
        from rag_mcp_server.libs.llm.ollama import OllamaLLM

        return OllamaLLM(**kwargs)
    if provider in ("openai", "deepseek"):
        # imported lazily to keep unit tests free of httpx (ADR 0018: deepseek
        # reuses OpenAILLM — base_url comes from the caller / settings)
        from rag_mcp_server.libs.llm.openai import OpenAILLM

        return OpenAILLM(**kwargs)
    raise ValueError(f"未知 LLM provider: {provider}（可选：{', '.join(KNOWN_PROVIDERS)}）")
