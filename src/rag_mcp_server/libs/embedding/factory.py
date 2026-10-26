"""Embedding provider factory — instantiate by config provider string."""

from __future__ import annotations

from rag_mcp_server.libs.embedding.base import BaseEmbedding
from rag_mcp_server.libs.embedding.fake import FakeEmbedding

KNOWN_PROVIDERS = ("fake", "ollama", "openai")


def create_embedding(provider: str, **kwargs: object) -> BaseEmbedding:
    """Build an embedding provider by name.

    `fake` returns a FakeEmbedding (unit tests / offline); `ollama` returns an
    OllamaEmbedding (real local backend); `openai` returns an OpenAIEmbedding
    (ADR 0018 — OpenAI-compatible /embeddings endpoint). Unknown providers raise
    ValueError (fail fast). Real provider imports are lazy so unit tests stay
    free of httpx.
    """
    if provider == "fake":
        return FakeEmbedding(**kwargs)
    if provider == "ollama":
        # imported lazily to keep unit tests free of the ollama dependency
        from rag_mcp_server.libs.embedding.ollama import OllamaEmbedding

        return OllamaEmbedding(**kwargs)
    if provider == "openai":
        # imported lazily to keep unit tests free of httpx
        from rag_mcp_server.libs.embedding.openai import OpenAIEmbedding

        return OpenAIEmbedding(**kwargs)
    raise ValueError(f"未知 embedding provider: {provider}（可选：{', '.join(KNOWN_PROVIDERS)}）")
