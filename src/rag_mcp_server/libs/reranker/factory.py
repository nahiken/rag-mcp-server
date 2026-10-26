"""Reranker provider factory — instantiate by provider string."""

from __future__ import annotations

from rag_mcp_server.libs.llm.base import BaseLLM
from rag_mcp_server.libs.reranker.base import NoneReranker
from rag_mcp_server.libs.reranker.llm_reranker import LLMReranker

KNOWN_PROVIDERS = ("none", "llm", "cross_encoder")


def create_reranker(provider: str, llm: BaseLLM | None = None, **kwargs):
    """Build a reranker backend by name.

    `none` returns a NoneReranker (keep original order — default); `llm` returns
    an LLMReranker and requires the `llm` to be injected (fail fast);
    `cross_encoder` returns a CrossEncoderReranker — the cross_encoder module is
    imported lazily (the branch) so constructing one never imports
    sentence-transformers (optional dependency; the model loads on first rerank).
    Unknown providers raise ValueError (fail fast).
    """
    if provider == "none":
        return NoneReranker(**kwargs)
    if provider == "llm":
        if llm is None:
            raise ValueError("create_reranker('llm') 需要注入 llm")
        return LLMReranker(llm=llm, **kwargs)
    if provider == "cross_encoder":
        from rag_mcp_server.libs.reranker.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(**kwargs)
    raise ValueError(f"未知 reranker provider: {provider}（可选：{', '.join(KNOWN_PROVIDERS)}）")
