"""Reranker providers: query + results -> reordered results, pluggable.

Top-level convenience exports for consumers (rag_service.query, P10.7 eval).
RerankError is defined in core/errors and re-exported here so consumers need
not know where it lives.
"""

from __future__ import annotations

from rag_mcp_server.core.errors import RerankError
from rag_mcp_server.libs.reranker.base import BaseReranker, NoneReranker
from rag_mcp_server.libs.reranker.factory import create_reranker
from rag_mcp_server.libs.reranker.llm_reranker import LLMReranker
from rag_mcp_server.libs.reranker.orchestrator import QueryReranker

__all__ = [
    "BaseReranker",
    "NoneReranker",
    "LLMReranker",
    "QueryReranker",
    "create_reranker",
    "RerankError",
]
