"""Vector stores: pluggable semantic retrieval backends via factory."""

from rag_mcp_server.libs.vector_store.base import BaseVectorStore, VectorRecord

__all__ = ["BaseVectorStore", "VectorRecord"]
