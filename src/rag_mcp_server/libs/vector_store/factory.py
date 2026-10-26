"""Vector store factory — instantiate by config provider string."""

from __future__ import annotations

from rag_mcp_server.libs.vector_store.base import BaseVectorStore

KNOWN_PROVIDERS = ("chroma",)


def create_vector_store(
    provider: str,
    persist_dir: str | None,
    collection: str,
) -> BaseVectorStore:
    """Build a vector store by name.

    `chroma` returns a ChromaVectorStore (embedded Chroma; persist_dir None ->
    ephemeral in-memory). Unknown providers raise ValueError (fail fast).
    The chroma import is lazy so unit tests of other modules stay free of the
    chromadb dependency surface.
    """
    if provider == "chroma":
        # imported lazily so unrelated unit suites don't import chromadb
        from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

        return ChromaVectorStore(persist_dir=persist_dir, collection=collection)
    raise ValueError(f"未知 vector store provider: {provider}（可选：{', '.join(KNOWN_PROVIDERS)}）")
