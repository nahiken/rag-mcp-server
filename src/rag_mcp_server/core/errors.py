"""Custom errors raised by the RAG MCP server."""


class RagMcpError(Exception):
    """Base class for all expected RAG MCP server errors."""


class DocumentLoadError(RagMcpError):
    """Raised when an importable document cannot be loaded."""


class ChunkStoreError(RagMcpError):
    """Raised when chunks cannot be read from or written to the store."""


class QueryError(RagMcpError):
    """Raised when a query cannot be executed against the store."""


class ConfigError(RagMcpError):
    """Raised when configuration is invalid (e.g. unknown retrieval strategy)."""


class EmbeddingError(RagMcpError):
    """Raised when an embedding provider violates its contract (e.g. dimension)."""


class LLMError(RagMcpError):
    """Raised when an LLM provider violates its contract (timeout, connect, non-2xx, malformed).

    Parallel to EmbeddingError: consumers distinguish "embedding failed" from
    "LLM failed" by type. `timeout_seconds` is only populated when the failure
    was a timeout; provider/base_url(model) are always present and sanitized
    (base_url userinfo is stripped by the provider).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        base_url: str = "",
        model: str = "",
        timeout_seconds: float | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        parts = [message]
        if provider:
            parts.append(f"provider={provider}")
        if base_url:
            parts.append(f"base_url={base_url}")
        if model:
            parts.append(f"model={model}")
        if timeout_seconds is not None:
            parts.append(f"timeout_seconds={timeout_seconds}")
        super().__init__("；".join(parts))


class RerankError(RagMcpError):
    """Raised when the reranking layer fails (LLM call or output parse).

    Parallel to LLMError: the orchestration wrapper catches this single type to
    fall back to the original order. Bare (no structured fields) — field-level
    diagnostics live on the underlying LLMError via the cause chain.
    """


class VectorStoreError(RagMcpError):
    """Raised when the vector store violates its contract (e.g. missing metadata keys)."""

