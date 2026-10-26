"""Abstract contract for embedding providers.

The single test seam for embedding behaviour: everything above (vector store,
ingestion pipeline, retrieval) talks to this interface, never to a concrete
provider. See docs/spec-p9-embedding.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_mcp_server.core.validation import validate_positive_int

__all__ = ["BaseEmbedding", "validate_positive_int"]


class BaseEmbedding(ABC):
    """Maps a batch of texts to equal-dimension dense float vectors.

    The input contract is enforced once here for every provider: texts must be
    a list of str. Empty input returns [] (not an error) and never reaches the
    provider. Subclasses implement the abstract _embed_texts hook.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text.

        Empty input returns an empty list (same shape as input); it is not an
        error. Vectors are row-aligned with `texts`. The contract is list[str]:
        a bare string or a non-str element is a TypeError.
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts 必须是 list[str]，得到 {type(texts).__name__}")
        if not texts:
            return []
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("texts 的每个元素必须是 str")
        return self._embed_texts(texts)

    @abstractmethod
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a validated, non-empty list of strings into vectors."""

    @abstractmethod
    def dim(self) -> int:
        """Report the output dimension of this provider."""
