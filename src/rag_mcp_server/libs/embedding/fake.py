"""Deterministic embedding for unit tests and offline smoke checks.

Uses sha256(text) mapped into [-1, 1] to fill a fixed dimension. Pure
determinism: no random state, no seed, standard-library hashlib only. Same text
always yields the identical vector; different texts differ in practice.
"""

from __future__ import annotations

import hashlib

from rag_mcp_server.libs.embedding.base import BaseEmbedding, validate_positive_int


class FakeEmbedding(BaseEmbedding):
    """Deterministic embedding; dim defaults to nomic-embed-text's 768."""

    def __init__(self, dim: int = 768) -> None:
        self._dim = validate_positive_int("dim", dim)

    def dim(self) -> int:
        return self._dim

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        if self._dim <= len(digest):
            # take the first dim bytes directly when enough entropy
            return [self._scale(b) for b in digest[: self._dim]]
        # dim larger than digest length: expand entropy by hashing the text
        # with per-index salt, so a default 768-dim vector is not 24 copies of
        # the same 32 bytes (which would make ~96% of dims fully correlated).
        extended = bytearray()
        index = 0
        while len(extended) < self._dim:
            extended += hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
            index += 1
        return [self._scale(b) for b in extended[: self._dim]]

    @staticmethod
    def _scale(byte: int) -> float:
        """Map a byte [0, 255] into [-1.0, 1.0]."""
        return (byte / 255.0) * 2.0 - 1.0
