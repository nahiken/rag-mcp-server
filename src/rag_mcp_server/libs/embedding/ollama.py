"""OllamaEmbedding — real embedding provider backed by a local Ollama server.

Talks to POST {base_url}/api/embed (verified on Ollama 0.32.15; the /api/embeddings
endpoint is deprecated/absent). Uses the model's default output dimension; no
`dimensions` field is sent. Dimension is discovered lazily on first encode and
cached; later calls validate against the cache and raise EmbeddingError on
mismatch. All provider failures map to EmbeddingError (no retries).

See docs/spec-p9-ollama-embedding.md.
"""

from __future__ import annotations

import math

import httpx

from rag_mcp_server.core.errors import EmbeddingError
from rag_mcp_server.libs.embedding.base import BaseEmbedding, validate_positive_int

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11451"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TIMEOUT = 30.0


class OllamaEmbedding(BaseEmbedding):
    """Embeddings via a local Ollama model (nomic-embed-text)."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.Transport | None = None,
    ) -> None:
        if not model or not isinstance(model, str):
            raise ValueError("model 不能为空")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url 不能为空且必须是字符串")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = validate_positive_int("batch_size", batch_size)
        self.timeout = timeout
        self._dim: int | None = None
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,  # None -> real httpx transport
        )

    def dim(self) -> int:
        """Report the output dimension, probing once if not yet known."""
        if self._dim is None:
            probe = self._encode_batch([""])
            self._dim = len(probe[0])
        return self._dim

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        # dimension is cached after the first real encode (US#9); subsequent
        # batches are validated against the cache. dim() standalone probes.
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_vecs = self._encode_batch(batch)
            if self._dim is None:
                self._dim = len(batch_vecs[0])
            for vec in batch_vecs:
                if len(vec) != self._dim:
                    raise EmbeddingError(
                        f"Embedding 维度不一致：期望 {self._dim}，实际 {len(vec)}"
                    )
                vectors.append(vec)
        return vectors

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts, "truncate": False}
        try:
            resp = self._client.post(f"{self.base_url}/api/embed", json=payload)
        except httpx.RequestError as exc:
            raise EmbeddingError(f"无法连接 Ollama（{self.base_url}）：{exc}") from exc

        # any non-2xx status is a provider error (spec: 非 2xx)
        if not 200 <= resp.status_code < 300:
            raise EmbeddingError(f"Ollama /api/embed 返回 {resp.status_code}：{resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise EmbeddingError("Ollama 响应不是有效 JSON") from exc

        if "error" in data:
            raise EmbeddingError(f"Ollama 返回错误：{data['error']}")

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError("Ollama 响应缺少 embeddings 字段")
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Ollama 返回 {len(embeddings)} 个向量，期望 {len(texts)}（条数不匹配）"
            )
        # contract is list[list[float]]: a scalar/dict row or a non-float/NaN
        # element is a broken backend response, reported as EmbeddingError
        # rather than leaked as a raw TypeError or silently passed through.
        for i, vec in enumerate(embeddings):
            if not isinstance(vec, list):
                raise EmbeddingError(
                    f"Ollama 返回第 {i} 个向量不是 list，得到 {type(vec).__name__}"
                    "（条数不匹配或协议异常）"
                )
            for j, x in enumerate(vec):
                if not isinstance(x, float) or x != x:
                    raise EmbeddingError(
                        f"Ollama 返回第 {i} 个向量第 {j} 个元素不是有效 float，得到 {x!r}"
                    )
        return embeddings
