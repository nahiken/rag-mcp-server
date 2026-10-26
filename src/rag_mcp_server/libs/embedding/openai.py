"""OpenAIEmbedding — OpenAI-compatible embeddings via {base_url}/embeddings.

Symmetric to OllamaEmbedding (BaseEmbedding._embed_texts + dim hooks, dim lazy
probe + cache + validation, EmbeddingError mapping, no retries). Model is the
OpenAI embedding model name (e.g. "text-embedding-3-small"). api_key priority:
constructor value > OPENAI_API_KEY env (ADR 0018 D26). All provider failures
(connect / non-2xx / invalid JSON / malformed payload / dim mismatch) map to
EmbeddingError — never a bare KeyError/TypeError leaks to consumers. The
constructor accepts ``transport`` so unit tests inject httpx.MockTransport.

See docs/adr/0018-multi-provider.md.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

import httpx

from rag_mcp_server.core.errors import EmbeddingError
from rag_mcp_server.core.validation import summary_body
from rag_mcp_server.libs.embedding.base import BaseEmbedding, validate_positive_int

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TIMEOUT = 30.0


def _sanitize_base_url(base_url: str) -> str:
    """Strip any userinfo (user:pass@) from a URL, keeping scheme/host/path."""
    parts = urlsplit(base_url)
    if parts.username or parts.password:
        netloc = parts.hostname or ""
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return base_url


class OpenAIEmbedding(BaseEmbedding):
    """Embeddings via an OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
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
        # api_key priority: constructor value > OPENAI_API_KEY env (ADR 0018 D26)
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
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
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "input": texts}
        try:
            resp = self._client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise EmbeddingError(f"无法连接 OpenAI（{_sanitize_base_url(self.base_url)}）：{exc}") from exc

        # any non-2xx status is a provider error (spec: 非 2xx); the body is
        # proxy-controlled and could echo credentials, so scrub it like the LLM
        # path (shared summary_body, review FIX)
        if not 200 <= resp.status_code < 300:
            raise EmbeddingError(
                f"OpenAI /embeddings 返回 {resp.status_code}：{summary_body(resp.text)}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise EmbeddingError("OpenAI 响应不是有效 JSON") from exc

        # a valid JSON that is not an object (list / null / str) is malformed
        if not isinstance(data, dict):
            raise EmbeddingError(
                f"OpenAI 响应顶层必须是 dict（实际 {type(data).__name__}）"
            )

        if "error" in data:
            raise EmbeddingError(f"OpenAI 返回错误：{data['error']}")

        items = data.get("data")
        if not isinstance(items, list):
            raise EmbeddingError(
                f"OpenAI 响应缺 data 字段（期望 list，实际 {type(items).__name__}）"
            )
        if len(items) != len(texts):
            raise EmbeddingError(
                f"OpenAI 返回 {len(items)} 个向量，期望 {len(texts)}（条数不匹配）"
            )
        # contract is list[list[float]]: a scalar/dict row or a non-float/NaN
        # element is a broken backend response, reported as EmbeddingError
        # rather than leaked as a raw TypeError or silently passed through.
        embeddings: list[list[float]] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise EmbeddingError(
                    f"OpenAI 返回第 {i} 个元素不是 dict，得到 {type(item).__name__}"
                )
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise EmbeddingError(
                    f"OpenAI 返回第 {i} 个向量不是 list，得到 {type(vec).__name__}"
                )
            for j, x in enumerate(vec):
                if not isinstance(x, float) or x != x:
                    raise EmbeddingError(
                        f"OpenAI 返回第 {i} 个向量第 {j} 个元素不是有效 float，得到 {x!r}"
                    )
            embeddings.append(vec)
        return embeddings
