"""OllamaLLM — real LLM provider backed by a local Ollama server.

Talks to POST {base_url}/api/chat ({"model", "messages", "stream": false}).
messages are passed through verbatim (extra keys like ``images`` are kept —
the P10.6 Vision extension point). No retries (one failure is a failure, so
fallback timing stays predictable); timeout defaults to 30.0 single value.
Every provider failure (timeout / connect / non-2xx / malformed response)
maps to LLMError — never a bare KeyError/TypeError leaks to consumers.
Response parsing is strict per-case, with readable "expected X got Y" messages;
content: "" is a valid result (empty value is legal, missing key is a protocol
error). The constructor accepts ``transport`` (mirroring OllamaEmbedding) so
unit tests inject httpx.MockTransport. Secrets never leak: the error's URL is
sanitized (userinfo stripped) and the raw RequestError string is never
concatenated in.

See docs/spec-p10-llm.md and ADR 0008.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import httpx

from rag_mcp_server.core.errors import LLMError
from rag_mcp_server.core.validation import summary_body
from rag_mcp_server.libs.llm.base import BaseLLM

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11451"
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


# Credential-scrubbed, flattened, truncated summary of a non-2xx response body
# — shared via core.validation.summary_body so LLM/embedding providers scrub
# identically (the body is proxy-controlled and could echo credentials).
_MAX_BODY_CHARS = 200


class OllamaLLM(BaseLLM):
    """Chat via a local Ollama server (/api/chat)."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.Transport | None = None,
    ) -> None:
        if not model or not isinstance(model, str):
            raise ValueError("model 不能为空")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url 不能为空且必须是字符串")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,  # None -> real httpx transport
        )

    def _chat(self, messages: list[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(
                "Ollama 请求超时",
                provider="ollama",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
                timeout_seconds=self.timeout,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"无法连接 Ollama（{_sanitize_base_url(self.base_url)}）",
                provider="ollama",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            ) from exc

        if not 200 <= resp.status_code < 300:
            raise LLMError(
                f"Ollama /api/chat 返回 {resp.status_code}：{summary_body(resp.text)}",
                provider="ollama",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(
                "Ollama 响应不是有效 JSON",
                provider="ollama",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            ) from exc

        # a valid JSON that is not an object (list / null / str) is a malformed
        # chat response; guard before .get so no bare AttributeError leaks
        if not isinstance(data, dict):
            raise LLMError(
                f"Ollama 响应顶层必须是 dict（实际 {type(data).__name__}）",
                provider="ollama",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )

        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMError(
                f"Ollama 响应缺 message 字段（期望 dict，实际 {type(message).__name__}）",
                provider="ollama",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError(
                f"Ollama message.content 必须是 str（实际 {type(content).__name__}）",
                provider="ollama",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )
        # content: "" is a valid result — empty value, not a protocol error
        return content
