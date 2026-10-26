"""OpenAILLM — OpenAI / DeepSeek-compatible chat provider.

Talks to POST {base_url}/chat/completions with a Bearer auth header (API key).
The model is the OpenAI model name (e.g. "gpt-4o-mini") or a DeepSeek model
("deepseek-chat") — DeepSeek reuses OpenAILLM by pointing base_url at its
OpenAI-compatible endpoint, no new class (ADR 0018). Response parsing is as
strict as OllamaLLM: missing choices / message / content are judged case-by-case
and always surface as LLMError, never a bare KeyError/TypeError leaks to
consumers. content: "" is a valid result (empty value legal, missing key is a
protocol error). api_key priority: constructor value > OPENAI_API_KEY env
(ADR 0018 D26). Secrets never leak: the error's URL is sanitized (userinfo
stripped) and the raw RequestError string is never concatenated in. The
constructor accepts ``transport`` (mirroring OllamaLLM) so unit tests inject
httpx.MockTransport.

See docs/adr/0018-multi-provider.md.

Azure (ADR 0018 #6) is intentionally not implemented yet: its deployment URL
assembly ({resource}.openai.azure.com/openai/deployments/{deployment}) has the
most churn and is most likely to break, so it is deferred. To add it later,
subclass OpenAILLM and override the URL path + api_version query param — no
other change needed (the request/response contract is identical).
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

import httpx

from rag_mcp_server.core.errors import LLMError
from rag_mcp_server.core.validation import summary_body
from rag_mcp_server.libs.llm.base import BaseLLM

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
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


class OpenAILLM(BaseLLM):
    """Chat via an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
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
        self.timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,  # None -> real httpx transport
        )

    def _chat(self, messages: list[dict]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "stream": False}
        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise LLMError(
                "OpenAI 请求超时",
                provider="openai",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
                timeout_seconds=self.timeout,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"无法连接 OpenAI（{_sanitize_base_url(self.base_url)}）",
                provider="openai",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            ) from exc

        if not 200 <= resp.status_code < 300:
            raise LLMError(
                # the body is proxy-controlled and could echo credentials; scrub
                # it like OllamaLLM (shared summary_body, review FIX)
                f"OpenAI /chat/completions 返回 {resp.status_code}：{summary_body(resp.text)}",
                provider="openai",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(
                "OpenAI 响应不是有效 JSON",
                provider="openai",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            ) from exc

        # a valid JSON that is not an object (list / null / str) is a malformed
        # chat response; guard before .get so no bare AttributeError leaks
        if not isinstance(data, dict):
            raise LLMError(
                f"OpenAI 响应顶层必须是 dict（实际 {type(data).__name__}）",
                provider="openai",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )

        # strict per-case parsing (aligned with OllamaLLM): missing choices /
        # message / content each get a readable error, never a bare KeyError.
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError(
                f"OpenAI 响应缺 choices 字段（期望非空 list，实际 {type(choices).__name__}）",
                provider="openai",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMError(
                f"OpenAI choices[0] 必须是 dict（实际 {type(choice).__name__}）",
                provider="openai",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMError(
                f"OpenAI 响应缺 choices[0].message 字段（期望 dict，实际 {type(message).__name__}）",
                provider="openai",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError(
                f"OpenAI message.content 必须是 str（实际 {type(content).__name__}）",
                provider="openai",
                base_url=_sanitize_base_url(self.base_url),
                model=self.model,
            )
        # content: "" is a valid result — empty value, not a protocol error
        return content
