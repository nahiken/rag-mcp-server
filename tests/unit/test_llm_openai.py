"""Tests for OpenAILLM — OpenAI / DeepSeek-compatible chat provider.

Red/green contract (docs/adr/0018-multi-provider.md, T2):
- requests POST {base_url}/chat/completions with {"model","messages","stream":false}
- Bearer Authorization header sent when api_key set; absent when not
- api_key priority: constructor value > OPENAI_API_KEY env (env fallback)
- strict response parsing, per-case, never a bare KeyError/TypeError escapes
- content:"" is a valid result; missing choices/message/content -> readable LLMError
- non-2xx / timeout / connect / invalid JSON -> LLMError; secrets never leak
- base_url default https://api.openai.com/v1; trailing slash stripped
"""

from __future__ import annotations

import json

import httpx
import pytest

from rag_mcp_server.core.errors import LLMError
from rag_mcp_server.libs.llm.openai import OpenAILLM


def make_transport(responses: list[dict]) -> httpx.MockTransport:
    """Return a MockTransport that replays canned JSON responses in order."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        resp = responses.pop(0)
        return httpx.Response(status_code=resp.get("status", 200), json=resp.get("json", resp))

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# ---------------------------------------------------------------------------
# Request construction + auth
# ---------------------------------------------------------------------------
def test_requests_chat_endpoint_with_expected_body():
    t = make_transport([{"json": chat_response("hi")}])
    llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test", transport=t)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hi"

    call = t.calls[0]  # type: ignore[attr-defined]
    assert call["url"].endswith("/chat/completions")
    assert call["body"] == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def test_sends_bearer_authorization_header_when_api_key_given():
    t = make_transport([{"json": chat_response("ok")}])
    llm = OpenAILLM(model="m", api_key="sk-secret", transport=t)
    llm.chat([{"role": "user", "content": "hi"}])
    assert t.calls[0]["headers"]["authorization"] == "Bearer sk-secret"  # type: ignore[attr-defined]


def test_no_authorization_header_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    t = make_transport([{"json": chat_response("ok")}])
    llm = OpenAILLM(model="m", api_key=None, transport=t)
    llm.chat([{"role": "user", "content": "hi"}])
    assert "authorization" not in {k.lower() for k in t.calls[0]["headers"]}  # type: ignore[attr-defined]


def test_default_base_url_is_openai_v1():
    t = make_transport([{"json": chat_response("ok")}])
    llm = OpenAILLM(model="m", transport=t)
    llm.chat([{"role": "user", "content": "hi"}])
    assert t.calls[0]["url"].startswith("https://api.openai.com/v1")  # type: ignore[attr-defined]


def test_base_url_trailing_slash_is_stripped():
    t = make_transport([{"json": chat_response("ok")}])
    llm = OpenAILLM(model="m", base_url="https://api.openai.com/v1/", transport=t)
    assert llm.base_url == "https://api.openai.com/v1"


def test_constructor_requires_model():
    with pytest.raises(ValueError):
        OpenAILLM(model="")
    with pytest.raises(ValueError):
        OpenAILLM(model=None)  # type: ignore[arg-type]


def test_constructor_requires_non_empty_base_url():
    with pytest.raises(ValueError):
        OpenAILLM(model="m", base_url="")


def test_constructor_defaults():
    llm = OpenAILLM(model="m")
    assert llm.base_url == "https://api.openai.com/v1"
    assert llm.timeout == 30.0


# ---------------------------------------------------------------------------
# api_key env fallback (constructor > env)
# ---------------------------------------------------------------------------
def test_api_key_reads_env_when_constructor_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    llm = OpenAILLM(model="m")
    assert llm.api_key == "sk-from-env"


def test_api_key_env_fallback_sends_auth_header(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    t = make_transport([{"json": chat_response("ok")}])
    llm = OpenAILLM(model="m", transport=t)
    llm.chat([{"role": "user", "content": "hi"}])
    assert t.calls[0]["headers"]["authorization"] == "Bearer sk-from-env"  # type: ignore[attr-defined]


def test_constructor_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    llm = OpenAILLM(model="m", api_key="sk-explicit")
    assert llm.api_key == "sk-explicit"


# ---------------------------------------------------------------------------
# Error mapping -> LLMError (timeout / connect / non-2xx / bad JSON)
# ---------------------------------------------------------------------------
def test_non_2xx_raises_llm_error_with_status_and_never_leaks_key():
    t = make_transport([{"status": 401, "json": {"error": {"message": "invalid key"}}}])
    llm = OpenAILLM(model="m", api_key="sk-bad", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "401" in str(ei.value)
    assert "sk-bad" not in str(ei.value)  # credentials never leak


def test_timeout_raises_llm_error_with_timeout_seconds():
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    llm = OpenAILLM(model="m", timeout=5.0, transport=httpx.MockTransport(timeout))
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert ei.value.timeout_seconds == 5.0


def test_connection_failure_raises_llm_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    llm = OpenAILLM(model="m", transport=httpx.MockTransport(boom))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "hi"}])


def test_redacts_userinfo_from_error_and_never_leaks_original():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("http://user:secret@api.openai.com")

    llm = OpenAILLM(
        model="m",
        base_url="http://user:secret@api.openai.com/v1",
        transport=httpx.MockTransport(boom),
    )
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "secret" not in str(ei.value)


def test_invalid_json_response_raises_llm_error():
    def bad_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    llm = OpenAILLM(model="m", transport=httpx.MockTransport(bad_json))
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "JSON" in str(ei.value)


def test_top_level_not_a_dict_raises_llm_error():
    t = make_transport([{"json": ["choices"]}])
    llm = OpenAILLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "dict" in str(ei.value)


# ---------------------------------------------------------------------------
# Response parsing — strict per-case (ADR 0018: 对齐 OllamaLLM)
# ---------------------------------------------------------------------------
def test_missing_choices_raises_readable_llm_error():
    t = make_transport([{"json": {"foo": "bar"}}])
    llm = OpenAILLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "choices" in str(ei.value)


def test_empty_choices_raises_readable_llm_error():
    t = make_transport([{"json": {"choices": []}}])
    llm = OpenAILLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "choices" in str(ei.value)


def test_choices_not_a_list_raises_readable_llm_error():
    t = make_transport([{"json": {"choices": "assistant"}}])
    llm = OpenAILLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "choices" in str(ei.value)


def test_choice_not_a_dict_raises_readable_llm_error():
    t = make_transport([{"json": {"choices": ["assistant"]}}])
    llm = OpenAILLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "choices[0]" in str(ei.value)


def test_missing_message_key_raises_readable_llm_error():
    t = make_transport([{"json": {"choices": [{"role": "assistant"}]}}])
    llm = OpenAILLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "message" in str(ei.value)


def test_missing_content_key_raises_readable_llm_error():
    t = make_transport([{"json": {"choices": [{"message": {"role": "assistant"}}]}}])
    llm = OpenAILLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "content" in str(ei.value)


def test_content_not_a_str_raises_readable_llm_error():
    t = make_transport([{"json": {"choices": [{"message": {"role": "assistant", "content": 3}}]}}])
    llm = OpenAILLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "content" in str(ei.value)


def test_empty_content_is_a_valid_result():
    t = make_transport([{"json": {"choices": [{"message": {"role": "assistant", "content": ""}}]}}])
    llm = OpenAILLM(model="m", transport=t)
    assert llm.chat([{"role": "user", "content": "hi"}]) == ""


def test_shares_base_contract():
    t = make_transport([{"json": chat_response("ok")}])
    llm = OpenAILLM(model="m", transport=t)
    with pytest.raises(ValueError):
        llm.chat([])
