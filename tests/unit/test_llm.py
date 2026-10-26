"""Tests for the pluggable LLM layer — BaseLLM contract + Fake/Ollama providers + factory.

Red/green contract (docs/spec-p10-llm.md, ticket 02, ADR 0008):
- BaseLLM.chat(messages) -> str, synchronous; single seam for all consumers
- contract validated once in the base class: non-empty list[dict]; each element
  role/content must be str; extra keys allowed (vision images extension); NO
  role whitelist (Ollama is the authority; typos surface via backend 4xx)
- empty messages -> ValueError; type/structure problems -> TypeError
- LLMError parallel to EmbeddingError (both RagMcpError); fields
  provider/base_url(sanitized)/model/timeout_seconds(only on timeout), all in str()
- OllamaLLM: POST {base_url}/api/chat {"model","messages","stream":false};
  messages passed through verbatim; no retry; timeout default 30.0; malformed
  response -> readable LLMError (never a bare KeyError/TypeError); content:""
  is a valid result; transport injection for MockTransport; no-secret two gates
- FakeLLM: deterministic; echoes last content; responses table matched on last
  content (miss -> echo); calls + call_count; accepts **kwargs
- factory create_llm: unknown provider -> ValueError; ollama lazy import

Prior art: tests/unit/test_ollama_embedding.py (MockTransport + transport.calls),
tests/unit/test_embedding.py (interface external behavior).
"""

from __future__ import annotations

import json

import httpx
import pytest

from rag_mcp_server.core.errors import LLMError, RagMcpError
from rag_mcp_server.libs.llm.base import BaseLLM
from rag_mcp_server.libs.llm.fake import FakeLLM
from rag_mcp_server.libs.llm.ollama import OllamaLLM


# ---------------------------------------------------------------------------
# TestLLMError — domain error fields & str()
# ---------------------------------------------------------------------------
def test_llm_error_is_rag_mcp_error():
    # ADR 0008 pt.3: LLMError(RagMcpError), parallel to EmbeddingError —
    # consumers distinguish failure source by type. RagMcpError(Exception)
    # (test_core_types.py) transitively guarantees LLMError is an Exception.
    assert issubclass(LLMError, RagMcpError)


def test_llm_error_fields_present_in_str():
    err = LLMError("boom", provider="ollama", base_url="http://host", model="m")
    text = str(err)
    assert "boom" in text
    assert "ollama" in text
    assert "m" in text


def test_llm_error_omits_timeout_seconds_when_not_timeout():
    err = LLMError("boom", provider="ollama", base_url="http://host", model="m")
    assert "timeout" not in str(err).lower()
    assert err.timeout_seconds is None


def test_llm_error_includes_timeout_seconds_when_timeout():
    err = LLMError("boom", provider="ollama", base_url="http://host", model="m", timeout_seconds=30.0)
    assert err.timeout_seconds == 30.0
    assert "30" in str(err)


# ---------------------------------------------------------------------------
# TestBaseLLMContract — contract validated once in the base class
# ---------------------------------------------------------------------------
class EchoLLM(BaseLLM):
    def _chat(self, messages: list[dict]) -> str:
        return str(messages)


def test_empty_messages_raise_value_error():
    with pytest.raises(ValueError):
        EchoLLM().chat([])


def test_messages_not_a_list_raises_type_error():
    with pytest.raises(TypeError):
        EchoLLM().chat("hello")


def test_element_not_a_dict_raises_type_error():
    with pytest.raises(TypeError):
        EchoLLM().chat(["hello"])


def test_missing_role_key_raises_type_error():
    with pytest.raises(TypeError):
        EchoLLM().chat([{"content": "hi"}])


def test_missing_content_key_raises_type_error():
    with pytest.raises(TypeError):
        EchoLLM().chat([{"role": "user"}])


def test_non_string_role_raises_type_error():
    with pytest.raises(TypeError):
        EchoLLM().chat([{"role": 1, "content": "hi"}])


def test_non_string_content_raises_type_error():
    with pytest.raises(TypeError):
        EchoLLM().chat([{"role": "user", "content": 3}])


def test_extra_keys_allowed_and_custom_role_passes_validation():
    # no role whitelist: a misspelled role passes base validation (the backend
    # surfaces it); extra keys (vision images extension) are allowed
    out = EchoLLM().chat([{"role": "sytem", "content": "prompt", "images": ["a.png"]}])
    assert out.startswith("[{'role': 'sytem'")


def test_valid_messages_reach_provider_hook():
    out = EchoLLM().chat([{"role": "user", "content": "hi"}])
    assert out == "[{'role': 'user', 'content': 'hi'}]"


# ---------------------------------------------------------------------------
# TestFakeLLM — deterministic echo + responses table + call recording
# ---------------------------------------------------------------------------
def test_fake_llm_echoes_last_content():
    llm = FakeLLM()
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hi"


def test_fake_llm_deterministic_same_messages_same_result():
    llm = FakeLLM()
    msgs = [{"role": "system", "content": "p"}, {"role": "user", "content": "hi"}]
    assert llm.chat(msgs) == llm.chat(msgs)


def test_fake_llm_echoes_last_content_of_multiple_messages():
    llm = FakeLLM()
    msgs = [{"role": "system", "content": "p"}, {"role": "user", "content": "the real question"}]
    assert llm.chat(msgs) == "the real question"


def test_fake_llm_responses_table_matches_last_content():
    llm = FakeLLM(responses={"hi": "hello back"})
    msgs = [{"role": "system", "content": "p"}, {"role": "user", "content": "hi"}]
    assert llm.chat(msgs) == "hello back"


def test_fake_llm_responses_table_miss_falls_back_to_echo():
    llm = FakeLLM(responses={"hi": "hello back"})
    assert llm.chat([{"role": "user", "content": "not in table"}]) == "not in table"


def test_fake_llm_records_calls_and_call_count():
    llm = FakeLLM()
    msgs = [{"role": "user", "content": "a"}]
    llm.chat(msgs)
    llm.chat([{"role": "user", "content": "b"}])
    assert llm.call_count == 2
    assert llm.calls == [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ]


def test_fake_llm_ignores_extra_kwargs():
    llm = FakeLLM(base_url="ignored", model="ignored")
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hi"


def test_fake_llm_shares_base_contract():
    # empty messages -> ValueError (contract enforced in the base class)
    with pytest.raises(ValueError):
        FakeLLM().chat([])


# ---------------------------------------------------------------------------
# make_transport — MockTransport replaying canned JSON responses
# ---------------------------------------------------------------------------
def make_transport(responses: list[dict]) -> httpx.MockTransport:
    """Return a MockTransport that replays canned JSON responses in order."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"method": request.method, "url": str(request.url), "body": json.loads(request.content)})
        resp = responses.pop(0)
        return httpx.Response(status_code=resp.get("status", 200), json=resp.get("json", resp))

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    return transport


# ---------------------------------------------------------------------------
# TestOllamaLLM — request construction
# ---------------------------------------------------------------------------
def test_ollama_requests_chat_endpoint_with_expected_body():
    t = make_transport([{"json": {"message": {"role": "assistant", "content": "hi"}}}])
    llm = OllamaLLM(model="m", base_url="http://127.0.0.1:11451", transport=t)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hi"

    call = t.calls[0]  # type: ignore[attr-defined]
    assert call["url"].endswith("/api/chat")
    assert call["body"] == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def test_ollama_passes_messages_through_verbatim_including_extra_keys():
    # extra keys (vision images) are preserved, not filtered (P10.6 extension point)
    t = make_transport([{"json": {"message": {"role": "assistant", "content": "ok"}}}])
    llm = OllamaLLM(model="m", transport=t)
    msgs = [{"role": "user", "content": "hi", "images": ["data:png;base64,AA=="]}]
    llm.chat(msgs)

    call = t.calls[0]  # type: ignore[attr-defined]
    assert call["body"]["messages"] == msgs


def test_ollama_default_base_url_used_when_not_given():
    t = make_transport([{"json": {"message": {"role": "assistant", "content": "ok"}}}])
    llm = OllamaLLM(model="m", transport=t)
    llm.chat([{"role": "user", "content": "hi"}])
    assert t.calls[0]["url"].startswith("http://127.0.0.1:11451")  # type: ignore[attr-defined]


def test_ollama_constructor_requires_model():
    with pytest.raises(ValueError):
        OllamaLLM(model="")
    with pytest.raises(ValueError):
        OllamaLLM(model=None)  # type: ignore[arg-type]


def test_ollama_constructor_requires_non_empty_base_url():
    with pytest.raises(ValueError):
        OllamaLLM(model="m", base_url="")


# ---------------------------------------------------------------------------
# TestOllamaLLM — error mapping (non-2xx / connection / timeout)
# ---------------------------------------------------------------------------
def test_ollama_non_2xx_raises_llm_error_with_status():
    t = make_transport([{"status": 500, "json": {"error": "boom"}}])
    llm = OllamaLLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "500" in str(ei.value)


def test_ollama_non_2xx_body_is_flattened_truncated_and_scrubbed():
    """S8 review fix: a non-2xx body is proxy-controlled, so the error message
    folds it to one line, truncates it, and drops userinfo-shaped substrings
    (user:pass@) — sensitive config must never surface in error text."""
    body = "line one\n" * 5 + "proxy login http://user:secret@127.0.0.1:1/x" + "z" * 500

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=502, text=body)

    llm = OllamaLLM(model="m", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    msg = str(ei.value)
    assert "secret" not in msg
    assert "user:secret" not in msg
    assert "\n" not in msg  # flattened to one line
    assert len(msg) < len(body)  # truncated


def test_ollama_connection_failure_raises_llm_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    llm = OllamaLLM(model="m", transport=httpx.MockTransport(boom))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "hi"}])


def test_ollama_timeout_raises_llm_error_with_timeout_seconds():
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    llm = OllamaLLM(model="m", timeout=30.0, transport=httpx.MockTransport(timeout))
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert ei.value.timeout_seconds == 30.0


def test_ollama_redacts_userinfo_from_error_and_never_leaks_original():
    # no-secret two gates: URL userinfo removed, raw RequestError str never
    # concatenated into the message
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("http://user:secret@127.0.0.1:11451")

    llm = OllamaLLM(model="m", base_url="http://user:secret@127.0.0.1:11451", transport=httpx.MockTransport(boom))
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "secret" not in str(ei.value)


# ---------------------------------------------------------------------------
# TestOllamaLLM — response parsing (malformed / empty content)
# ---------------------------------------------------------------------------
def test_ollama_missing_message_key_raises_readable_llm_error():
    t = make_transport([{"json": {"foo": "bar"}}])
    llm = OllamaLLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "message" in str(ei.value)


def test_ollama_message_not_a_dict_raises_readable_llm_error():
    t = make_transport([{"json": {"message": "assistant"}}])
    llm = OllamaLLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "message" in str(ei.value)


def test_ollama_missing_content_key_raises_readable_llm_error():
    t = make_transport([{"json": {"message": {"role": "assistant"}}}])
    llm = OllamaLLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "content" in str(ei.value)


def test_ollama_content_not_a_str_raises_readable_llm_error():
    t = make_transport([{"json": {"message": {"role": "assistant", "content": 3}}}])
    llm = OllamaLLM(model="m", transport=t)
    with pytest.raises(LLMError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "content" in str(ei.value)


def test_ollama_empty_content_is_a_valid_result():
    t = make_transport([{"json": {"message": {"role": "assistant", "content": ""}}}])
    llm = OllamaLLM(model="m", transport=t)
    assert llm.chat([{"role": "user", "content": "hi"}]) == ""


def test_ollama_shares_base_contract():
    t = make_transport([{"json": {"message": {"role": "assistant", "content": "ok"}}}])
    llm = OllamaLLM(model="m", transport=t)
    with pytest.raises(ValueError):
        llm.chat([])


# ---------------------------------------------------------------------------
# TestCreateLLM — factory
# ---------------------------------------------------------------------------
def test_create_llm_fake_returns_fake_llm():
    from rag_mcp_server.libs.llm.factory import create_llm

    assert isinstance(create_llm("fake"), FakeLLM)


def test_create_llm_ollama_returns_ollama_llm():
    from rag_mcp_server.libs.llm.factory import create_llm

    assert isinstance(create_llm("ollama", model="m", base_url="http://127.0.0.1:11451"), OllamaLLM)


def test_create_llm_unknown_provider_raises_value_error():
    from rag_mcp_server.libs.llm.factory import create_llm

    with pytest.raises(ValueError) as ei:
        create_llm("unknown-provider")
    assert "unknown-provider" in str(ei.value)
    assert "fake" in str(ei.value)
    assert "ollama" in str(ei.value)


def test_create_llm_openai_returns_openai_llm():
    from rag_mcp_server.libs.llm.factory import create_llm
    from rag_mcp_server.libs.llm.openai import OpenAILLM

    llm = create_llm("openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1")
    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gpt-4o-mini"


def test_create_llm_deepseek_returns_openai_llm():
    # ADR 0018: DeepSeek reuses OpenAILLM — caller passes its base_url, no new class
    from rag_mcp_server.libs.llm.factory import create_llm
    from rag_mcp_server.libs.llm.openai import OpenAILLM

    llm = create_llm("deepseek", model="deepseek-chat", base_url="https://api.deepseek.com/v1")
    assert isinstance(llm, OpenAILLM)
    assert llm.base_url == "https://api.deepseek.com/v1"


# ---------------------------------------------------------------------------
# Package exports — libs/llm/__init__.py
# ---------------------------------------------------------------------------
def test_libs_llm_package_exports():
    import rag_mcp_server.libs.llm as llm_pkg

    assert llm_pkg.BaseLLM is BaseLLM
    assert llm_pkg.FakeLLM is FakeLLM
    assert llm_pkg.LLMError is LLMError
    assert callable(llm_pkg.create_llm)


def test_llm_error_reexported_from_libs_llm_package():
    import rag_mcp_server.libs.llm as llm_pkg

    assert llm_pkg.LLMError is LLMError
