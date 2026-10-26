"""E2E: real Ollama chat chain through the factory -> provider seam.

Validates the one thing the unit tests cannot: the ACTUAL /api/chat response
shape on this machine (spec Further Notes: the response shape was NOT personally
verified; the mock tests anchor on the official shape). A stream:false request
is expected to return a single JSON object {"message":{"role","content"}} —
NOT NDJSON stream lines.

Requires local Ollama @ http://127.0.0.1:11451 with model qwen2.5:7b pulled.
Marked integration: excluded from the default run (`-m 'not integration'`).

Chain under test (single seam, top of the stack):
  create_llm("ollama", base_url, model)  ->  OllamaLLM.chat(messages)  ->  POST /api/chat
"""

from __future__ import annotations

import httpx
import pytest

from rag_mcp_server.libs.llm.factory import create_llm

OLLAMA_BASE_URL = "http://127.0.0.1:11451"
OLLAMA_MODEL = "qwen2.5:7b"
# 7B cold-load can take ~10s on first call; allow generous headroom.
SMALL_TIMEOUT = 60.0

pytestmark = pytest.mark.integration


def _server_reachable() -> bool:
    try:
        httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


requires_ollama = pytest.mark.skipif(not _server_reachable(), reason="本地 Ollama 不可达")


@requires_ollama
def test_factory_creates_ollama_llm_and_returns_text():
    llm = create_llm("ollama", base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, timeout=SMALL_TIMEOUT)
    reply = llm.chat([{"role": "user", "content": "回复一个字：好"}])
    assert isinstance(reply, str)
    assert reply.strip()  # non-empty


@requires_ollama
def test_ollama_chat_response_is_single_json_object_not_stream_lines():
    """The response shape that stream:false should produce — a single JSON
    object with message.content, not newline-delimited stream frames."""
    llm = create_llm("ollama", base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, timeout=SMALL_TIMEOUT)
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False},
        timeout=SMALL_TIMEOUT,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    msg = data.get("message")
    assert isinstance(msg, dict)
    assert isinstance(msg.get("content"), str)
