"""Tests for OpenAIEmbedding — OpenAI-compatible embeddings provider.

Red/green contract (docs/adr/0018-multi-provider.md, T2):
- requests POST {base_url}/embeddings with {"model","input"} + Bearer auth
- vectors row-aligned with input; batches input by batch_size, concatenates in order
- dim() lazily probes with [""] on first call and caches
- later dim mismatch -> EmbeddingError
- non-2xx / connect / invalid JSON / missing data / count mismatch -> EmbeddingError
- api_key priority: constructor > OPENAI_API_KEY env
- base_url default https://api.openai.com/v1; empty input -> [] without network
"""

from __future__ import annotations

import json

import httpx
import pytest

from rag_mcp_server.core.errors import EmbeddingError
from rag_mcp_server.libs.embedding.openai import OpenAIEmbedding


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


def embed_payload(vectors: list[list[float]]) -> dict:
    return {"data": [{"embedding": v, "index": i} for i, v in enumerate(vectors)]}


# ---------------------------------------------------------------------------
# Request construction + auth
# ---------------------------------------------------------------------------
def test_requests_embeddings_endpoint_with_expected_body_and_auth():
    t = make_transport([{"json": embed_payload([[0.1, 0.2], [0.3, 0.4]])}])
    emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="sk-test", transport=t)
    vectors = emb.embed_texts(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    call = t.calls[0]  # type: ignore[attr-defined]
    assert call["url"].endswith("/embeddings")
    assert call["body"] == {"model": "text-embedding-3-small", "input": ["a", "b"]}
    assert call["headers"]["authorization"] == "Bearer sk-test"


def test_no_authorization_header_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    t = make_transport([{"json": embed_payload([[0.1]])}])
    emb = OpenAIEmbedding(model="m", api_key=None, transport=t)
    emb.embed_texts(["a"])
    assert "authorization" not in {k.lower() for k in t.calls[0]["headers"]}  # type: ignore[attr-defined]


def test_default_base_url_is_openai_v1():
    t = make_transport([{"json": embed_payload([[0.1]])}])
    emb = OpenAIEmbedding(model="m", transport=t)
    emb.embed_texts(["a"])
    assert t.calls[0]["url"].startswith("https://api.openai.com/v1")  # type: ignore[attr-defined]


def test_base_url_trailing_slash_is_stripped():
    emb = OpenAIEmbedding(model="m", base_url="https://api.openai.com/v1/")
    assert emb.base_url == "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# api_key env fallback (constructor > env)
# ---------------------------------------------------------------------------
def test_api_key_reads_env_when_constructor_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    emb = OpenAIEmbedding(model="m")
    assert emb.api_key == "sk-from-env"


def test_api_key_env_fallback_sends_auth_header(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    t = make_transport([{"json": embed_payload([[0.1]])}])
    emb = OpenAIEmbedding(model="m", transport=t)
    emb.embed_texts(["a"])
    assert t.calls[0]["headers"]["authorization"] == "Bearer sk-from-env"  # type: ignore[attr-defined]


def test_constructor_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    emb = OpenAIEmbedding(model="m", api_key="sk-explicit")
    assert emb.api_key == "sk-explicit"


# ---------------------------------------------------------------------------
# Batching and row alignment
# ---------------------------------------------------------------------------
def test_batches_input_by_batch_size_and_concatenates_in_order():
    t = make_transport(
        [
            {"json": embed_payload([[1.0], [2.0]])},
            {"json": embed_payload([[3.0]])},
        ]
    )
    emb = OpenAIEmbedding(model="m", batch_size=2, transport=t)
    assert emb.embed_texts(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]
    assert len(t.calls) == 2  # type: ignore[attr-defined]
    assert t.calls[0]["body"]["input"] == ["a", "b"]  # type: ignore[attr-defined]
    assert t.calls[1]["body"]["input"] == ["c"]  # type: ignore[attr-defined]


def test_partial_final_batch_is_sent_as_is_no_padding():
    t = make_transport([{"json": embed_payload([[1.0], [2.0]])}])
    emb = OpenAIEmbedding(model="m", batch_size=2, transport=t)
    emb.embed_texts(["a", "b"])
    assert t.calls[0]["body"]["input"] == ["a", "b"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Dimension: lazy probe, cache, validation
# ---------------------------------------------------------------------------
def test_dim_lazily_probes_with_empty_string_and_caches():
    t = make_transport([{"json": embed_payload([[0.0] * 768])}])
    emb = OpenAIEmbedding(model="m", transport=t)
    assert emb.dim() == 768
    assert t.calls[0]["body"]["input"] == [""]  # type: ignore[attr-defined]
    # second dim() call does not re-probe
    assert emb.dim() == 768
    assert len(t.calls) == 1  # type: ignore[attr-defined]


def test_embed_texts_dim_mismatch_raises_embedding_error():
    # first encode caches 768, a later batch returns 3-dim -> contract broken
    t = make_transport(
        [
            {"json": embed_payload([[0.0] * 768])},  # "warmup" -> caches 768
            {"json": embed_payload([[1.0, 2.0, 3.0]])},  # "a" -> mismatch
        ]
    )
    emb = OpenAIEmbedding(model="m", transport=t)
    emb.embed_texts(["warmup"])  # caches dim = 768, succeeds
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


# ---------------------------------------------------------------------------
# Error mapping -> EmbeddingError
# ---------------------------------------------------------------------------
def test_connection_failure_raises_embedding_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    emb = OpenAIEmbedding(model="m", transport=httpx.MockTransport(boom))
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_non_2xx_raises_embedding_error():
    t = make_transport([{"status": 500, "json": {"error": {"message": "boom"}}}])
    emb = OpenAIEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_any_2xx_status_is_accepted():
    # spec: only non-2xx is an error; a 202 with valid body must succeed
    t = make_transport([{"status": 202, "json": embed_payload([[0.1, 0.2]])}])
    emb = OpenAIEmbedding(model="m", transport=t)
    assert emb.embed_texts(["a"]) == [[0.1, 0.2]]


def test_invalid_json_response_raises_embedding_error():
    def bad_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    emb = OpenAIEmbedding(model="m", transport=httpx.MockTransport(bad_json))
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_top_level_not_a_dict_raises_embedding_error():
    t = make_transport([{"json": ["data"]}])
    emb = OpenAIEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_error_body_raises_embedding_error():
    t = make_transport([{"json": {"error": {"message": "model not found"}}}])
    emb = OpenAIEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_missing_data_field_raises_embedding_error():
    t = make_transport([{"json": {"object": "list"}}])
    emb = OpenAIEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError) as ei:
        emb.embed_texts(["a"])
    assert "data" in str(ei.value)


def test_data_not_a_list_raises_embedding_error():
    t = make_transport([{"json": {"data": "oops"}}])
    emb = OpenAIEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError) as ei:
        emb.embed_texts(["a"])
    assert "data" in str(ei.value)


def test_count_mismatch_raises_embedding_error():
    t = make_transport([{"json": embed_payload([[0.1]])}])
    emb = OpenAIEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a", "b"])


def test_row_not_a_dict_raises_embedding_error():
    t = make_transport([{"json": {"data": ["not-a-dict"]}}])
    emb = OpenAIEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_embedding_not_a_list_raises_embedding_error():
    t = make_transport([{"json": {"data": [{"embedding": "oops"}]}}])
    emb = OpenAIEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


# ---------------------------------------------------------------------------
# Empty input + constructor validation
# ---------------------------------------------------------------------------
def test_empty_input_returns_empty_list_without_network():
    t = make_transport([])  # if a request is made, IndexError surfaces
    emb = OpenAIEmbedding(model="m", transport=t)
    assert emb.embed_texts([]) == []
    assert len(t.calls) == 0  # type: ignore[attr-defined]


def test_constructor_requires_model():
    with pytest.raises((TypeError, ValueError)):
        OpenAIEmbedding(model="")


def test_constructor_requires_non_empty_base_url():
    with pytest.raises(ValueError):
        OpenAIEmbedding(model="m", base_url="")


def test_constructor_rejects_batch_size_below_one():
    with pytest.raises(ValueError):
        OpenAIEmbedding(model="m", batch_size=0)


def test_constructor_defaults():
    emb = OpenAIEmbedding(model="m")
    assert emb.base_url == "https://api.openai.com/v1"
    assert emb.batch_size == 16
    assert emb.timeout == 30.0
