"""Tests for OllamaEmbedding — mock-HTTP unit tests (no real network).

Red/green contract (docs/spec-p9-ollama-embedding.md, task 3):
- requests POST {base_url}/api/embed with {"model", "input", "truncate": false}
- no `dimensions` field sent (uses model default dimension)
- batches input by batch_size, calls sequentially, concatenates in input order
- partial final batch is sent as-is (no padding)
- dim() lazily probes with [""] on first call and caches the real dimension
- later calls validate returned dim matches cache; mismatch -> EmbeddingError
- connection failure / non-2xx / {"error":...} body / missing embeddings /
  count mismatch -> EmbeddingError (Chinese message); no retry
- embed_texts([]) returns [] and does not touch the network
- constructor: model required & non-empty; batch_size < 1 -> ValueError
"""

from __future__ import annotations

import json

import httpx
import pytest

from rag_mcp_server.core.errors import EmbeddingError
from rag_mcp_server.libs.embedding.ollama import OllamaEmbedding


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
# Request construction
# ---------------------------------------------------------------------------
def test_requests_embed_endpoint_with_expected_body():
    t = make_transport([{"json": {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}}])
    emb = OllamaEmbedding(model="nomic-embed-text", transport=t)
    emb.embed_texts(["a", "b"])

    call = t.calls[0]  # type: ignore[attr-defined]
    assert call["url"].endswith("/api/embed")
    assert call["body"] == {"model": "nomic-embed-text", "input": ["a", "b"], "truncate": False}


def test_no_dimensions_field_is_sent():
    t = make_transport([{"json": {"embeddings": [[0.1]]}}])
    emb = OllamaEmbedding(model="m", transport=t)
    emb.embed_texts(["a"])
    assert "dimensions" not in t.calls[0]["body"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Batching and row alignment
# ---------------------------------------------------------------------------
def test_batches_input_by_batch_size_and_concatenates_in_order():
    t = make_transport(
        [
            {"json": {"embeddings": [[1.0], [2.0]]}},
            {"json": {"embeddings": [[3.0]]}},
        ]
    )
    emb = OllamaEmbedding(model="m", batch_size=2, transport=t)
    vectors = emb.embed_texts(["a", "b", "c"])

    assert vectors == [[1.0], [2.0], [3.0]]
    assert len(t.calls) == 2  # type: ignore[attr-defined]
    assert t.calls[0]["body"]["input"] == ["a", "b"]  # type: ignore[attr-defined]
    assert t.calls[1]["body"]["input"] == ["c"]  # type: ignore[attr-defined]


def test_partial_final_batch_is_sent_as_is_no_padding():
    t = make_transport([{"json": {"embeddings": [[1.0], [2.0]]}}])
    emb = OllamaEmbedding(model="m", batch_size=2, transport=t)
    emb.embed_texts(["a", "b"])
    assert t.calls[0]["body"]["input"] == ["a", "b"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Dimension: lazy probe, cache, validation
# ---------------------------------------------------------------------------
def test_dim_lazily_probes_with_empty_string_and_caches():
    t = make_transport([{"json": {"embeddings": [[0.0] * 768]}}])
    emb = OllamaEmbedding(model="m", transport=t)
    assert emb.dim() == 768
    assert t.calls[0]["body"]["input"] == [""]  # type: ignore[attr-defined]
    # second dim() call does not re-probe
    assert emb.dim() == 768
    assert len(t.calls) == 1  # type: ignore[attr-defined]


def test_embed_texts_dim_mismatch_raises_embedding_error():
    # first encode caches 768, a later batch returns 3-dim -> contract broken
    t = make_transport(
        [
            {"json": {"embeddings": [[0.0] * 768]}},  # "warmup" -> caches 768
            {"json": {"embeddings": [[1.0, 2.0, 3.0]]}},  # "a" -> mismatch
        ]
    )
    emb = OllamaEmbedding(model="m", transport=t)
    emb.embed_texts(["warmup"])  # caches dim = 768, succeeds
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


# ---------------------------------------------------------------------------
# Error mapping -> EmbeddingError
# ---------------------------------------------------------------------------
def test_connection_failure_raises_embedding_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    emb = OllamaEmbedding(model="m", transport=httpx.MockTransport(boom))
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_non_2xx_raises_embedding_error():
    t = make_transport([{"status": 500, "json": {"error": "boom"}}])
    emb = OllamaEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_any_2xx_status_is_accepted():
    # spec: only non-2xx is an error; a 202 with valid body must succeed
    t = make_transport([{"status": 202, "json": {"embeddings": [[0.1, 0.2]]}}])
    emb = OllamaEmbedding(model="m", transport=t)
    assert emb.embed_texts(["a"]) == [[0.1, 0.2]]


def test_error_body_raises_embedding_error():
    t = make_transport([{"status": 200, "json": {"error": 'model "nope" not found'}}])
    emb = OllamaEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_missing_embeddings_field_raises_embedding_error():
    t = make_transport([{"status": 200, "json": {"total_duration": 1}}])
    emb = OllamaEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a"])


def test_count_mismatch_raises_embedding_error():
    t = make_transport([{"json": {"embeddings": [[0.1]]}}])
    emb = OllamaEmbedding(model="m", transport=t)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["a", "b"])


# ---------------------------------------------------------------------------
# Empty input and constructor validation
# ---------------------------------------------------------------------------
def test_empty_input_returns_empty_list_without_network():
    t = make_transport([])  # if a request is made, IndexError surfaces
    emb = OllamaEmbedding(model="m", transport=t)
    assert emb.embed_texts([]) == []
    assert len(t.calls) == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Input guards (shared with BaseEmbedding; exercised here for Ollama too)
# ---------------------------------------------------------------------------
def test_rejects_bare_string_without_network():
    t = make_transport([])
    emb = OllamaEmbedding(model="m", transport=t)
    with pytest.raises(TypeError):
        emb.embed_texts("hi")
    assert len(t.calls) == 0  # type: ignore[attr-defined]


def test_rejects_non_str_elements_without_network():
    t = make_transport([])
    emb = OllamaEmbedding(model="m", transport=t)
    with pytest.raises(TypeError):
        emb.embed_texts([None, "ok"])
    assert len(t.calls) == 0  # type: ignore[attr-defined]


def test_constructor_requires_model():
    with pytest.raises((TypeError, ValueError)):
        OllamaEmbedding(model="")  # empty model rejected


def test_constructor_rejects_batch_size_below_one():
    with pytest.raises(ValueError):
        OllamaEmbedding(model="m", batch_size=0)


def test_constructor_defaults_base_url_and_timeout():
    emb = OllamaEmbedding(model="m")
    assert emb.base_url == "http://127.0.0.1:11451"
    assert emb.batch_size == 16
    assert emb.timeout == 30.0


def test_constructor_timeout_is_configurable():
    emb = OllamaEmbedding(model="m", timeout=5.0)
    assert emb.timeout == 5.0
