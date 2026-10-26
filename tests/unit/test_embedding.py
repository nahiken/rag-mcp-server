"""Tests for the BaseEmbedding seam — FakeEmbedding external behaviour.

Red/green contract (docs/spec-p9-embedding.md, task 2):
- embed_texts maps a batch of texts to equal-dim float vectors
- empty input returns [] (same shape as input)
- FakeEmbedding is deterministic: same text -> identical vector
- dim() is introspectable and matches returned vector length
- dim is configurable via constructor (default 768, aligned with nomic-embed-text)
- batch calls are idempotent
"""

from __future__ import annotations

import pytest

from rag_mcp_server.libs.embedding.base import BaseEmbedding
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.embedding.factory import create_embedding


def test_fake_embedding_returns_equal_dim_vectors():
    emb = FakeEmbedding()
    vectors = emb.embed_texts(["hello world", "检索系统"])
    assert len(vectors) == 2
    for v in vectors:
        assert len(v) == emb.dim()
        assert all(isinstance(x, float) for x in v)


def test_fake_embedding_is_deterministic():
    emb = FakeEmbedding()
    text = "同一段文本"
    first = emb.embed_texts([text])[0]
    second = emb.embed_texts([text])[0]
    assert first == second
    assert all(a == b for a, b in zip(first, second))


def test_fake_embedding_default_dim_matches_real_model():
    # default aligns with nomic-embed-text output dimension
    assert FakeEmbedding().dim() == 768


def test_fake_embedding_dim_is_configurable():
    emb = FakeEmbedding(dim=3)
    assert emb.dim() == 3
    vectors = emb.embed_texts(["abc"])
    assert len(vectors[0]) == 3


def test_fake_embedding_empty_input_returns_empty_list():
    emb = FakeEmbedding()
    assert emb.embed_texts([]) == []


def test_fake_embedding_batch_is_idempotent():
    emb = FakeEmbedding()
    texts = ["a", "b", "c"]
    run1 = emb.embed_texts(texts)
    run2 = emb.embed_texts(texts)
    assert run1 == run2


def test_fake_embedding_vectors_are_bounded():
    # sha256 mapping lands in [-1, 1]
    emb = FakeEmbedding(dim=16)
    for v in emb.embed_texts(["x", "y", "z"]):
        assert all(-1.0 <= x <= 1.0 for x in v)


def test_fake_embedding_different_texts_usually_differ():
    emb = FakeEmbedding(dim=64)
    a = emb.embed_texts(["alpha"])[0]
    b = emb.embed_texts(["beta"])[0]
    assert a != b


def test_fake_embedding_default_dim_is_not_periodic():
    # dim > digest length must extend entropy (distinct blocks), not tile the
    # same 32-byte digest 24 times — periodic blocks give ~96% correlated dims
    emb = FakeEmbedding()
    a = emb.embed_texts(["alpha"])[0]
    assert len(a) == 768
    blocks = {tuple(a[i * 32:(i + 1) * 32]) for i in range(24)}
    assert len(blocks) > 1


def test_fake_embedding_dim_33_crosses_digest_boundary():
    emb = FakeEmbedding(dim=33)
    a = emb.embed_texts(["alpha"])[0]
    assert len(a) == 33
    b = emb.embed_texts(["beta"])[0]
    assert a != b
    assert FakeEmbedding(dim=33).embed_texts(["alpha"])[0] == a  # deterministic


def test_base_embedding_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbedding()  # type: ignore[abstract]


def test_factory_creates_fake():
    emb = create_embedding("fake")
    assert isinstance(emb, FakeEmbedding)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        create_embedding("not-a-provider")


def test_factory_creates_ollama():
    # lazy import keeps the unit suite free of the ollama/httpx dependency
    from rag_mcp_server.libs.embedding.ollama import OllamaEmbedding

    emb = create_embedding("ollama", model="nomic-embed-text")
    assert isinstance(emb, OllamaEmbedding)
    assert emb.model == "nomic-embed-text"


def test_factory_creates_openai():
    # ADR 0018: OpenAI-compatible /embeddings endpoint; lazy import keeps the
    # unit suite free of httpx (constructor makes no network call)
    from rag_mcp_server.libs.embedding.openai import OpenAIEmbedding

    emb = create_embedding("openai", model="text-embedding-3-small")
    assert isinstance(emb, OpenAIEmbedding)
    assert emb.model == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Input guards promoted from .scratch edge findings (E1/E2/E3/E5): the
# list[str] contract is enforced by BaseEmbedding, not each implementation.
# ---------------------------------------------------------------------------
def test_embed_texts_rejects_bare_string():
    # a bare string must not be silently char-encoded into N vectors
    with pytest.raises(TypeError):
        FakeEmbedding(dim=4).embed_texts("hi")


def test_embed_texts_rejects_non_str_elements():
    with pytest.raises(TypeError):
        FakeEmbedding(dim=4).embed_texts([None, "ok"])


def test_embed_texts_empty_list_returns_empty_list():
    # empty list is not an error and does not touch the provider
    assert FakeEmbedding().embed_texts([]) == []


def test_fake_embedding_rejects_negative_dim():
    with pytest.raises((ValueError, TypeError)):
        FakeEmbedding(dim=-1)


def test_fake_embedding_rejects_non_int_dim():
    with pytest.raises((ValueError, TypeError)):
        FakeEmbedding(dim="3")
    with pytest.raises((ValueError, TypeError)):
        FakeEmbedding(dim=None)


def test_fake_embedding_rejects_zero_dim():
    with pytest.raises((ValueError, TypeError)):
        FakeEmbedding(dim=0)
