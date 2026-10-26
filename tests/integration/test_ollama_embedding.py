"""Integration smoke tests for OllamaEmbedding against a real local Ollama.

Marked @pytest.mark.integration and excluded from the default pytest run
(pyproject addopts "-m not integration"). Run explicitly with:
    PYTHONPATH=src python -m pytest -m integration tests/integration
Requires a local Ollama serving nomic-embed-text on 127.0.0.1:11451.
"""

from __future__ import annotations

import pytest

from rag_mcp_server.libs.embedding.ollama import DEFAULT_OLLAMA_BASE_URL, OllamaEmbedding

pytestmark = pytest.mark.integration

MODEL = "nomic-embed-text"
DIM = 768


@pytest.fixture
def ollama() -> OllamaEmbedding:
    return OllamaEmbedding(model=MODEL, base_url=DEFAULT_OLLAMA_BASE_URL)


def test_real_embed_returns_768_dim(ollama: OllamaEmbedding) -> None:
    vectors = ollama.embed_texts(["hello", "世界", "检索系统"])
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == DIM


def test_real_dim_matches_expected(ollama: OllamaEmbedding) -> None:
    assert ollama.dim() == DIM


def test_real_batching_keeps_row_alignment(ollama: OllamaEmbedding) -> None:
    # batch_size 2 forces two real requests for 3 inputs
    small = OllamaEmbedding(model=MODEL, base_url=DEFAULT_OLLAMA_BASE_URL, batch_size=2)
    texts = ["alpha", "beta", "gamma"]
    vectors = small.embed_texts(texts)
    assert len(vectors) == len(texts)
    assert all(len(v) == DIM for v in vectors)
