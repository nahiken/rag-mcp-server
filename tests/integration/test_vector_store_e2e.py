"""End-to-end verification of the vector store against real backends.

Real Ollama embeddings (nomic-embed-text, 768-dim) + real embedded Chroma on
disk. Exercises the full store-level chain the spec promises
(docs/spec-p9-vstore.md user stories 2/4/6/7): embed -> add -> query ranks by
semantic similarity (not keyword overlap), score = 1/(1+distance) descending,
persist_dir survives reopen.

Marked @pytest.mark.integration and excluded from the default pytest run
(pyproject addopts "-m not integration"). Run explicitly with:
    PYTHONPATH=src python -m pytest -m integration tests/integration
Requires a local Ollama serving nomic-embed-text on 127.0.0.1:11451
(precedent: tests/integration/test_ollama_embedding.py).

The app layer is wired to the vector store as of task 9 (p9-query-001,
config-driven hybrid query) — see tests/integration/test_hybrid_retrieval.py for
the app-level real-backend chain (ingest -> hybrid query). This module remains
the deepest store-level chain: real embedding -> real persistent vector store.
"""

from __future__ import annotations

import pytest

from rag_mcp_server.libs.embedding.ollama import DEFAULT_OLLAMA_BASE_URL, OllamaEmbedding
from rag_mcp_server.libs.vector_store.base import VectorRecord
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

pytestmark = pytest.mark.integration

MODEL = "nomic-embed-text"
DIM = 768

# Three semantically distinct English passages; a query on topic X must rank
# the X passage first even though the passages share common words.
PASSAGES = {
    "tech::0": (
        "vector databases and semantic search power retrieval augmented generation systems",
        {"document_id": "tech", "source_path": "tech.md", "chunk_id": "tech::0", "content_hash": "t1"},
    ),
    "cook::0": (
        "a guide to baking sourdough bread with a perfect crispy golden crust",
        {"document_id": "cook", "source_path": "cook.md", "chunk_id": "cook::0", "content_hash": "c1"},
    ),
    "pet::0": (
        "household tips for caring for indoor cats and keeping their litter trays clean",
        {"document_id": "pet", "source_path": "pet.md", "chunk_id": "pet::0", "content_hash": "p1"},
    ),
}

QUERIES = {
    "semantic search database": "tech::0",
    "baking bread recipe": "cook::0",
    "cat care tips": "pet::0",
}


@pytest.fixture
def ollama() -> OllamaEmbedding:
    return OllamaEmbedding(model=MODEL, base_url=DEFAULT_OLLAMA_BASE_URL)


@pytest.fixture
def records() -> list[VectorRecord]:
    return [
        VectorRecord(chunk_id=cid, metadata=meta, text=text)
        for cid, (text, meta) in PASSAGES.items()
    ]


def _embed(ollama: OllamaEmbedding, texts: list[str]) -> list[list[float]]:
    return ollama.embed_texts(texts)


def test_real_semantic_ranking_puts_topic_first(ollama, records, tmp_path):
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection="e2e_rank")
    store.add(records, _embed(ollama, [text for text, _ in PASSAGES.values()]))
    for query, expected_first in QUERIES.items():
        results = store.query(_embed(ollama, [query])[0], top_k=3)
        assert results, f"query {query!r} returned no results"
        # most relevant topic first
        assert results[0].chunk_id == expected_first, (
            f"query {query!r} ranked {results[0].chunk_id} first, expected {expected_first}"
        )
        # similarity descending, all scores are real floats
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(s is not None and isinstance(s, float) for s in scores)
        # result carries text + metadata (US#7, ADR 0004)
        assert results[0].text == PASSAGES[expected_first][0]
        assert results[0].metadata["source_path"] == PASSAGES[expected_first][1]["source_path"]


def test_real_persist_reopen_is_readable(ollama, records, tmp_path):
    store1 = ChromaVectorStore(persist_dir=str(tmp_path), collection="e2e_persist")
    store1.add(records, _embed(ollama, [text for text, _ in PASSAGES.values()]))
    # reopen a fresh instance on the same persist_dir (US#4)
    store2 = ChromaVectorStore(persist_dir=str(tmp_path), collection="e2e_persist")
    back = store2.get_by_ids(["cook::0", "missing", "tech::0"])
    assert [r.chunk_id for r in back] == ["cook::0", "tech::0"]  # request order, missing skipped
    assert back[0].text == PASSAGES["cook::0"][0]
    assert back[0].score is None  # no similarity context outside query


def test_real_delete_then_query_excludes_removed(ollama, records, tmp_path):
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection="e2e_delete")
    store.add(records, _embed(ollama, [text for text, _ in PASSAGES.values()]))
    store.delete(["cook::0"])
    results = store.query(_embed(ollama, ["baking bread recipe"])[0], top_k=3)
    assert "cook::0" not in {r.chunk_id for r in results}
    assert store.get_by_ids(["cook::0"]) == []
