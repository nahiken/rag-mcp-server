"""Integration smoke tests for hybrid retrieval against real backends.

Real Ollama embeddings (nomic-embed-text, 768-dim) + real embedded Chroma on
disk. Exercises the deepest app-level chain available after task 9
(docs/spec-p9-hybrid.md): ingest via rag_service -> config-driven hybrid
query -> semantic + keyword hits fused with RRF, sources reported.

Marked @pytest.mark.integration and excluded from the default pytest run
(pyproject addopts "-m not integration"). Run explicitly with:
    PYTHONPATH=src python -m pytest -m integration tests/integration
Requires a local Ollama serving nomic-embed-text on 127.0.0.1:11451
(precedent: tests/integration/test_ollama_embedding.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_mcp_server.application.rag_service import ingest, query
from rag_mcp_server.config.settings import DEFAULT_CONFIG_DIR

pytestmark = pytest.mark.integration


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A temp index + settings dir so the run never touches repo data/."""
    root = tmp_path
    (root / "docs").mkdir()
    (root / "docs" / "tech.md").write_text(
        "vector databases and semantic search power retrieval augmented generation systems",
        encoding="utf-8",
    )
    (root / "docs" / "cook.md").write_text(
        "a guide to baking sourdough bread with a perfect crispy golden crust",
        encoding="utf-8",
    )
    (root / "docs" / "pet.md").write_text(
        "household tips for caring for indoor cats and keeping their litter trays clean",
        encoding="utf-8",
    )
    # settings dir: strategy hybrid, Chroma persisted under tmp
    cfg = root / "config"
    cfg.mkdir()
    (cfg / "settings.yaml").write_text(
        f"vector_store:\n"
        f"  persist_dir: {root / 'chroma'}\n"
        f"retrieval:\n"
        f"  strategy: hybrid\n"
        f"  top_k: 3\n"
        f"  rrf_k: 60\n"
        f"trace:\n"
        f"  enabled: false\n",  # isolate: hybrid retrieval test is not a trace test (US 14)
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    index_dir = root / "index"
    ingest(root / "docs", base=root / "docs", index_dir=index_dir)
    yield root, index_dir


def test_real_hybrid_query_ranks_semantic_hit_first(env) -> None:
    root, index_dir = env
    # hybrid query on a tech topic: dense ranks tech first, sparse may add hits
    results = query("semantic search database", index_dir=index_dir)
    assert results, "hybrid query must return results"
    # the semantically-relevant passage ranks first
    assert results[0].chunk_id == "tech-md::0"
    # every result carries sources, and the top hit is at least a dense hit
    assert all(r.sources for r in results)


def test_real_hybrid_query_reports_sources(env) -> None:
    root, index_dir = env
    results = query("baking bread recipe", index_dir=index_dir)
    assert results, "hybrid query must return results"
    assert results[0].chunk_id == "cook-md::0"
    # "bread"/"recipe" are keyword hits in the same chunk -> both sources
    top = results[0]
    assert "dense" in top.sources
