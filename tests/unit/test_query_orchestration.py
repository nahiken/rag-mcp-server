"""Tests for the query orchestration seam — rag_service.query.

Red/green contract (docs/spec-p9-hybrid.md, task 9, ticket 09-T1):
- query(text, index_dir, top_k, pipeline=None) is config-driven by
  retrieval.strategy (dense / sparse / hybrid)
- pipeline is the injection seam, symmetric with ingest: test passes the same
  pipeline (FakeEmbedding + temp Chroma) so store + embedding are the source
- invalid strategy fails fast with ConfigError at the query entry
- results carry chunk_id / source_path / score / text / sources
- no-result / empty store -> empty list, never an exception

The only test seam is rag_service.query with an injected pipeline. All offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_mcp_server.application.rag_service import ingest, query
from rag_mcp_server.core.errors import ConfigError
from rag_mcp_server.ingestion.pipeline import IngestPipeline
from rag_mcp_server.libs.embedding.fake import FakeEmbedding
from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore


@pytest.fixture
def offline_env():
    """A temp index + fake pipeline, shared between ingest and query."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "docs" / "rag-notes.md").write_text(
            "RAG 使用关键词检索 baseline，检索结果按 score 排序。",
            encoding="utf-8",
        )
        (root / "docs" / "mcp-notes.md").write_text(
            "MCP 是 Model Context Protocol，用于连接工具与数据。",
            encoding="utf-8",
        )
        embedding = FakeEmbedding(dim=4)
        store = ChromaVectorStore(persist_dir=None, collection="test_orch_isolated")
        # ADR 0016: pipeline collection must match the store collection
        pipeline = IngestPipeline(
            embedding=embedding, store=store, index_dir=root / "out",
            collection="test_orch_isolated",
        )
        ingest(root / "docs", base=root / "docs", index_dir=root / "out", pipeline=pipeline,
               trace_dir=root)
        yield root, root / "out", pipeline


def _patch_strategy(monkeypatch, strategy: str):
    """Force load_settings to return a given retrieval strategy.

    trace is disabled so these strategy tests never write to the repo logs/ —
    they test retrieval strategy, not trace (US 14: tests isolate the trace dir).
    """
    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.config.settings import DEFAULTS

    def fake_load_settings():
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
        data["retrieval"]["strategy"] = strategy
        data["trace"] = {"enabled": False, "dir": "logs"}
        return data

    monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)


# -- sparse strategy --------------------------------------------------------
def test_sparse_strategy_returns_keyword_hits(offline_env, monkeypatch):
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "sparse")
    results = query("MCP", index_dir=index_dir, pipeline=pipeline)
    assert results, "sparse query must hit the MCP chunk"
    assert results[0].chunk_id == "test_orch_isolated::mcp-notes-md::0"
    assert results[0].sources == ["sparse"]


# -- dense strategy ---------------------------------------------------------
def test_dense_strategy_returns_semantic_hits(offline_env, monkeypatch):
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "dense")
    results = query("MCP", index_dir=index_dir, pipeline=pipeline)
    assert results, "dense query must return results"
    for r in results:
        assert r.sources == ["dense"]


# -- hybrid strategy --------------------------------------------------------
def test_hybrid_strategy_fuses_both_routes(offline_env, monkeypatch):
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "hybrid")
    results = query("MCP", index_dir=index_dir, pipeline=pipeline)
    assert results, "hybrid query must return results"
    # MCP is a dense and sparse hit -> both sources present on the top hit
    assert results[0].sources == ["dense", "sparse"]


# -- result shape -----------------------------------------------------------
def test_results_carry_expected_fields(offline_env, monkeypatch):
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "sparse")
    results = query("RAG", index_dir=index_dir, pipeline=pipeline)
    assert results
    result = results[0]
    assert result.chunk_id
    assert result.source_path
    assert isinstance(result.score, float)
    assert result.text
    assert isinstance(result.sources, list)


def test_top_k_limits_results(offline_env, monkeypatch):
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "hybrid")
    results = query("检索", index_dir=index_dir, pipeline=pipeline, top_k=1)
    assert len(results) <= 1


# -- empty / failure paths --------------------------------------------------
def test_empty_store_returns_empty_list(offline_env):
    _, index_dir, _ = offline_env
    with tempfile.TemporaryDirectory() as tmp:
        empty_root = Path(tmp)
        empty_embedding = FakeEmbedding(dim=4)
        empty_store = ChromaVectorStore(persist_dir=None, collection="test_orch_empty")
        empty_pipeline = IngestPipeline(
            embedding=empty_embedding, store=empty_store, index_dir=empty_root / "out"
        )
        results = query("anything", index_dir=empty_root / "out", pipeline=empty_pipeline,
                        trace_dir=empty_root)
        assert results == []


def test_invalid_strategy_fails_fast_with_config_error(offline_env, monkeypatch):
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "bogus")
    with pytest.raises(ConfigError):
        query("MCP", index_dir=index_dir, pipeline=pipeline)


def test_empty_text_returns_empty_across_strategies(offline_env, monkeypatch):
    """空/空白查询文本无查询意图：所有策略都必须返回 []，不能随意返回库内 top_k。"""
    _, index_dir, pipeline = offline_env
    for strategy in ("sparse", "dense", "hybrid"):
        _patch_strategy(monkeypatch, strategy)
        assert query("", index_dir=index_dir, pipeline=pipeline) == []
        assert query("   ", index_dir=index_dir, pipeline=pipeline) == []


def test_invalid_strategy_fails_even_with_empty_text(offline_env, monkeypatch):
    """非法策略在空文本下也必须 fail-fast（不能藏在空结果后面）。"""
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "bogus")
    with pytest.raises(ConfigError):
        query("", index_dir=index_dir, pipeline=pipeline)


def test_negative_top_k_fails_fast_across_strategies(offline_env, monkeypatch):
    """非法 top_k（<=0）在所有策略下一致 fail-fast，不能静默截断。"""
    _, index_dir, pipeline = offline_env
    for strategy in ("sparse", "dense", "hybrid"):
        _patch_strategy(monkeypatch, strategy)
        with pytest.raises(ValueError):
            query("MCP", index_dir=index_dir, pipeline=pipeline, top_k=-1)
        with pytest.raises(ValueError):
            query("MCP", index_dir=index_dir, pipeline=pipeline, top_k=0)


def test_top_k_none_reads_config_default(offline_env, monkeypatch):
    """top_k=None 时读配置 retrieval.top_k（死配置接线）；无配置字段回退默认 3。"""
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "sparse")

    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.config.settings import DEFAULTS

    def fake_load_settings():
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
        data["retrieval"]["strategy"] = "sparse"
        data["retrieval"]["top_k"] = 1
        data["trace"] = {"enabled": False, "dir": "logs"}
        return data

    monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)
    results = query("检索", index_dir=index_dir, pipeline=pipeline, top_k=None)
    assert len(results) <= 1, f"配置 top_k=1 应截断，实际 {len(results)} 条"


def test_strategy_from_settings_is_respected(offline_env, monkeypatch):
    """When strategy is not passed, it comes from load_settings()."""
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "sparse")
    results = query("MCP", index_dir=index_dir, pipeline=pipeline)
    assert results
    assert results[0].sources == ["sparse"]


# -- rerank wiring (P10.3) --------------------------------------------------
def test_rerank_true_with_injected_llm_reranker_reorders(offline_env, monkeypatch):
    """rerank=True + injected LLMRanker (FakeLLM scoring array) reorders results."""
    import json as _json

    from rag_mcp_server.libs.llm.fake import FakeLLM
    from rag_mcp_server.libs.reranker.llm_reranker import LLMReranker

    # LLM scores the MCP chunk higher than the RAG chunk
    def scored(messages):
        # echo the highest-scoring id; simplest: return a canned array covering
        # both possible chunk ids with MCP scored above RAG
        return _json.dumps(
            [
                {"passage_id": "test_orch_isolated::mcp-notes-md::0", "score": 3, "reasoning": "on topic"},
                {"passage_id": "test_orch_isolated::rag-notes-md::0", "score": 1, "reasoning": "off topic"},
            ]
        )

    llm = FakeLLM()
    llm._chat = scored  # type: ignore[method-assign]

    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "hybrid")
    results = query("MCP", index_dir=index_dir, pipeline=pipeline,
                    rerank=True, reranker=LLMReranker(llm=llm))
    assert results, "reranked query must return results"
    assert results[0].chunk_id == "test_orch_isolated::mcp-notes-md::0"
    assert results[0].score == 3.0  # score = LLM score
    assert results[0].sources == ["dense", "sparse"]  # sources preserved


def test_rerank_false_keeps_default_order(offline_env, monkeypatch):
    """rerank=False keeps the default retrieval order (no reranking)."""
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "hybrid")
    results = query("MCP", index_dir=index_dir, pipeline=pipeline, rerank=False)
    assert results, "query must return results"
    assert results[0].chunk_id == "test_orch_isolated::mcp-notes-md::0"  # hybrid default order


def test_rerank_none_reads_config_default_off(offline_env, monkeypatch):
    """rerank=None reads settings rerank.enabled (default off -> no rerank stage)."""
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "hybrid")
    results = query("MCP", index_dir=index_dir, pipeline=pipeline, rerank=None)
    assert results  # just must not raise / change shape


def test_rerank_unknown_provider_fails_fast_with_config_error(offline_env, monkeypatch):
    """rerank.enabled + unknown provider -> ConfigError (fail fast, not a silent no-op).

    P1 review finding: a bogus provider previously degraded to keep-order without
    error, bypassing the factory fail-fast. The provider is validated at the
    assembly seam, mirroring _RETRIEVAL_STRATEGIES (rag_service).
    """
    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "hybrid")

    import rag_mcp_server.config.settings as settings_mod
    from rag_mcp_server.config.settings import DEFAULTS

    def fake_load_settings():
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
        data["retrieval"]["strategy"] = "hybrid"
        data["rerank"] = {"enabled": True, "provider": "bogus"}
        data["trace"] = {"enabled": False, "dir": "logs"}
        return data

    monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)
    with pytest.raises(ConfigError):
        query("MCP", index_dir=index_dir, pipeline=pipeline)


def test_rerank_enabled_consumes_enlarged_candidate_pool(offline_env, monkeypatch):
    """US12/P3: rerank enabled feeds a top_k * CANDIDATE_MULTIPLIER candidate pool.

    top_k=1 + rerank=True: the hybrid branch must fuse up to top_k * 3
    candidates and hand them to the rerank backend, which truncates back to
    top_k. A recording reranker proves the enlarged pool is actually consumed
    (the P3 review finding: fusion truncated to top_k before rerank, so the
    quality enhancement could never promote passages outside the initial top_k).
    """
    from rag_mcp_server.libs.reranker.base import BaseReranker
    from rag_mcp_server.retrieval.hybrid import CANDIDATE_MULTIPLIER

    class RecordingReranker(BaseReranker):
        def __init__(self):
            self.seen_input = 0

        def rerank(self, query, results, top_k):
            self.seen_input = len(results)
            return results[:top_k]

    recorder = RecordingReranker()

    _, index_dir, pipeline = offline_env
    _patch_strategy(monkeypatch, "hybrid")
    results = query("MCP", index_dir=index_dir, pipeline=pipeline,
                    top_k=1, rerank=True, reranker=recorder)

    assert len(results) <= 1  # truncated back to top_k after rerank
    # the fixture store has 2 chunks; fused at top_k*3 (capped by corpus) -> the
    # backend saw > top_k candidates. Before the fix it saw exactly top_k=1.
    assert recorder.seen_input > 1
    assert recorder.seen_input <= 1 * CANDIDATE_MULTIPLIER
