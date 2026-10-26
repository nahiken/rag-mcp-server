"""Tests for CrossEncoderReranker — local cross-encoder passage re-ranking.

Red/green contract (docs/implementation-plans.md 方案 6, ADR 0009):
- CrossEncoderReranker.rerank(query, results, top_k) -> list[RetrievalResult];
  never change chunk_id/text/source_path, only reorder (and score)
- scores are the model's raw logit on (query, passage) pairs (larger = more
  relevant); the model is loaded lazily on first rerank and cached on the
  instance
- a fake scorer (a callable returning a score list) is injected for unit tests
  so no real model is ever downloaded; the real load path is exercised with a
  fake `sentence_transformers` module in sys.modules
- load/predict failure raises RerankError so QueryReranker falls back to the
  original order (ADR 0009)
- empty results -> [] without loading a model
- create_reranker("cross_encoder") returns an instance without touching the
  model (sentence-transformers stays an optional dependency)

Prior art: tests/unit/test_reranker.py (LLM backend + orchestration), the lazy
import pattern in rag_service._apply_rerank.
"""

from __future__ import annotations

import sys
import types

import pytest

from rag_mcp_server.core.errors import RerankError
from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.libs.reranker.base import BaseReranker
from rag_mcp_server.libs.reranker.cross_encoder import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
)
from rag_mcp_server.libs.reranker.factory import create_reranker


def _result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_path=f"docs/{chunk_id}.md",
        score=score,
        text=f"text of {chunk_id}",
        sources=["dense"],
        metadata={"image_refs": [f"{chunk_id}.png"], "document_id": "secret"},
    )


def _fake_st_module(model_cls):
    """A fake ``sentence_transformers`` module exposing ``CrossEncoder``."""
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.CrossEncoder = model_cls
    return fake_st


# ---------------------------------------------------------------------------
# contract + defaults
# ---------------------------------------------------------------------------
def test_cross_encoder_is_base_reranker():
    assert isinstance(CrossEncoderReranker(), BaseReranker)


def test_default_model_name_and_lazy_model():
    rr = CrossEncoderReranker()
    assert rr.model_name == DEFAULT_CROSS_ENCODER_MODEL
    assert rr.max_length == 512
    assert rr._model is None  # nothing loaded at construction


# ---------------------------------------------------------------------------
# fake scorer: reorder + score + contract (ADR 0009)
# ---------------------------------------------------------------------------
def test_rerank_reorders_by_score_descending_and_updates_score():
    rr = CrossEncoderReranker(scorer=lambda pairs: [0.1, 0.9, 0.5])
    results = [_result("a"), _result("b"), _result("c")]
    out = rr.rerank("q", results, top_k=3)

    assert [r.chunk_id for r in out] == ["b", "c", "a"]
    assert [r.score for r in out] == [0.9, 0.5, 0.1]
    # contract fields preserved; metadata only carries image_refs (P10.6 T3)
    assert out[0].text == "text of b"
    assert out[0].source_path == "docs/b.md"
    assert out[0].sources == ["dense"]
    assert out[0].metadata == {"image_refs": ["b.png"]}


def test_rerank_truncates_to_top_k():
    rr = CrossEncoderReranker(scorer=lambda pairs: [0.1, 0.9, 0.5, 0.8])
    results = [_result("a"), _result("b"), _result("c"), _result("d")]
    out = rr.rerank("q", results, top_k=2)

    assert [r.chunk_id for r in out] == ["b", "d"]  # top-2 by score
    assert len(out) == 2


def test_rerank_passes_query_passage_pairs_to_scorer():
    seen: dict = {}

    def scorer(pairs):
        seen["pairs"] = list(pairs)
        return [0.5, 0.2]

    rr = CrossEncoderReranker(scorer=scorer)
    out = rr.rerank("my query", [_result("a"), _result("b")], top_k=2)
    assert seen["pairs"] == [("my query", "text of a"), ("my query", "text of b")]
    assert [r.chunk_id for r in out] == ["a", "b"]


def test_rerank_does_not_mutate_input_results():
    rr = CrossEncoderReranker(scorer=lambda pairs: [0.9, 0.1])
    results = [_result("a"), _result("b")]
    rr.rerank("q", results, top_k=2)

    assert [r.chunk_id for r in results] == ["a", "b"]  # original order untouched
    assert results[0].score == 1.0  # original scores untouched


def test_rerank_empty_results_returns_empty_without_loading_model():
    rr = CrossEncoderReranker()  # no scorer -> would load a real model
    out = rr.rerank("q", [], top_k=3)

    assert out == []
    assert rr._model is None  # never attempted to load


def test_rerank_invalid_top_k_fails_fast():
    rr = CrossEncoderReranker(scorer=lambda pairs: [0.5])
    with pytest.raises(ValueError):
        rr.rerank("q", [_result("a")], top_k=0)
    with pytest.raises(ValueError):
        rr.rerank("q", [_result("a")], top_k=-1)


# ---------------------------------------------------------------------------
# real load path (fake sentence_transformers in sys.modules): lazy + cached
# ---------------------------------------------------------------------------
def test_load_model_lazily_and_caches_on_first_rerank(monkeypatch):
    class FakeCrossEncoder:
        instances: list = []

        def __init__(self, model_name, max_length=None):
            self.model_name = model_name
            self.max_length = max_length
            FakeCrossEncoder.instances.append(self)

        def predict(self, pairs):
            return [0.9, 0.1, 0.5]

    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_st_module(FakeCrossEncoder))
    rr = CrossEncoderReranker(model_name="local/model", max_length=128)
    assert rr._model is None

    results = [_result("a"), _result("b"), _result("c")]
    out = rr.rerank("q", results, top_k=2)
    assert [r.chunk_id for r in out] == ["a", "c"]  # 0.9 > 0.5 > 0.1
    assert rr._model is FakeCrossEncoder.instances[0]
    assert rr._model.model_name == "local/model"
    assert rr._model.max_length == 128

    rr.rerank("q", results, top_k=2)  # second call reuses the cached model
    assert len(FakeCrossEncoder.instances) == 1  # loaded exactly once


def test_load_model_failure_raises_rerank_error(monkeypatch):
    # sentence_transformers "missing" (or halted) -> ImportError -> RerankError
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    rr = CrossEncoderReranker()
    with pytest.raises(RerankError):
        rr.rerank("q", [_result("a")], top_k=1)


def test_predict_failure_via_real_path_raises_rerank_error(monkeypatch):
    class BoomModel:
        def predict(self, pairs):
            raise RuntimeError("gpu oom")

    fake_st = _fake_st_module(lambda model_name, max_length=None: BoomModel())
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    rr = CrossEncoderReranker()
    with pytest.raises(RerankError) as ei:
        rr.rerank("q", [_result("a")], top_k=1)
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_scorer_failure_raises_rerank_error():
    def boom(pairs):
        raise RuntimeError("scorer down")

    rr = CrossEncoderReranker(scorer=boom)
    with pytest.raises(RerankError) as ei:
        rr.rerank("q", [_result("a")], top_k=1)
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_scorer_wrong_length_raises_rerank_error():
    """review FIX: a score list with the wrong length must surface as
    RerankError (orchestrator falls back), never a bare IndexError."""
    rr = CrossEncoderReranker(scorer=lambda pairs: [0.5])  # 1 score, 2 pairs
    with pytest.raises(RerankError):
        rr.rerank("q", [_result("a"), _result("b")], top_k=2)


def test_scorer_non_numeric_raises_rerank_error():
    """review FIX: a non-numeric score must surface as RerankError, not a
    bare TypeError/ValueError leaking to the caller."""
    rr = CrossEncoderReranker(scorer=lambda pairs: ["high", 0.5])
    with pytest.raises(RerankError):
        rr.rerank("q", [_result("a"), _result("b")], top_k=2)


# ---------------------------------------------------------------------------
# factory: create_reranker("cross_encoder") — no model touched
# ---------------------------------------------------------------------------
def test_create_reranker_cross_encoder_returns_instance_without_model():
    rr = create_reranker("cross_encoder")
    assert isinstance(rr, CrossEncoderReranker)
    assert rr._model is None  # no sentence-transformers import / model load


def test_create_reranker_cross_encoder_passes_kwargs():
    rr = create_reranker("cross_encoder", model_name="local/path", max_length=256)
    assert rr.model_name == "local/path"
    assert rr.max_length == 256
