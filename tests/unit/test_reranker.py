"""Tests for the pluggable Reranker layer — None/LLM backends + orchestration.

Red/green contract (docs/spec-p10-rerank.md, ticket 03, ADR 0009):
- BaseReranker.rerank(query, results, top_k) -> list[RetrievalResult]; never
  change chunk_id/text/source_path, only reorder (and score)
- NoneReranker keeps original order (default fallback)
- LLMReranker asks the injected LLM to score each candidate 0-3 (JSON array of
  {passage_id, score, reasoning}); strict parse (top-level list, dict items,
  passage_id+score present, numeric score) -> RerankError on violation; unknown
  ids silently skipped; [] is success; reranked score = LLM 0-3 score, metadata
  stays clean
- RerankError(RagMcpError) parallel to LLMError (bare, no fields)
- QueryReranker orchestrates config + fallback: backend raises -> original
  order + fallback flag into trace; returns pure list
- create_reranker: "none" -> NoneReranker; "llm" requires llm; "cross_encoder"
  -> CrossEncoderReranker (no model touched); unknown -> ValueError
- rag_service.query: rerank (bool|None) + reranker injection seam, wired after
  all three strategy branches

Prior art: tests/unit/test_llm.py (FakeLLM responses injection), tests/unit/
test_query_orchestration.py (offline_env fixture).
"""

from __future__ import annotations

import json

import pytest

from rag_mcp_server.core.errors import LLMError, RagMcpError, RerankError
from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.libs.llm.fake import FakeLLM
from rag_mcp_server.libs.reranker.base import BaseReranker, NoneReranker
from rag_mcp_server.libs.reranker.factory import create_reranker
from rag_mcp_server.libs.reranker.llm_reranker import LLMReranker
from rag_mcp_server.libs.reranker.orchestrator import QueryReranker


def _result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_path=f"docs/{chunk_id}.md",
        score=score,
        text=f"text of {chunk_id}",
        sources=["dense", "sparse"],
    )


# ---------------------------------------------------------------------------
# RerankError — a RagMcpError parallel to LLMError (ADR 0009 pt.4)
# ---------------------------------------------------------------------------
def test_rerank_error_is_rag_mcp_error():
    assert issubclass(RerankError, RagMcpError)


def test_rerank_error_is_parallel_to_llm_error():
    assert RerankError is not LLMError
    assert not issubclass(RerankError, LLMError)
    assert not issubclass(LLMError, RerankError)


# ---------------------------------------------------------------------------
# TestNoneReranker — keep original order (default fallback)
# ---------------------------------------------------------------------------
def test_none_reranker_keeps_order_and_top_k():
    results = [_result("a", 0.9), _result("b", 0.5), _result("c", 0.7)]
    out = NoneReranker().rerank("q", results, top_k=2)
    assert [r.chunk_id for r in out] == ["a", "b"]  # order unchanged, truncated to top_k
    assert out[0].score == 0.9  # scores untouched


def test_none_reranker_does_not_mutate_input_list():
    results = [_result("a"), _result("b")]
    NoneReranker().rerank("q", results, top_k=10)
    assert [r.chunk_id for r in results] == ["a", "b"]


def test_none_reranker_invalid_top_k_fails_fast():
    with pytest.raises(ValueError):
        NoneReranker().rerank("q", [_result("a")], top_k=0)
    with pytest.raises(ValueError):
        NoneReranker().rerank("q", [_result("a")], top_k=-1)


def test_none_reranker_top_k_larger_than_results_returns_all():
    results = [_result("a"), _result("b")]
    out = NoneReranker().rerank("q", results, top_k=10)
    assert [r.chunk_id for r in out] == ["a", "b"]


# ---------------------------------------------------------------------------
# TestLLMReranker — 0-3 scoring array; strict parse; fallback to RerankError
# ---------------------------------------------------------------------------
def test_llm_reranker_reorders_by_score_and_updates_score():
    # LLM scores c highest, a lowest -> order c, b, a
    payload = json.dumps(
        [
            {"passage_id": "a", "score": 1, "reasoning": "weak"},
            {"passage_id": "b", "score": 2, "reasoning": "ok"},
            {"passage_id": "c", "score": 3, "reasoning": "best"},
        ]
    )
    llm = _chatting_llm(payload)

    reranker = LLMReranker(llm=llm)
    results = [_result("a", 0.9), _result("b", 0.5), _result("c", 0.7)]
    out = reranker.rerank("q", results, top_k=3)

    assert [r.chunk_id for r in out] == ["c", "b", "a"]
    assert out[0].score == 3.0  # score = LLM score
    assert out[1].score == 2.0
    # contract fields preserved; no rerank trace in the result (RetrievalResult
    # has no metadata field — the type itself keeps the contract clean)
    assert out[0].text == "text of c"
    assert out[0].source_path == "docs/c.md"
    assert out[0].sources == ["dense", "sparse"]


def test_llm_reranker_scores_are_floats():
    payload = json.dumps([{"passage_id": "a", "score": 1.5, "reasoning": ""}])
    llm = _chatting_llm(payload)
    out = LLMReranker(llm=llm).rerank("q", [_result("a")], top_k=1)
    assert out[0].score == 1.5


def test_llm_reranker_unknown_id_silently_skipped():
    payload = json.dumps([{"passage_id": "ghost", "score": 3, "reasoning": ""}])
    llm = _chatting_llm(payload)
    out = LLMReranker(llm=llm).rerank("q", [_result("a")], top_k=1)
    assert out == []  # no known id referenced -> nothing reranked


def test_llm_reranker_empty_response_is_success():
    llm = _chatting_llm("[]")
    out = LLMReranker(llm=llm).rerank("q", [_result("a"), _result("b")], top_k=2)
    assert out == []


def test_llm_reranker_truncates_to_top_k():
    payload = json.dumps(
        [
            {"passage_id": "a", "score": 3, "reasoning": ""},
            {"passage_id": "b", "score": 2, "reasoning": ""},
            {"passage_id": "c", "score": 1, "reasoning": ""},
        ]
    )
    llm = _chatting_llm(payload)
    out = LLMReranker(llm=llm).rerank("q", [_result("a"), _result("b"), _result("c")], top_k=2)
    assert [r.chunk_id for r in out] == ["a", "b"]


# -- parse failures -> RerankError -----------------------------------------
@pytest.mark.parametrize("raw", ["not json", "{no quote", '"a string"', "42"])
def test_llm_reranker_non_json_or_scalar_raises_rerank_error(raw):
    llm = _chatting_llm(raw)
    with pytest.raises(RerankError):
        LLMReranker(llm=llm).rerank("q", [_result("a")], top_k=1)


def test_llm_reranker_top_level_not_list_raises():
    llm = _chatting_llm(json.dumps({"passage_id": "a", "score": 3}))
    with pytest.raises(RerankError):
        LLMReranker(llm=llm).rerank("q", [_result("a")], top_k=1)


def test_llm_reranker_item_not_dict_raises():
    llm = _chatting_llm(json.dumps(["a", "b"]))
    with pytest.raises(RerankError):
        LLMReranker(llm=llm).rerank("q", [_result("a")], top_k=1)


def test_llm_reranker_missing_score_key_raises():
    llm = _chatting_llm(json.dumps([{"passage_id": "a", "reasoning": ""}]))
    with pytest.raises(RerankError):
        LLMReranker(llm=llm).rerank("q", [_result("a")], top_k=1)


def test_llm_reranker_missing_passage_id_key_raises():
    llm = _chatting_llm(json.dumps([{"score": 3, "reasoning": ""}]))
    with pytest.raises(RerankError):
        LLMReranker(llm=llm).rerank("q", [_result("a")], top_k=1)


def test_llm_reranker_non_numeric_score_raises():
    llm = _chatting_llm(json.dumps([{"passage_id": "a", "score": "high", "reasoning": ""}]))
    with pytest.raises(RerankError):
        LLMReranker(llm=llm).rerank("q", [_result("a")], top_k=1)


def test_llm_reranker_llm_error_maps_to_rerank_error():
    class BoomLLM(FakeLLM):
        def _chat(self, messages: list[dict]) -> str:
            raise LLMError("provider down", provider="ollama")

    with pytest.raises(RerankError) as ei:
        LLMReranker(llm=BoomLLM()).rerank("q", [_result("a")], top_k=1)
    assert isinstance(ei.value.__cause__, LLMError)  # cause chain preserved


def test_llm_reranker_invalid_top_k_fails_fast():
    llm = _chatting_llm("[]")
    with pytest.raises(ValueError):
        LLMReranker(llm=llm).rerank("q", [_result("a")], top_k=0)


# ---------------------------------------------------------------------------
# TestQueryReranker — orchestration: fallback on RerankError, else reranked
# ---------------------------------------------------------------------------
def test_query_reranker_reranks_when_backend_succeeds():
    payload = json.dumps([{"passage_id": "a", "score": 1, "reasoning": ""},
                          {"passage_id": "b", "score": 3, "reasoning": ""}])
    results = [_result("a", 0.9), _result("b", 0.5)]
    qr = QueryReranker(LLMReranker(llm=_chatting_llm(payload)))
    out = qr.rerank("q", results, top_k=2)
    assert [r.chunk_id for r in out] == ["b", "a"]
    assert out[0].score == 3.0


def test_query_reranker_falls_back_to_original_order_on_backend_error():
    class BoomReranker(BaseReranker):
        def rerank(self, query, results, top_k):
            raise RerankError("backend down")

    results = [_result("a", 0.9), _result("b", 0.5)]
    qr = QueryReranker(BoomReranker())
    out = qr.rerank("q", results, top_k=2)
    assert [r.chunk_id for r in out] == ["a", "b"]  # original order kept
    assert out[0].score == 0.9  # scores untouched
    assert qr.last_fallback is True  # caller reads the flag for trace


def test_query_reranker_none_backend_keeps_order_and_no_fallback():
    results = [_result("a", 0.9), _result("b", 0.5)]
    qr = QueryReranker(NoneReranker())
    out = qr.rerank("q", results, top_k=2)
    assert [r.chunk_id for r in out] == ["a", "b"]
    assert qr.last_fallback is False


def test_query_reranker_truncates_to_top_k():
    results = [_result("a", 0.9), _result("b", 0.5), _result("c", 0.7)]
    qr = QueryReranker(NoneReranker())
    out = qr.rerank("q", results, top_k=1)
    assert len(out) == 1


def test_query_reranker_invalid_top_k_fails_fast():
    qr = QueryReranker(NoneReranker())
    with pytest.raises(ValueError):
        qr.rerank("q", [_result("a")], top_k=0)


# ---------------------------------------------------------------------------
# TestCreateReranker — factory
# ---------------------------------------------------------------------------
def test_create_reranker_none_returns_none_reranker():
    assert isinstance(create_reranker("none"), NoneReranker)


def test_create_reranker_llm_with_llm_returns_llm_reranker():
    rr = create_reranker("llm", llm=FakeLLM())
    assert isinstance(rr, LLMReranker)


def test_create_reranker_llm_without_llm_raises_value_error():
    with pytest.raises(ValueError):
        create_reranker("llm")


def test_create_reranker_cross_encoder_returns_cross_encoder_reranker():
    from rag_mcp_server.libs.reranker.cross_encoder import CrossEncoderReranker

    rr = create_reranker("cross_encoder")
    assert isinstance(rr, CrossEncoderReranker)
    assert rr._model is None  # constructing never loads the real model


def test_create_reranker_unknown_provider_raises_value_error():
    with pytest.raises(ValueError) as ei:
        create_reranker("bogus")
    assert "none" in str(ei.value)
    assert "llm" in str(ei.value)
    assert "cross_encoder" in str(ei.value)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _chatting_llm(payload: str) -> FakeLLM:
    """A FakeLLM whose _chat returns the canned payload (echo the table match)."""
    llm = FakeLLM()
    llm._chat = lambda messages: payload  # type: ignore[method-assign]
    return llm
