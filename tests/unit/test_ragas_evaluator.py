"""Tests for RagasEvaluator — optional Answer-scoring extension (P10.7 T3-01).

Red/green contract (`.scratch/phase10/eval-system/t3-issues/01-ragas-evaluator.md`):
- RagasEvaluator(BaseEvaluator); evaluate returns dict[str, float] scalars,
  contract-identical to CustomEvaluator/CompositeEvaluator/EvalRunner
- scoring backend is injectable: default = real ragas (lazy-imported), tests
  inject a mock backend so scoring logic is testable without ragas installed
- default backend missing ragas -> clear ImportError (not bare ModuleNotFound);
  even an injected LLM does not bypass the missing-dependency guard (the real
  ragas backend is the default scoring path)
- evaluate() without an injected backend guards on RagasScoringBackend() first,
  then falls back to score_with_llm (installing ragas but not injecting a
  backend -> scalars, never NotImplementedError)
- score_with_llm: query/reference/golden ids + injected BaseLLM -> parsed
  scalars clamped to [0,1]; fenced JSON tolerated; bad output -> ValueError;
  LLMError bubbles; the golden ids appear in the prompt
- importing rag_mcp_server.observability.evaluation never imports ragas

Prior art: tests/unit/test_reranker.py (injected LLM + strict parse),
tests/unit/test_evaluator.py (hand-built RetrievalResult lists).
"""

from __future__ import annotations

import importlib.util

import pytest

from rag_mcp_server.core.errors import LLMError
from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.observability.evaluation import ragas_evaluator
from rag_mcp_server.observability.evaluation.composite_evaluator import CompositeEvaluator
from rag_mcp_server.observability.evaluation.ragas_evaluator import RagasEvaluator, RagasScoringBackend
from rag_mcp_server.libs.llm.fake import FakeLLM


def _ragas_installed() -> bool:
    return importlib.util.find_spec("ragas") is not None


def _result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_path=f"docs/{chunk_id}.md",
        score=score,
        text=f"text for {chunk_id}",
    )


def _llm(payload: str) -> FakeLLM:
    llm = FakeLLM()
    llm._chat = lambda messages: payload  # type: ignore[method-assign]
    return llm


def _backend(metrics: dict[str, float]):
    class StubBackend:
        def score(self, query: str, retrieved, expected_chunk_ids: list[str], reference_answer: str):
            return dict(metrics)

    return StubBackend()


# ---------------------------------------------------------------------------
# Default backend: real ragas is missing here -> clear ImportError
# ---------------------------------------------------------------------------
def test_ragas_backend_import_error_when_ragas_missing():
    if _ragas_installed():  # pragma: no cover
        pytest.skip("ragas installed; ImportError path not applicable")
    with pytest.raises(ImportError, match="ragas"):
        RagasScoringBackend()


def test_evaluate_raises_import_error_when_ragas_missing():
    if _ragas_installed():  # pragma: no cover
        pytest.skip("ragas installed; ImportError path not applicable")
    evaluator = RagasEvaluator(llm=FakeLLM())
    with pytest.raises(ImportError, match="ragas"):
        evaluator.evaluate(
            query="q",
            retrieved=[_result("a::0")],
            expected_chunk_ids=["a::0"],
            reference_answer="answer",
        )


def test_evaluate_import_error_not_bypassed_by_injected_llm():
    if _ragas_installed():  # pragma: no cover
        pytest.skip("ragas installed; ImportError path not applicable")
    evaluator = RagasEvaluator(llm=_llm('{"faithfulness": 0.9}'))
    with pytest.raises(ImportError, match="ragas"):
        evaluator.evaluate(
            query="q",
            retrieved=[_result("a::0")],
            expected_chunk_ids=["a::0"],
            reference_answer="answer",
        )


def test_evaluate_without_backend_falls_back_to_llm_when_ragas_installed(monkeypatch):
    """R1 wiring: with ragas installed but no backend injected, evaluate()
    returns scalars via score_with_llm — never NotImplementedError."""
    guard = object()
    monkeypatch.setattr(ragas_evaluator, "RagasScoringBackend", lambda: guard)
    llm = _llm('{"faithfulness": 0.9, "answer_relevancy": 0.7}')
    evaluator = RagasEvaluator(llm=llm)
    metrics = evaluator.evaluate(
        query="q",
        retrieved=[_result("a::0")],
        expected_chunk_ids=["a::0"],
        reference_answer="answer",
    )
    assert metrics == {"faithfulness": 0.9, "answer_relevancy": 0.7}


# ---------------------------------------------------------------------------
# evaluate with injected mock backend (no ragas needed)
# ---------------------------------------------------------------------------
def test_evaluate_returns_backend_scalars_with_injected_backend():
    evaluator = RagasEvaluator(
        llm=FakeLLM(),
        scoring_backend=_backend({"faithfulness": 0.8, "answer_relevancy": 0.7}),
    )
    metrics = evaluator.evaluate(
        query="q",
        retrieved=[_result("a::0")],
        expected_chunk_ids=["a::0"],
        reference_answer="answer",
    )
    assert metrics == {"faithfulness": 0.8, "answer_relevancy": 0.7}


def test_evaluate_passes_through_all_arguments_to_backend():
    captured = {}

    class CaptureBackend:
        def score(self, query, retrieved, expected_chunk_ids, reference_answer):
            captured.update(
                query=query,
                retrieved=list(retrieved),
                expected_chunk_ids=list(expected_chunk_ids),
                reference_answer=reference_answer,
            )
            return {"faithfulness": 0.5}

    evaluator = RagasEvaluator(llm=FakeLLM(), scoring_backend=CaptureBackend())
    evaluator.evaluate(
        query="q",
        retrieved=[_result("a::0")],
        expected_chunk_ids=["a::0"],
        reference_answer="answer",
    )
    assert captured == {
        "query": "q",
        "retrieved": [_result("a::0")],
        "expected_chunk_ids": ["a::0"],
        "reference_answer": "answer",
    }


# ---------------------------------------------------------------------------
# score_with_llm — thin LLM scoring
# ---------------------------------------------------------------------------
def test_score_with_llm_parses_scalars():
    llm = _llm('{"faithfulness": 0.9, "answer_relevancy": 0.75}')
    evaluator = RagasEvaluator(llm=llm)
    metrics = evaluator.score_with_llm("q", [_result("a::0")], ["a::0"], "answer")
    assert metrics == {"faithfulness": 0.9, "answer_relevancy": 0.75}


def test_score_with_llm_tolerates_fenced_json():
    llm = _llm('```json\n{"faithfulness": 0.9}\n```')
    evaluator = RagasEvaluator(llm=llm)
    metrics = evaluator.score_with_llm("q", [_result("a::0")], ["a::0"], "answer")
    assert metrics == {"faithfulness": 0.9}


def test_score_with_llm_clamps_values_to_zero_one():
    llm = _llm('{"faithfulness": 1.4, "answer_relevancy": -0.2}')
    evaluator = RagasEvaluator(llm=llm)
    metrics = evaluator.score_with_llm("q", [_result("a::0")], ["a::0"], "answer")
    assert metrics == {"faithfulness": 1.0, "answer_relevancy": 0.0}


def test_score_with_llm_prompt_includes_golden_chunk_ids():
    captured = {}

    class CaptureLLM(FakeLLM):
        def _chat(self, messages: list[dict]) -> str:
            captured["prompt"] = messages[-1]["content"]
            return '{"faithfulness": 0.9}'

    evaluator = RagasEvaluator(llm=CaptureLLM())
    evaluator.score_with_llm("q", [_result("a::0")], ["a::0", "b::0"], "answer")
    assert "Golden chunks: a::0, b::0" in captured["prompt"]


def test_score_with_llm_bad_json_raises_value_error():
    llm = _llm("not json")
    evaluator = RagasEvaluator(llm=llm)
    with pytest.raises(ValueError, match="JSON"):
        evaluator.score_with_llm("q", [_result("a::0")], ["a::0"], "answer")


def test_score_with_llm_non_numeric_value_raises_value_error():
    llm = _llm('{"faithfulness": "high"}')
    evaluator = RagasEvaluator(llm=llm)
    with pytest.raises(ValueError, match="数值"):
        evaluator.score_with_llm("q", [_result("a::0")], ["a::0"], "answer")


def test_score_with_llm_empty_result_raises_value_error():
    llm = _llm("{}")
    evaluator = RagasEvaluator(llm=llm)
    with pytest.raises(ValueError, match="至少需要"):
        evaluator.score_with_llm("q", [_result("a::0")], ["a::0"], "answer")


def test_score_with_llm_llm_error_propagates():
    class BoomLLM(FakeLLM):
        def _chat(self, messages: list[dict]) -> str:
            raise LLMError("provider down", provider="ollama")

    evaluator = RagasEvaluator(llm=BoomLLM())
    with pytest.raises(LLMError):
        evaluator.score_with_llm("q", [_result("a::0")], ["a::0"], "answer")


# ---------------------------------------------------------------------------
# Composite + import-side-effect-free
# ---------------------------------------------------------------------------
def test_ragas_evaluator_composes_with_custom_evaluator_via_composite():
    evaluator = CompositeEvaluator(
        [
            RagasEvaluator(llm=FakeLLM(), scoring_backend=_backend({"faithfulness": 0.8})),
            _custom_evaluator(),
        ]
    )
    metrics = evaluator.evaluate(
        query="q",
        retrieved=[_result("a::0")],
        expected_chunk_ids=["a::0"],
        reference_answer="answer",
    )
    assert metrics == {"faithfulness": 0.8, "hit_rate": 1.0, "mrr": 1.0}


def test_importing_evaluation_package_does_not_import_ragas(monkeypatch):
    import sys

    sys.modules["ragas"] = None
    try:
        import rag_mcp_server.observability.evaluation as evaluation
        assert hasattr(evaluation, "RagasEvaluator")
        assert "ragas" not in sys.modules or sys.modules["ragas"] is None
    finally:
        del sys.modules["ragas"]


def _custom_evaluator():
    from rag_mcp_server.libs.evaluator.custom import CustomEvaluator

    return CustomEvaluator()
