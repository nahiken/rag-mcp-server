from __future__ import annotations

import pytest

from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.libs.evaluator import BaseEvaluator, CustomEvaluator
from rag_mcp_server.observability.evaluation.composite_evaluator import CompositeEvaluator


def _result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_path=f"docs/{chunk_id}.md",
        score=score,
        text=f"text for {chunk_id}",
    )


def test_custom_evaluator_hits_when_any_expected_chunk_is_retrieved():
    metrics = CustomEvaluator().evaluate(
        query="where is the setup guide?",
        retrieved=[_result("intro::0"), _result("setup::1")],
        expected_chunk_ids=["setup::1"],
    )

    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}


def test_custom_evaluator_misses_when_no_expected_chunk_is_retrieved():
    metrics = CustomEvaluator().evaluate(
        query="where is the setup guide?",
        retrieved=[_result("intro::0"), _result("usage::1")],
        expected_chunk_ids=["setup::1"],
    )

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


def test_custom_evaluator_accepts_multiple_expected_chunks_and_uses_earliest_match():
    metrics = CustomEvaluator().evaluate(
        query="how do I configure storage?",
        retrieved=[_result("overview::0"), _result("storage::2"), _result("storage::1")],
        expected_chunk_ids=["storage::1", "storage::2"],
    )

    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}


def test_custom_evaluator_scores_all_results_passed_to_it():
    metrics = CustomEvaluator().evaluate(
        query="how do I configure storage?",
        retrieved=[_result("overview::0"), _result("usage::1"), _result("storage::2")],
        expected_chunk_ids=["storage::2"],
    )

    assert metrics == {"hit_rate": 1.0, "mrr": pytest.approx(1 / 3)}


def test_custom_evaluator_requires_at_least_one_expected_chunk_id():
    with pytest.raises(ValueError):
        CustomEvaluator().evaluate(
            query="empty expectation",
            retrieved=[_result("overview::0")],
            expected_chunk_ids=[],
        )


class StaticEvaluator(BaseEvaluator):
    def __init__(self, metrics: dict[str, float]):
        self.metrics = metrics

    def evaluate(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        expected_chunk_ids: list[str],
        reference_answer: str = "",
    ) -> dict[str, float]:
        return dict(self.metrics)


def test_composite_evaluator_merges_scalar_metrics():
    evaluator = CompositeEvaluator(
        [
            StaticEvaluator({"hit_rate": 1.0}),
            StaticEvaluator({"mrr": 0.5}),
        ]
    )

    metrics = evaluator.evaluate(
        query="where is the setup guide?",
        retrieved=[_result("setup::1")],
        expected_chunk_ids=["setup::1"],
        reference_answer="See the setup guide.",
    )

    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}


def test_composite_evaluator_fails_fast_on_duplicate_metric_names():
    evaluator = CompositeEvaluator(
        [
            StaticEvaluator({"hit_rate": 1.0}),
            StaticEvaluator({"hit_rate": 0.0}),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate evaluator metric"):
        evaluator.evaluate(
            query="where is the setup guide?",
            retrieved=[_result("setup::1")],
            expected_chunk_ids=["setup::1"],
        )


def test_composite_evaluator_rejects_non_scalar_report_values():
    evaluator = CompositeEvaluator([StaticEvaluator({"hit_rate": 1.0, "per_query": []})])

    with pytest.raises(TypeError, match="must be a number"):
        evaluator.evaluate(
            query="where is the setup guide?",
            retrieved=[_result("setup::1")],
            expected_chunk_ids=["setup::1"],
        )
