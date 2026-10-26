from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.libs.evaluator import BaseEvaluator
from rag_mcp_server.observability.evaluation import EvalRunner, GoldenTestCase, load_golden_test_set


def _result(chunk_id: str, text: str = "retrieved text") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_path=f"docs/{chunk_id}.md",
        score=1.0,
        text=text,
    )


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_golden_test_set_loads_valid_cases(tmp_path: Path):
    path = tmp_path / "golden.json"
    _write_json(
        path,
        [
            {
                "query": "如何配置检索？",
                "expected_chunk_ids": ["retrieval-guide-md::0"],
                "expected_sources": ["retrieval-guide.md"],
                "reference_answer": "",
            }
        ],
    )

    cases = load_golden_test_set(path)

    assert cases == [
        GoldenTestCase(
            query="如何配置检索？",
            expected_chunk_ids=["retrieval-guide-md::0"],
            expected_sources=["retrieval-guide.md"],
            reference_answer="",
        )
    ]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{"expected_chunk_ids": ["a::0"], "expected_sources": [], "reference_answer": ""}],
        [{"query": "q", "expected_sources": [], "reference_answer": ""}],
        [{"query": "q", "expected_chunk_ids": ["a::0"], "reference_answer": ""}],
        [{"query": "q", "expected_chunk_ids": ["a::0"], "expected_sources": []}],
        [{"query": "", "expected_chunk_ids": ["a::0"], "expected_sources": [], "reference_answer": ""}],
        [{"query": "q", "expected_chunk_ids": [], "expected_sources": [], "reference_answer": ""}],
        [{"query": "q", "expected_chunk_ids": "a::0", "expected_sources": [], "reference_answer": ""}],
        [{"query": "q", "expected_chunk_ids": ["a::0"], "expected_sources": "a.md", "reference_answer": ""}],
        [{"query": "q", "expected_chunk_ids": ["a::0"], "expected_sources": [], "reference_answer": None}],
    ],
)
def test_load_golden_test_set_rejects_bad_cases(tmp_path: Path, payload):
    path = tmp_path / "golden.json"
    _write_json(path, payload)

    with pytest.raises(ValueError):
        load_golden_test_set(path)


class MetricsEvaluator(BaseEvaluator):
    def __init__(self, metrics_by_query: dict[str, dict[str, float]]):
        self.metrics_by_query = metrics_by_query
        self.calls: list[tuple[str, list[RetrievalResult], list[str], str]] = []

    def evaluate(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        expected_chunk_ids: list[str],
        reference_answer: str = "",
    ) -> dict[str, float]:
        self.calls.append((query, retrieved, expected_chunk_ids, reference_answer))
        return dict(self.metrics_by_query[query])


def test_eval_runner_runs_search_and_aggregates_metrics_per_query():
    test_set = [
        GoldenTestCase("alpha query", ["alpha-md::0"], ["alpha.md"], "alpha reference"),
        GoldenTestCase("beta query", ["beta-md::0"], ["beta.md"], ""),
    ]
    evaluator = MetricsEvaluator(
        {
            "alpha query": {"hit_rate": 1.0, "mrr": 1.0, "coverage": 0.5},
            "beta query": {"hit_rate": 0.0, "mrr": 0.0, "coverage": 1.0},
        }
    )

    def search(query: str) -> list[RetrievalResult]:
        return {
            "alpha query": [_result("alpha-md::0"), _result("other-md::0")],
            "beta query": [_result("other-md::0")],
        }[query]

    report = EvalRunner().run(test_set, evaluator, search)

    assert report["hit_rate"] == 0.5
    assert report["mrr"] == 0.5
    assert report["coverage"] == 0.75
    assert report["per_query"] == [
        {
            "query": "alpha query",
            "expected_chunk_ids": ["alpha-md::0"],
            "expected_sources": ["alpha.md"],
            "retrieved_chunk_ids": ["alpha-md::0", "other-md::0"],
            "hit": 1.0,
            "rr": 1.0,
        },
        {
            "query": "beta query",
            "expected_chunk_ids": ["beta-md::0"],
            "expected_sources": ["beta.md"],
            "retrieved_chunk_ids": ["other-md::0"],
            "hit": 0.0,
            "rr": 0.0,
        },
    ]
    assert "retrieved text" not in json.dumps(report)
    assert evaluator.calls[0][0] == "alpha query"
    assert evaluator.calls[0][2] == ["alpha-md::0"]
    assert evaluator.calls[0][3] == "alpha reference"


@pytest.mark.parametrize("metrics", [{"hit_rate": 1.0}, {"mrr": 1.0}])
def test_eval_runner_requires_hit_rate_and_mrr(metrics: dict[str, float]):
    evaluator = MetricsEvaluator({"query": metrics})

    with pytest.raises(ValueError, match="hit_rate.*mrr"):
        EvalRunner().run(
            [GoldenTestCase("query", ["doc-md::0"], ["doc.md"], "")],
            evaluator,
            lambda query: [_result("doc-md::0")],
        )


def test_eval_runner_propagates_search_errors():
    evaluator = MetricsEvaluator({"query": {"hit_rate": 1.0, "mrr": 1.0}})

    def broken_search(query: str) -> list[RetrievalResult]:
        raise RuntimeError("search exploded")

    with pytest.raises(RuntimeError, match="search exploded"):
        EvalRunner().run(
            [GoldenTestCase("query", ["doc-md::0"], ["doc.md"], "")],
            evaluator,
            broken_search,
        )
