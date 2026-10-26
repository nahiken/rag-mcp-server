from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.libs.evaluator import BaseEvaluator

__all__ = ["EvalRunner", "GoldenTestCase", "load_golden_test_set"]


@dataclass(frozen=True)
class GoldenTestCase:
    query: str
    expected_chunk_ids: list[str]
    expected_sources: list[str]
    reference_answer: str


class EvalRunner:
    def run(
        self,
        test_set: list[GoldenTestCase],
        evaluator: BaseEvaluator,
        search_fn: Callable[[str], list[RetrievalResult]],
    ) -> dict[str, Any]:
        if not test_set:
            raise ValueError("golden test set must contain at least one case")

        per_query: list[dict[str, Any]] = []
        totals: dict[str, float] = {}
        for case in test_set:
            retrieved = search_fn(case.query)
            metrics = evaluator.evaluate(case.query, retrieved, case.expected_chunk_ids, case.reference_answer)
            self._validate_required_metrics(metrics)

            for name, value in metrics.items():
                if not isinstance(value, Real):
                    raise TypeError(f"Evaluator metric {name!r} must be a number")
                totals[name] = totals.get(name, 0.0) + float(value)

            per_query.append(
                {
                    "query": case.query,
                    "expected_chunk_ids": list(case.expected_chunk_ids),
                    "expected_sources": list(case.expected_sources),
                    "retrieved_chunk_ids": [result.chunk_id for result in retrieved],
                    "hit": float(metrics["hit_rate"]),
                    "rr": float(metrics["mrr"]),
                }
            )

        count = len(test_set)
        report = {name: value / count for name, value in totals.items()}
        report["per_query"] = per_query
        return report

    @staticmethod
    def _validate_required_metrics(metrics: dict[str, float]) -> None:
        if "hit_rate" not in metrics or "mrr" not in metrics:
            raise ValueError("Evaluator metrics must include hit_rate and mrr")


def load_golden_test_set(path: Path | str) -> list[GoldenTestCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("golden test set must be a non-empty list")
    return [_golden_case(item, index) for index, item in enumerate(payload, start=1)]


def _golden_case(item: Any, index: int) -> GoldenTestCase:
    if not isinstance(item, dict):
        raise ValueError(f"golden test case #{index} must be an object")

    required = ("query", "expected_chunk_ids", "expected_sources", "reference_answer")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"golden test case #{index} missing required fields: {', '.join(missing)}")

    query = item["query"]
    expected_chunk_ids = item["expected_chunk_ids"]
    expected_sources = item["expected_sources"]
    reference_answer = item["reference_answer"]

    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"golden test case #{index} query must be a non-empty string")
    if not _is_string_list(expected_chunk_ids) or not expected_chunk_ids:
        raise ValueError(f"golden test case #{index} expected_chunk_ids must be a non-empty list of strings")
    if not _is_string_list(expected_sources):
        raise ValueError(f"golden test case #{index} expected_sources must be a list of strings")
    if not isinstance(reference_answer, str):
        raise ValueError(f"golden test case #{index} reference_answer must be a string")

    return GoldenTestCase(
        query=query,
        expected_chunk_ids=list(expected_chunk_ids),
        expected_sources=list(expected_sources),
        reference_answer=reference_answer,
    )


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
