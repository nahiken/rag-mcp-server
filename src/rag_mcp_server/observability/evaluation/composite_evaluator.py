from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.libs.evaluator.base import BaseEvaluator

__all__ = ["CompositeEvaluator"]


class CompositeEvaluator(BaseEvaluator):
    def __init__(self, evaluators: Iterable[BaseEvaluator]):
        self._evaluators = list(evaluators)

    def evaluate(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        expected_chunk_ids: list[str],
        reference_answer: str = "",
    ) -> dict[str, float]:
        merged: dict[str, float] = {}
        for evaluator in self._evaluators:
            metrics = evaluator.evaluate(query, retrieved, expected_chunk_ids, reference_answer)
            for name, value in metrics.items():
                if name in merged:
                    raise ValueError(f"Duplicate evaluator metric: {name}")
                if not isinstance(value, Real):
                    raise TypeError(f"Evaluator metric {name!r} must be a number")
                merged[name] = float(value)
        return merged
