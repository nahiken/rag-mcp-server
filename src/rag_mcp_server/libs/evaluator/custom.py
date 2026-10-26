from __future__ import annotations

from rag_mcp_server.core.types import RetrievalResult
from rag_mcp_server.libs.evaluator.base import BaseEvaluator

__all__ = ["CustomEvaluator"]


class CustomEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        expected_chunk_ids: list[str],
        reference_answer: str = "",
    ) -> dict[str, float]:
        if not expected_chunk_ids:
            raise ValueError("expected_chunk_ids must contain at least one chunk id")

        expected = set(expected_chunk_ids)
        reciprocal_rank = 0.0
        for rank, result in enumerate(retrieved, start=1):
            if result.chunk_id in expected:
                reciprocal_rank = 1.0 / rank
                break

        hit = 1.0 if reciprocal_rank else 0.0
        return {"hit_rate": hit, "mrr": reciprocal_rank}
