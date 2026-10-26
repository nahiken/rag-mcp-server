from __future__ import annotations

from abc import ABC, abstractmethod

from rag_mcp_server.core.types import RetrievalResult

__all__ = ["BaseEvaluator"]


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved: list[RetrievalResult],
        expected_chunk_ids: list[str],
        reference_answer: str = "",
    ) -> dict[str, float]:
        raise NotImplementedError
