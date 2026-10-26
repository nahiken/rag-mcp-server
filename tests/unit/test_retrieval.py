from __future__ import annotations

import unittest

from rag_mcp_server.core.types import Chunk
from rag_mcp_server.retrieval.keyword_retriever import DEFAULT_TOP_K, keyword_search


def make_chunk(document_id: str, text: str, order: int = 0, source_path: str = "notes.txt") -> Chunk:
    return Chunk(
        document_id=document_id,
        text=text,
        source_path=source_path,
        order=order,
        metadata={"file_type": "txt"},
    )


class KeywordSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            make_chunk(
                "rag-notes",
                "RAG 使用关键词检索 baseline，检索结果按 score 排序。",
                order=0,
                source_path="docs/rag-notes.md",
            ),
            make_chunk(
                "mcp-notes",
                "MCP 是 Model Context Protocol，用于连接工具与数据。",
                order=0,
                source_path="docs/mcp-notes.md",
            ),
            make_chunk(
                "rag-notes",
                "关键词检索会返回 top 3 结果，并附带引用来源。",
                order=1,
                source_path="docs/rag-notes.md",
            ),
        ]

    def test_score_is_keyword_match_count(self) -> None:
        results = keyword_search("RAG", self.chunks)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "rag-notes::0")
        self.assertEqual(results[0].score, 1.0)
        self.assertIn("RAG", results[0].text)

    def test_chunk_with_more_keyword_hits_scores_higher(self) -> None:
        # "检索" appears twice in chunk0 ("检索" twice) and once in chunk2
        results = keyword_search("检索", self.chunks)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "rag-notes::0")
        self.assertEqual(results[0].score, 2.0)
        self.assertEqual(results[1].chunk_id, "rag-notes::1")
        self.assertEqual(results[1].score, 1.0)

    def test_results_sorted_by_score_descending(self) -> None:
        chunks = [
            make_chunk("d0", "apple 一个"),
            make_chunk("d1", "apple apple 两个"),
            make_chunk("d2", "apple apple apple 三个"),
        ]
        results = keyword_search("apple", chunks)
        self.assertEqual([r.score for r in results], [3.0, 2.0, 1.0])
        self.assertEqual([r.chunk_id for r in results], ["d2::0", "d1::0", "d0::0"])

    def test_top_k_limits_results(self) -> None:
        chunks = [make_chunk(f"d{i}", "关键词 命中 内容", order=i) for i in range(5)]
        results = keyword_search("关键词", chunks, top_k=3)
        self.assertEqual(len(results), 3)

    def test_default_top_k_is_3(self) -> None:
        self.assertEqual(DEFAULT_TOP_K, 3)

    def test_zero_score_chunks_are_excluded(self) -> None:
        results = keyword_search("完全不存在的词", self.chunks)
        self.assertEqual(results, [])

    def test_no_result_returns_empty_list_not_exception(self) -> None:
        results = keyword_search("zzz-not-found", self.chunks)
        self.assertIsInstance(results, list)
        self.assertEqual(results, [])

    def test_result_carries_source_path_and_text(self) -> None:
        results = keyword_search("MCP", self.chunks)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.source_path, "docs/mcp-notes.md")
        self.assertIn("MCP", result.text)

    def test_scores_are_floats(self) -> None:
        results = keyword_search("RAG", self.chunks)
        self.assertIsInstance(results[0].score, float)

    def test_each_result_traces_back_to_a_real_chunk(self) -> None:
        """引用来源可追溯：返回的 chunk_id/source_path 必须对应 store 中真实存在的 chunk。"""
        results = keyword_search("检索", self.chunks)
        by_id = {c.chunk_id: c for c in self.chunks}
        for result in results:
            self.assertIn(result.chunk_id, by_id)
            chunk = by_id[result.chunk_id]
            self.assertEqual(result.source_path, chunk.source_path)
            self.assertEqual(result.text, chunk.text)


if __name__ == "__main__":
    unittest.main()
