"""Tests for the optional chunking variants (ADR 0020 T5): fixed-length windows
and semantic (embedding-similarity) grouping.

Red/green contract:
- chunk_document_fixed: uniform chunk_size windows + overlap cover the whole
  text without losing characters; Chunks are isomorphic to chunk_document
  (file_type + optional image_refs metadata); empty document -> [].
- chunk_document_semantic: paragraphs are grouped by deterministic embedding
  similarity (cosine >= threshold merges, < threshold splits); a group is
  capped at max_group paragraphs; an over-long paragraph/group is continued by
  max_chars; embedding failure/wrong-shaped result degrades to paragraph
  chunking (chunk_document); empty document -> [].
- default chunk_document behaviour is unchanged (regression).

All embeddings are deterministic fakes so the suite runs offline and repeats.
"""

from __future__ import annotations

import math

import pytest

from rag_mcp_server.core.types import Document
from rag_mcp_server.ingestion.chunker import chunk_document
from rag_mcp_server.ingestion.splitter import (
    chunk_document_fixed,
    chunk_document_semantic,
)
from rag_mcp_server.libs.embedding.base import BaseEmbedding
from rag_mcp_server.libs.embedding.fake import FakeEmbedding


def make_doc(text: str, file_type: str = "txt") -> Document:
    return Document(
        id="d",
        text=text,
        source_path="d.txt" if file_type == "txt" else "d.md",
        file_type=file_type,
    )


class TopicFakeEmbedding(BaseEmbedding):
    """Deterministic keyword embedding: paragraphs sharing a topic keyword get
    the same unit vector (cosine 1.0); different topics get orthogonal vectors
    (cosine 0.0). ``fail=True`` simulates an unavailable embedding provider."""

    _TOPICS = ["RAG", "MCP", "向量"]

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail
        self._dim = len(self._TOPICS)

    def dim(self) -> int:
        return self._dim

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            raise RuntimeError("embedding unavailable (offline)")
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for i, topic in enumerate(self._TOPICS):
            if topic in text:
                vec[i] = 1.0
        return vec


class WrongCountEmbedding(BaseEmbedding):
    """An embedding that always returns a single vector (unusable shape)."""

    def dim(self) -> int:
        return 2

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]]


# -- chunk_document_fixed ----------------------------------------------------
class TestFixedLength:
    def test_concat_equals_original_when_no_overlap(self) -> None:
        text = "a" * 100 + "b" * 100 + "c" * 50  # 250 chars
        chunks = chunk_document_fixed(make_doc(text), chunk_size=100, overlap=0)
        assert "".join(c.text for c in chunks) == text
        assert [c.order for c in chunks] == [0, 1, 2]
        assert [c.chunk_id for c in chunks] == ["d::0", "d::1", "d::2"]

    def test_overlap_windows_cover_every_character(self) -> None:
        text = "x" * 250
        chunk_size, overlap = 100, 25
        chunks = chunk_document_fixed(make_doc(text), chunk_size=chunk_size, overlap=overlap)
        step = chunk_size - overlap
        # windows start at k*step while start < len(text)
        assert len(chunks) == math.ceil(len(text) / step)  # ceil(250/75)=4
        # first window starts at the beginning, last reaches the end (no loss)
        assert chunks[0].text == text[:100]
        assert chunks[-1].text.endswith(text[-25:])
        for k, chunk in enumerate(chunks):
            start = k * step
            assert chunk.text == text[start : start + chunk_size]

    def test_chunk_count_matches_window_count(self) -> None:
        text = "z" * 1000
        chunk_size, overlap = 300, 100
        step = chunk_size - overlap  # 200
        expected = math.ceil(len(text) / step)
        chunks = chunk_document_fixed(make_doc(text), chunk_size=chunk_size, overlap=overlap)
        assert len(chunks) == expected
        assert [c.text for c in chunks][0] == text[:300]
        assert chunks[-1].text == text[800:1000]

    def test_metadata_carries_file_type_and_image_refs(self) -> None:
        text = "A" * 50 + " [IMAGE: fig1] " + "B" * 50
        chunks = chunk_document_fixed(make_doc(text), chunk_size=500, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].metadata["file_type"] == "txt"
        assert chunks[0].metadata["image_refs"] == ["fig1"]
        assert chunks[0].source_path == "d.txt"

    def test_empty_document_returns_empty(self) -> None:
        assert chunk_document_fixed(make_doc("")) == []


# -- chunk_document_semantic ------------------------------------------------
class TestSemantic:
    def test_splits_adjacent_paragraphs_on_low_similarity(self) -> None:
        doc = make_doc("RAG 段一\n\nRAG 段二\n\nMCP 段三")
        chunks = chunk_document_semantic(doc, TopicFakeEmbedding(), max_chars=800, threshold=0.6)
        assert [c.text for c in chunks] == ["RAG 段一\n\nRAG 段二", "MCP 段三"]
        assert [c.order for c in chunks] == [0, 1]

    def test_merges_high_similarity_paragraphs_into_one_chunk(self) -> None:
        doc = make_doc("RAG 段一\n\nRAG 段二\n\nRAG 段三")
        chunks = chunk_document_semantic(doc, TopicFakeEmbedding(), max_chars=800, threshold=0.6)
        assert len(chunks) == 1
        assert chunks[0].text == "RAG 段一\n\nRAG 段二\n\nRAG 段三"

    def test_over_long_paragraph_continued_by_max_chars(self) -> None:
        text = "RAG " + "x" * 2000  # 2004 chars > max_chars
        chunks = chunk_document_semantic(
            make_doc(text), TopicFakeEmbedding(), max_chars=800, threshold=0.6
        )
        assert "".join(c.text for c in chunks) == text
        assert all(len(c.text) <= 800 for c in chunks)
        assert len(chunks) == 3  # ceil(2004/800)

    def test_max_group_caps_paragraphs_per_chunk(self) -> None:
        doc = make_doc("RAG p1\n\nRAG p2\n\nRAG p3\n\nRAG p4\n\nRAG p5")
        chunks = chunk_document_semantic(
            doc, TopicFakeEmbedding(), max_chars=800, threshold=0.6, max_group=3
        )
        assert len(chunks) == 2
        assert chunks[0].text == "RAG p1\n\nRAG p2\n\nRAG p3"
        assert chunks[1].text == "RAG p4\n\nRAG p5"

    def test_falls_back_to_paragraph_chunking_on_embedding_error(self) -> None:
        doc = make_doc("段一\n\n段二\n\n段三")
        chunks = chunk_document_semantic(doc, TopicFakeEmbedding(fail=True), max_chars=800)
        assert chunks == chunk_document(doc, max_chars=800)
        assert [c.text for c in chunks] == ["段一", "段二", "段三"]

    def test_falls_back_when_embedding_returns_wrong_shape(self) -> None:
        doc = make_doc("段一\n\n段二\n\n段三")
        chunks = chunk_document_semantic(doc, WrongCountEmbedding(), max_chars=800)
        assert chunks == chunk_document(doc, max_chars=800)

    def test_metadata_carries_file_type_and_image_refs(self) -> None:
        doc = make_doc("RAG 内容 [IMAGE: fig1]\n\nRAG 更多")
        chunks = chunk_document_semantic(doc, TopicFakeEmbedding(), max_chars=800, threshold=0.6)
        assert chunks[0].metadata["file_type"] == "txt"
        assert chunks[0].metadata["image_refs"] == ["fig1"]

    def test_empty_document_returns_empty(self) -> None:
        assert chunk_document_semantic(make_doc(""), TopicFakeEmbedding()) == []


# -- regression: default chunk_document unchanged ---------------------------
class TestDefaultChunkDocument:
    def test_paragraph_splitting_unchanged(self) -> None:
        doc = make_doc("第一段。\n\n第二段。\n\n第三段。", file_type="md")
        chunks = chunk_document(doc)
        assert len(chunks) == 3
        assert [c.text for c in chunks] == ["第一段。", "第二段。", "第三段。"]
        assert [c.order for c in chunks] == [0, 1, 2]
        assert chunks[0].metadata == {"file_type": "md"}

    def test_long_paragraph_continuation_unchanged(self) -> None:
        doc = make_doc("x" * 2000)
        chunks = chunk_document(doc, max_chars=800)
        assert len(chunks) == 3
        assert "".join(c.text for c in chunks) == "x" * 2000


# -- pipeline wiring (settings-driven splitter selection) -------------------
class TestPipelineSplitter:
    def _pipeline(self, tmp_path, monkeypatch, splitter_cfg: dict):
        import rag_mcp_server.config.settings as settings_mod
        from rag_mcp_server.config.settings import DEFAULTS
        from rag_mcp_server.ingestion.pipeline import IngestPipeline
        from rag_mcp_server.libs.vector_store.chroma_store import ChromaVectorStore

        def fake_load_settings():
            data = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
            data["trace"] = {"enabled": False, "dir": "logs"}
            data["ingestion"].update(splitter_cfg)
            return data

        monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)
        embedding = FakeEmbedding(dim=4)
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "storage" / "chroma"), collection="test_splitter"
        )
        pipe = IngestPipeline(
            embedding=embedding,
            store=store,
            index_dir=tmp_path / "storage" / "index",
            collection="test_splitter",
        )
        return pipe

    def test_default_splitter_paragraph_unchanged(self, tmp_path, monkeypatch) -> None:
        pipe = self._pipeline(tmp_path, monkeypatch, {"splitter": "paragraph"})
        src = tmp_path / "source"
        src.mkdir()
        (src / "notes.txt").write_text("第一段。\n\n第二段。\n\n第三段。", encoding="utf-8")
        summary = pipe.ingest(src, base=src)
        assert summary.total_chunks == 3
        ids = [f"test_splitter::notes-txt::{i}" for i in range(3)]
        texts = [r.text for r in pipe.store.get_by_ids(ids)]
        assert texts == ["第一段。", "第二段。", "第三段。"]

    def test_settings_splitter_fixed_length_wired(self, tmp_path, monkeypatch) -> None:
        pipe = self._pipeline(
            tmp_path, monkeypatch, {"splitter": "fixed_length", "chunk_size": 20, "overlap": 0}
        )
        src = tmp_path / "source"
        src.mkdir()
        (src / "notes.txt").write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
        summary = pipe.ingest(src, base=src)
        assert summary.total_chunks == 2
        records = pipe.store.get_by_ids(["test_splitter::notes-txt::0", "test_splitter::notes-txt::1"])
        assert {r.text for r in records} == {"abcdefghijklmnopqrst", "uvwxyz"}

    def test_unknown_splitter_falls_back_to_paragraph(self, tmp_path, monkeypatch) -> None:
        pipe = self._pipeline(tmp_path, monkeypatch, {"splitter": "bogus"})
        src = tmp_path / "source"
        src.mkdir()
        (src / "notes.txt").write_text("第一段。\n\n第二段。", encoding="utf-8")
        summary = pipe.ingest(src, base=src)
        assert summary.total_chunks == 2
        texts = [r.text for r in pipe.store.get_by_ids(["test_splitter::notes-txt::0", "test_splitter::notes-txt::1"])]
        assert texts == ["第一段。", "第二段。"]
