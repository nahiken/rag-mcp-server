"""Tests for MetadataEnricher — rule-based title/summary/tags + optional LLM enhance.

Red/green contract (docs/spec-p10-transform.md, ticket 05, ADR 0011):
- rule fallback: title (markdown heading / first line / first sentence / prefix),
  summary (first N sentences, truncated), tags (jieba top-N, filtered)
- LLM enhance: strict JSON {"title","summary","tags"}; non-empty used; failure /
  invalid JSON -> rule fallback + enriched_by="rule" + enrich_fallback_reason
- metadata flags: enriched_by="llm"|"rule"; error chunk marked enriched_by="error"

The only seam is the transform module with FakeLLM injected. All offline.
"""

from __future__ import annotations

import json

from rag_mcp_server.core.types import Chunk
from rag_mcp_server.ingestion.transform.metadata_enricher import MetadataEnricher
from rag_mcp_server.libs.llm.fake import FakeLLM


def _chunk(text: str) -> Chunk:
    return Chunk(
        document_id="d",
        text=text,
        source_path="docs/d.md",
        order=0,
        metadata={"file_type": "md"},
    )


# -- rule fallback ----------------------------------------------------------
def test_rule_enrich_injects_non_empty_title_summary_tags():
    text = "# RAG 检索原理\n\nRAG 通过检索增强生成。它结合向量检索与生成模型。混合检索用稠密与稀疏。"
    out = MetadataEnricher().transform([_chunk(text)])
    md = out[0].metadata
    assert md["enriched_by"] == "rule"
    assert md["title"] == "RAG 检索原理"  # markdown heading wins
    assert md["summary"]  # non-empty
    assert md["tags"]  # non-empty list


def test_rule_title_falls_back_to_first_line():
    text = "第一行标题\n第二行内容。"
    out = MetadataEnricher().transform([_chunk(text)])
    assert out[0].metadata["title"] == "第一行标题"


def test_rule_tags_are_jieba_frequency_based():
    text = "机器学习 机器学习 深度学习 神经网络 检索 检索 检索 向量"
    out = MetadataEnricher().transform([_chunk(text)])
    tags = out[0].metadata["tags"]
    assert isinstance(tags, list)
    assert tags  # non-empty
    # high-frequency terms appear; 检索 appears 3x so it should rank high
    assert "检索" in tags


def test_rule_tags_filter_single_chars_and_punct():
    text = "的 了 是 检索 向量 啊 ！ ？ 机器学习"
    out = MetadataEnricher().transform([_chunk(text)])
    tags = out[0].metadata["tags"]
    assert all(len(t) > 1 for t in tags)  # no single chars
    assert "检索" in tags
    assert "向量" in tags


def test_rule_tags_filter_multi_char_stopwords():
    """Review P6: multi-character function words (我们/因为/这个…) must not
    become tags, even when frequent (spec「过滤停用词」字面)."""
    text = "我们 我们 因为 这个 检索 检索 因为 因为 数据库"
    out = MetadataEnricher().transform([_chunk(text)])
    tags = out[0].metadata["tags"]
    assert "我们" not in tags
    assert "因为" not in tags
    assert "这个" not in tags
    assert "检索" in tags  # real content terms survive
    assert "数据库" in tags


def test_rule_enrich_preserves_original_metadata():
    out = MetadataEnricher().transform([_chunk("正文内容。")])
    assert out[0].metadata["file_type"] == "md"


# -- LLM enhance ------------------------------------------------------------
def test_llm_enhance_uses_strict_json_when_valid():
    payload = json.dumps({"title": "LLM 标题", "summary": "LLM 摘要。", "tags": ["tag1", "tag2"]})
    llm = _chatting_llm(payload)
    out = MetadataEnricher(use_llm=True, llm=llm).transform([_chunk("原文。")])
    md = out[0].metadata
    assert md["enriched_by"] == "llm"
    assert md["title"] == "LLM 标题"
    assert md["summary"] == "LLM 摘要。"
    assert md["tags"] == ["tag1", "tag2"]


def test_llm_enhance_invalid_json_falls_back_to_rule():
    llm = _chatting_llm("不是 JSON")
    out = MetadataEnricher(use_llm=True, llm=llm).transform([_chunk("# 标题\n正文。")])
    md = out[0].metadata
    assert md["enriched_by"] == "rule"
    assert md["enrich_fallback_reason"] == "llm_failed"
    assert md["title"]  # rule fallback title present


def test_llm_enhance_wrong_shape_falls_back_to_rule():
    llm = _chatting_llm(json.dumps(["a", "b"]))  # list, not dict
    out = MetadataEnricher(use_llm=True, llm=llm).transform([_chunk("# 标题\n正文。")])
    assert out[0].metadata["enriched_by"] == "rule"
    assert out[0].metadata["enrich_fallback_reason"] == "llm_failed"


def test_llm_enhance_missing_tags_field_falls_back_to_rule():
    llm = _chatting_llm(json.dumps({"title": "T", "summary": "S"}))  # no tags
    out = MetadataEnricher(use_llm=True, llm=llm).transform([_chunk("正文。")])
    assert out[0].metadata["enriched_by"] == "rule"
    assert out[0].metadata["enrich_fallback_reason"] == "llm_failed"


def test_llm_enhance_llm_exception_falls_back_to_rule():
    class BoomLLM(FakeLLM):
        def _chat(self, messages: list[dict]) -> str:
            raise RuntimeError("llm down")

    out = MetadataEnricher(use_llm=True, llm=BoomLLM()).transform([_chunk("# 标题\n正文。")])
    md = out[0].metadata
    assert md["enriched_by"] == "rule"
    assert md["enrich_fallback_reason"] == "llm_failed"
    assert md["title"]  # rule fallback


def test_llm_enhance_non_str_return_falls_back_to_rule():
    """Audit F1: an LLM returning None / a dict (outside the BaseLLM str
    contract) is a *soft* failure -> rule fallback, never a hard error."""
    class NoneLLM(FakeLLM):
        def _chat(self, messages):
            return None

    class DictLLM(FakeLLM):
        def _chat(self, messages):
            return {"x": 1}

    for llm in (NoneLLM(), DictLLM()):
        md = MetadataEnricher(use_llm=True, llm=llm).transform([_chunk("# 标题\n正文。")])[0].metadata
        assert md["enriched_by"] == "rule", md
        assert md["enrich_fallback_reason"] == "llm_failed"
        assert "enrich_error" not in md


def test_llm_enhance_fenced_json_is_parsed():
    """Audit F2: real LLMs wrap JSON in ```json fences; align with
    LLMReranker._parse (review P7) and parse it instead of falling back."""
    import json as _json

    payload = _json.dumps({"title": "T", "summary": "S", "tags": ["a"]})
    raw = "```json\n" + payload + "\n```"
    llm = _chatting_llm(raw)
    md = MetadataEnricher(use_llm=True, llm=llm).transform([_chunk("原文。")])[0].metadata
    assert md["enriched_by"] == "llm"
    assert md["title"] == "T"


def test_llm_enhance_blank_summary_or_tags_falls_back_to_rule():
    """Audit F3: 'valid non-empty output' means no blank summary and no blank
    tag items. A blank summary -> soft fallback. Blank tag items are stripped
    out; if nothing real remains -> soft fallback; otherwise the clean list is
    kept."""
    import json as _json

    # blank summary -> fallback
    llm = _chatting_llm(_json.dumps({"title": "T", "summary": "", "tags": ["a"]}))
    md = MetadataEnricher(use_llm=True, llm=llm).transform([_chunk("# 标题\n正文。")])[0].metadata
    assert md["enriched_by"] == "rule"
    assert md["enrich_fallback_reason"] == "llm_failed"

    # all-blank tags -> fallback
    llm = _chatting_llm(_json.dumps({"title": "T", "summary": "S", "tags": [" ", ""]}))
    md = MetadataEnricher(use_llm=True, llm=llm).transform([_chunk("# 标题\n正文。")])[0].metadata
    assert md["enriched_by"] == "rule"
    assert md["enrich_fallback_reason"] == "llm_failed"

    # some blank tag items -> kept, but the list is cleaned of blanks
    llm = _chatting_llm(_json.dumps({"title": "T", "summary": "S", "tags": ["a", "", "  ", "b"]}))
    md = MetadataEnricher(use_llm=True, llm=llm).transform([_chunk("# 标题\n正文。")])[0].metadata
    assert md["enriched_by"] == "llm"
    assert md["tags"] == ["a", "b"]  # blanks stripped, real tags survive


def test_use_llm_without_llm_stays_rule():
    out = MetadataEnricher(use_llm=True).transform([_chunk("正文。")])
    assert out[0].metadata["enriched_by"] == "rule"


# -- hard per-chunk failure (review P2/P8) ----------------------------------
def test_hard_rule_failure_marks_enriched_by_error_with_reason():
    """Review P8: a real exception in the rule path (not a subclass cheat) must
    trigger _process_one's try/except and keep the original chunk."""
    class FlakyEnricher(MetadataEnricher):
        def _rule_based_enrich(self, text):
            if "boom" in text:
                raise RuntimeError("rule crash")
            return {"title": "t", "summary": "s", "tags": ["x"]}

    out = FlakyEnricher().transform([_chunk("boom"), _chunk("ok")])
    md0 = out[0].metadata
    assert md0["enriched_by"] == "error"  # isolated, not fatal
    assert md0["enrich_error"]  # review P2: the reason is recorded (spec L40)
    assert out[0].text == "boom"  # original chunk kept
    assert out[1].metadata["enriched_by"] == "rule"


# -- prompt override (review P1) -------------------------------------------
def test_prompt_path_file_content_is_used(tmp_path):
    """A prompt_path file overrides the built-in template when no explicit
    prompt is passed (config/prompts/metadata_enrichment.txt seam)."""
    import json

    f = tmp_path / "enrich.txt"
    f.write_text("自定义提示：{chunk_text}", encoding="utf-8")
    payload = json.dumps({"title": "T", "summary": "S", "tags": ["a"]})

    enricher = MetadataEnricher(use_llm=True, llm=_chatting_llm(payload), prompt_path=str(f))
    assert "{chunk_text}" in enricher._prompt
    out = enricher.transform([_chunk("原文。")])
    assert out[0].metadata["enriched_by"] == "llm"
    assert out[0].metadata["title"] == "T"


def test_prompt_explicit_beats_prompt_path(tmp_path):
    f = tmp_path / "enrich.txt"
    f.write_text("文件提示：{chunk_text}", encoding="utf-8")

    enricher = MetadataEnricher(prompt="直传：{chunk_text}", prompt_path=str(f))
    assert enricher._prompt == "直传：{chunk_text}"


def test_missing_prompt_path_falls_back_to_default(tmp_path):
    from rag_mcp_server.ingestion.transform.metadata_enricher import DEFAULT_ENRICH_PROMPT

    enricher = MetadataEnricher(prompt_path=str(tmp_path / "nope.txt"))
    assert enricher._prompt == DEFAULT_ENRICH_PROMPT


def test_prompt_without_placeholder_falls_back_to_rule():
    """Review P7: a prompt lacking {chunk_text} cannot be built -> soft fallback
    to the rule result (aligned with ChunkRefiner's {text} guard)."""
    payload = json.dumps({"title": "T", "summary": "S", "tags": ["a"]})
    llm = _chatting_llm(payload)
    out = MetadataEnricher(use_llm=True, llm=llm, prompt="没有占位符的提示").transform([_chunk("# 标题\n正文。")])
    md = out[0].metadata
    assert md["enriched_by"] == "rule"
    assert md["enrich_fallback_reason"] == "llm_failed"
    assert md["title"]  # rule fallback title present


# -- helper -----------------------------------------------------------------
def _chatting_llm(payload: str) -> FakeLLM:
    llm = FakeLLM()
    llm._chat = lambda messages: payload  # type: ignore[method-assign]
    return llm
