"""Tests for ChunkRefiner — rule-based denoising + optional LLM rewrite.

Red/green contract (docs/spec-p10-transform.md, ticket 04, ADR 0011):
- BaseTransform.transform(chunks, trace=None) -> list[Chunk]
- ChunkRefiner rules: code-block preserve, footer/separator
  removal, HTML comment removal, HTML tag strip, whitespace collapse, line
  trailing trim. Clean text is NOT over-cleaned.
- optional LLM rewrite: prompt -> LLM; non-empty result used; failure/empty ->
  rule result kept; metadata refined_by="llm"|"rule" (+ fallback reason)
- per-chunk exceptions don't affect other chunks

The only seam is the transform module with FakeLLM injected for the LLM path.
All offline.
"""

from __future__ import annotations

import json

from rag_mcp_server.core.types import Chunk
from rag_mcp_server.ingestion.transform.chunk_refiner import ChunkRefiner
from rag_mcp_server.libs.llm.fake import FakeLLM


def _chunk(text: str) -> Chunk:
    return Chunk(
        document_id="d",
        text=text,
        source_path="docs/d.md",
        order=0,
        metadata={"file_type": "md"},
    )


# -- rule-based denoising ---------------------------------------------------
def test_removes_footer_separator_with_page_number():
    # realistic two-sided PDF footer: ───── Page 12 ─────
    text = "正文内容。\n" + "─" * 20 + "Page 12" + "─" * 20 + "\n继续正文。"
    out = ChunkRefiner().transform([_chunk(text)])
    result = out[0].text
    assert "Page 12" not in result
    assert "正文内容。" in result
    assert "继续正文。" in result


def test_removes_html_comment():
    text = "正文。<!-- 这是注释，应删除 -->继续。"
    out = ChunkRefiner().transform([_chunk(text)])
    assert "注释" not in out[0].text
    assert "正文。" in out[0].text


def test_strips_html_tags_but_keeps_content():
    text = "<p>第一段</p><div>第二段</div>"
    out = ChunkRefiner().transform([_chunk(text)])
    assert "<p>" not in out[0].text
    assert "<div>" not in out[0].text
    assert "第一段" in out[0].text
    assert "第二段" in out[0].text


def test_collapses_extra_whitespace():
    text = "第一行。\n\n\n\n第二行。  第三行。"
    out = ChunkRefiner().transform([_chunk(text)])
    assert "\n\n\n" not in out[0].text
    assert "   " not in out[0].text
    assert "第一行。" in out[0].text


def test_preserves_code_block_internal_format():
    text = "说明文字。\n```python\ndef f(a,  b):\n    return a+b\n```\n结尾。"
    out = ChunkRefiner().transform([_chunk(text)])
    result = out[0].text
    # code block inner content is preserved verbatim (whitespace not collapsed)
    assert "def f(a,  b):" in result
    assert "    return a+b" in result
    assert "```python" in result


def test_clean_text_not_over_cleaned():
    text = "这是一段干净的中文文本，没有任何噪声。\n保持原样。"
    out = ChunkRefiner().transform([_chunk(text)])
    assert out[0].text == text


def test_mixed_noise_all_removed():
    text = ("<h1>标题</h1>\n<!-- 页眉注释 -->\n正文第一段。\n"
            + "─" * 15 + "Confidential" + "─" * 15 + "\n正文第二段。")
    out = ChunkRefiner().transform([_chunk(text)])
    result = out[0].text
    assert "<h1>" not in result
    assert "页眉注释" not in result
    assert "Confidential" not in result
    assert "正文第一段。" in result
    assert "正文第二段。" in result


def test_ocr_noise_whitespace_collapsed():
    text = "第一 段。\n\n\n第二  段。"
    out = ChunkRefiner().transform([_chunk(text)])
    assert "\n\n\n" not in out[0].text


# -- metadata flag ----------------------------------------------------------
def test_rule_refine_marks_refined_by_rule():
    out = ChunkRefiner().transform([_chunk("  带空白的文本  ")])
    assert out[0].metadata["refined_by"] == "rule"
    assert out[0].metadata["file_type"] == "md"  # original metadata preserved


# -- LLM rewrite path -------------------------------------------------------
def test_llm_rewrite_used_when_non_empty():
    llm = _chatting_llm("这是 LLM 重写后的干净文本。")
    out = ChunkRefiner(use_llm=True, llm=llm).transform([_chunk("原文（带噪声）。")])
    assert out[0].text == "这是 LLM 重写后的干净文本。"
    assert out[0].metadata["refined_by"] == "llm"


def test_llm_rewrite_failure_falls_back_to_rule():
    class BoomLLM(FakeLLM):
        def _chat(self, messages: list[dict]) -> str:
            raise RuntimeError("llm down")

    out = ChunkRefiner(use_llm=True, llm=BoomLLM()).transform([_chunk("  原文  ")])
    # rule result kept (whitespace trimmed), flag = rule + fallback reason
    assert "原文" in out[0].text
    assert out[0].metadata["refined_by"] == "rule"
    assert out[0].metadata["refine_fallback_reason"] == "llm_failed"


def test_llm_rewrite_empty_result_falls_back_to_rule():
    llm = _chatting_llm("")
    out = ChunkRefiner(use_llm=True, llm=llm).transform([_chunk("  原文  ")])
    assert out[0].metadata["refined_by"] == "rule"
    assert out[0].metadata["refine_fallback_reason"] == "llm_failed"


def test_use_llm_without_llm_stays_rule():
    out = ChunkRefiner(use_llm=True).transform([_chunk("原文")])
    assert out[0].metadata["refined_by"] == "rule"


# -- per-chunk isolation ----------------------------------------------------
def test_one_chunk_failure_does_not_affect_others():
    """A hard failure in the rule path (real _process_one try/except, review P8)
    keeps the failing chunk's original text + refined_by="error" and never
    blocks the batch."""
    class FlakyRefiner(ChunkRefiner):
        def _rule_based_refine(self, text):
            if "boom" in text:
                raise RuntimeError("rule crash")
            return text

    chunks = [_chunk("a"), _chunk("boom"), _chunk("c")]
    out = FlakyRefiner().transform(chunks)
    assert len(out) == 3
    assert out[0].metadata["refined_by"] == "rule"
    assert out[1].metadata["refined_by"] == "error"  # isolated, not fatal
    assert out[1].text == "boom"  # original chunk kept verbatim
    assert out[1].metadata["refine_error"]  # audit F4: reason recorded (mirrors enrich_error)
    assert out[2].metadata["refined_by"] == "rule"


# -- prompt override (review P1) -------------------------------------------
def test_prompt_path_file_content_is_used(tmp_path):
    """A prompt_path file overrides the built-in template when no explicit
    prompt is passed (config/prompts/chunk_refinement.txt seam)."""
    f = tmp_path / "refine.txt"
    f.write_text("自定义清洗提示：{text}", encoding="utf-8")

    refiner = ChunkRefiner(use_llm=True, llm=_chatting_llm("LLM 结果"), prompt_path=str(f))
    assert "{text}" in refiner._prompt
    # the LLM call receives the file-based prompt, not the default one
    out = refiner.transform([_chunk("原文")])
    assert out[0].text == "LLM 结果"
    assert out[0].metadata["refined_by"] == "llm"


def test_prompt_explicit_beats_prompt_path(tmp_path):
    """Direct prompt is the highest priority (unit tests / runtime override):
    it must win over a prompt_path file."""
    f = tmp_path / "refine.txt"
    f.write_text("文件提示：{text}", encoding="utf-8")

    refiner = ChunkRefiner(use_llm=True, llm=_chatting_llm("LLM 结果"),
                           prompt="直传提示：{text}", prompt_path=str(f))
    assert refiner._prompt == "直传提示：{text}"


def test_missing_prompt_path_falls_back_to_default(tmp_path):
    """A missing prompt_path file falls back to the built-in template (config
    error must not crash the transform)."""
    from rag_mcp_server.ingestion.transform.chunk_refiner import DEFAULT_REFINE_PROMPT

    refiner = ChunkRefiner(prompt_path=str(tmp_path / "nope.txt"))
    assert refiner._prompt == DEFAULT_REFINE_PROMPT


# -- helper -----------------------------------------------------------------
def _chatting_llm(payload: str) -> FakeLLM:
    llm = FakeLLM()
    llm._chat = lambda messages: payload  # type: ignore[method-assign]
    return llm
