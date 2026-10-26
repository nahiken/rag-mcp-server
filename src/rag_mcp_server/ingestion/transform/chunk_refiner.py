"""ChunkRefiner — rule-based denoising + optional LLM rewrite.

Rules: preserve code blocks (extract
to a placeholder, restore later), remove footer/separator lines, HTML comments,
strip HTML tags, collapse extra whitespace, trim line trailing spaces. The
optional LLM path rewrites the text via an injected BaseLLM (non-empty result
used; failure/empty -> the rule-refined text is kept). metadata flag:
``refined_by`` = "llm" | "rule"; on LLM failure also ``refine_fallback_reason``.
Per-chunk degradation: a hard failure keeps the original chunk (``refined_by =
"error"``) and never blocks the batch. The prompt can be overridden via the
``config/prompts/chunk_refinement.txt`` file (review P1: ``prompt_path`` seam;
explicit ``prompt`` stays the highest priority). See docs/spec-p10-transform.md.
"""

from __future__ import annotations

import re

from rag_mcp_server.core.types import Chunk
from rag_mcp_server.ingestion.transform.base import BaseTransform, _with_metadata, load_prompt_text
from rag_mcp_server.libs.llm.base import BaseLLM

DEFAULT_REFINE_PROMPT = """你是一个文档分块清洗助手。给定从文档提取的原始文本块，请：
1. 去除提取过程的噪声、乱码或格式残留
2. 保留所有有意义的内容与上下文
3. 保持文本连贯、自包含
4. 保留专有名词、引用、[IMAGE: id] 占位符原样
不要：新增原文没有的信息、总结或大幅缩短、改变原意、添加 Markdown 格式。

输出：只输出清洗后的纯文本，长度与原文接近。

原文：
{text}

清洗后："""

# File override (review P1): config/prompts/chunk_refinement.txt — same content
# as DEFAULT_REFINE_PROMPT, applied only when prompt_path is passed.
DEFAULT_REFINE_PROMPT_PATH = "config/prompts/chunk_refinement.txt"

_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
_SEPARATOR_LINE = re.compile(
    r"─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}",
    re.IGNORECASE | re.DOTALL,
)
_REMAINING_SEPARATOR = re.compile(r"─{10,}")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r" {2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


class ChunkRefiner(BaseTransform):
    """Refine chunk text: rules always, optional LLM rewrite on top."""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        use_llm: bool = False,
        prompt: str | None = None,
        prompt_path: str | None = None,
    ) -> None:
        self._llm = llm
        self.use_llm = use_llm
        self._prompt = load_prompt_text(
            prompt, prompt_path, DEFAULT_REFINE_PROMPT
        )

    def _process_one(self, chunk: Chunk) -> Chunk:
        try:
            refined = self._rule_based_refine(chunk.text)
            refined_by = "rule"
            fallback_reason: str | None = None
            if self.use_llm and self._llm is not None:
                llm_result = self._llm_refine(refined)
                if llm_result is not None and llm_result.strip():
                    refined = llm_result
                    refined_by = "llm"
                else:
                    fallback_reason = "llm_failed"
            return _with_metadata(
                chunk,
                {"refined_by": refined_by, **( {"refine_fallback_reason": fallback_reason} if fallback_reason else {})},
                text=refined,
            )
        except Exception as exc:
            # hard per-chunk failure: keep original chunk, mark error + reason
            # (audit F4: refine_error mirrors MetadataEnricher's enrich_error)
            return _with_metadata(
                chunk,
                {"refined_by": "error", "refine_error": str(exc)},
            )

    # -- rules ----------------------------------------------------------
    def _rule_based_refine(self, text: str) -> str:
        code_blocks: list[str] = []

        def _extract(match: re.Match) -> str:
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        text = _CODE_BLOCK_PATTERN.sub(_extract, text)
        text = _SEPARATOR_LINE.sub("", text)
        text = _REMAINING_SEPARATOR.sub("", text)
        text = _HTML_COMMENT.sub("", text)
        text = _HTML_TAG.sub("", text)
        text = _MULTI_SPACE.sub(" ", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)
        return text.strip()

    # -- LLM -------------------------------------------------------------
    def _llm_refine(self, text: str) -> str | None:
        if "{text}" not in self._prompt:
            return None
        prompt = self._prompt.replace("{text}", text)
        try:
            result = self._llm.chat([{"role": "user", "content": prompt}])
        except Exception:
            return None
        return result.strip() if result else None
