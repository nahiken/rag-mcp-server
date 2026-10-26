"""MetadataEnricher — inject semantic metadata (title / summary / tags).

Rule fallback (always): title from markdown heading / first line / first
sentence / prefix; summary from the first sentences (truncated); tags from jieba
high-frequency terms (single chars / stopwords / pure punctuation filtered). The
optional LLM path asks an injected BaseLLM for strict JSON ``{"title","summary","tags"}``;
valid non-empty output is used, anything else (invalid JSON / wrong shape / LLM
failure) falls back to the rule result. metadata flag: ``enriched_by`` = "llm" |
"rule"; on LLM failure also ``enrich_fallback_reason``; on a hard per-chunk
failure ``enriched_by="error"`` + ``enrich_error`` (review P2). The prompt can
be overridden via the ``config/prompts/metadata_enrichment.txt`` file (review
P1: ``prompt_path`` seam; explicit ``prompt`` stays the highest priority) and
its ``{chunk_text}`` placeholder is guarded (review P7). See
docs/spec-p10-transform.md and ADR 0011.
"""

from __future__ import annotations

import json
import re

import jieba

from rag_mcp_server.core.types import Chunk
from rag_mcp_server.ingestion.transform.base import BaseTransform, _with_metadata, load_prompt_text
from rag_mcp_server.libs.llm.base import BaseLLM

DEFAULT_ENRICH_PROMPT = """你是一个元数据提取助手。分析下面的文本块，输出严格的 JSON 对象：
{"title": "<简明标题，最多 150 字>", "summary": "<2-3 句摘要，最多 500 字>", "tags": ["<标签>", ...]}

要求：标题清晰非泛泛；摘要抓住要义而非复述标题；标签 3-10 个，是文本中提到的具体主题/概念/实体。
只输出 JSON，不要任何其他文字。

文本块：
{chunk_text}"""

# File override (review P1): config/prompts/metadata_enrichment.txt — same
# content as DEFAULT_ENRICH_PROMPT, applied only when prompt_path is passed.
DEFAULT_ENRICH_PROMPT_PATH = "config/prompts/metadata_enrichment.txt"

# punctuation / stop-ish tokens that never become a tag
_NOISE = re.compile(r"^[^\w一-鿿]+$")
# multi-character function words that jieba keeps whole but are not useful tags
# (review P6 — spec「过滤停用词」字面; single chars are already dropped below)
_STOPWORDS = {
    "一个",
    "这个",
    "那个",
    "这些",
    "那些",
    "我们",
    "他们",
    "你们",
    "自己",
    "这样",
    "那样",
    "什么",
    "怎么",
    "如何",
    "因为",
    "所以",
    "但是",
    "不过",
    "以及",
    "可以",
    "进行",
    "没有",
    "已经",
    "还是",
    "就是",
    "现在",
    "目前",
    "其中",
    "然后",
    "还有",
    "需要",
    "对于",
    "通过",
    "并且",
    "不是",
}
MAX_TAGS = 10
MAX_TITLE = 150
MAX_SUMMARY = 500
MAX_SUMMARY_SENTENCES = 3
# LLM input truncation (review P7): keep the prompt bounded; documented in spec.
MAX_LLM_CHUNK_CHARS = 2000


class MetadataEnricher(BaseTransform):
    """Enrich chunk metadata: rules always, optional LLM enhance on top."""

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
            prompt, prompt_path, DEFAULT_ENRICH_PROMPT
        )

    def _process_one(self, chunk: Chunk) -> Chunk:
        try:
            base = self._rule_based_enrich(chunk.text)
            enriched_by = "rule"
            fallback_reason: str | None = None
            if self.use_llm and self._llm is not None:
                llm_result = self._llm_enrich(chunk.text)
                if llm_result is not None:
                    base = llm_result
                    enriched_by = "llm"
                else:
                    fallback_reason = "llm_failed"
            return _with_metadata(
                chunk,
                {
                    **base,
                    "enriched_by": enriched_by,
                    **( {"enrich_fallback_reason": fallback_reason} if fallback_reason else {}),
                },
            )
        except Exception as exc:
            # hard per-chunk failure: keep original chunk, mark error + reason
            # (review P2 — spec L40「硬异常标 enriched_by=error + enrich_error」)
            return _with_metadata(
                chunk,
                {"enriched_by": "error", "enrich_error": str(exc)},
            )

    # -- rule fallback ----------------------------------------------------
    def _rule_based_enrich(self, text: str) -> dict:
        return {
            "title": self._extract_title(text),
            "summary": self._extract_summary(text),
            "tags": self._extract_tags(text),
        }

    def _extract_title(self, text: str) -> str:
        # 1) markdown heading
        m = re.match(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
        if m:
            return m.group(1).strip()[:MAX_TITLE]
        # 2) first line, short and not sentence-ending
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if first_line and len(first_line) <= 100 and not first_line.endswith((".", "。", ",", "；")):
            return first_line.strip()[:MAX_TITLE]
        # 3) first sentence
        sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
        if sentences and sentences[0]:
            title = re.sub(r"[.!?。！？]+$", "", sentences[0]).strip()
            if title:
                return title[:MAX_TITLE]
        # 4) first 100 chars
        prefix = text.strip()[:100]
        return prefix + "..." if len(prefix) == 100 and len(text.strip()) > 100 else prefix or "Untitled"

    def _extract_summary(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
        summary = " ".join(sentences[:MAX_SUMMARY_SENTENCES]).strip()
        if len(summary) > MAX_SUMMARY:
            return summary[: MAX_SUMMARY - 3] + "..."
        return summary

    def _extract_tags(self, text: str) -> list[str]:
        words = [w.strip() for w in jieba.lcut(text) if w.strip()]
        # filter single chars / pure punctuation / multi-char stopwords (P6)
        candidates = [
            w for w in words
            if len(w) > 1 and w not in _STOPWORDS and not _NOISE.match(w)
        ]
        # frequency sort (stable): most frequent first
        freq: dict[str, int] = {}
        for w in candidates:
            freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
        # dedupe keeping order
        seen: set[str] = set()
        tags: list[str] = []
        for w, _ in ranked:
            if w in seen:
                continue
            seen.add(w)
            tags.append(w)
            if len(tags) >= MAX_TAGS:
                break
        return tags

    # -- LLM enhance -------------------------------------------------------
    def _llm_enrich(self, text: str) -> dict | None:
        if "{chunk_text}" not in self._prompt:
            return None  # placeholder missing -> can't build the prompt (review P7)
        prompt = self._prompt.replace("{chunk_text}", text[:MAX_LLM_CHUNK_CHARS])
        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
            if not isinstance(raw, str):
                return None  # audit F1: non-str (None/dict) is a soft failure, not a crash
            raw = raw.strip()
        except Exception:
            return None
        # audit F2: tolerate fenced code blocks (aligned with LLMReranker._parse,
        # review P7 — real LLMs commonly wrap JSON in ```json fences)
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        # strict shape: dict with title/summary str, tags list of str
        if not isinstance(parsed, dict):
            return None
        if not isinstance(parsed.get("title"), str) or not isinstance(parsed.get("summary"), str):
            return None
        tags = parsed.get("tags")
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            return None
        if not parsed["title"].strip():
            return None
        # audit F3: "valid non-empty output" — blank summary/tags are not useful
        # enhanced metadata; treat them as a parse failure (soft fallback).
        if not parsed["summary"].strip():
            return None
        tags = [t.strip() for t in tags if t.strip()]
        if not tags:
            return None
        return {"title": parsed["title"], "summary": parsed["summary"], "tags": tags}
