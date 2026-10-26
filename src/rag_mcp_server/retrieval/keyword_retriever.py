"""Simple word-frequency keyword retrieval baseline.

The query is split into keywords (ASCII words and CJK runs). Each
chunk is scored by counting how many times each keyword appears as a
substring of the chunk text. A score of 0 means the chunk is not
returned. No BM25 or vector retrieval in the first version.
"""

from __future__ import annotations

import re

from rag_mcp_server.core.types import Chunk, RetrievalResult

DEFAULT_TOP_K = 3

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Split text into case-normalized keyword tokens.

    English words are tokenized by [_WORD_RE]; a multi-character CJK
    run is kept whole so that "检索" stays one keyword instead of
    being split into its individual characters.
    """
    tokens: list[str] = []
    for part in re.split(r"[^a-zA-Z0-9_一-鿿]+", text):
        if not part:
            continue
        if part.isascii():
            tokens.extend(m.lower() for m in _WORD_RE.findall(part))
        else:
            tokens.append(part)
    return tokens


def _score_chunk(chunk: Chunk, keywords: list[str]) -> float:
    lower_text = chunk.text.lower()
    return float(sum(lower_text.count(keyword) for keyword in keywords))


def keyword_search(
    query: str,
    chunks: list[Chunk],
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    keywords = _tokenize(query)
    scored: list[tuple[float, Chunk]] = []
    if keywords:
        for chunk in chunks:
            score = _score_chunk(chunk, keywords)
            if score > 0:
                scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = [
        RetrievalResult(
            chunk_id=chunk.chunk_id,
            source_path=chunk.source_path,
            score=score,
            text=chunk.text,
        )
        for score, chunk in scored[:top_k]
    ]
    return results
