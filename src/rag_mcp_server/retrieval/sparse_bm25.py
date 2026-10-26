"""Sparse retrieval: true BM25 over an in-memory corpus (ADR 0003).

The corpus is injected at construction (list[Chunk]); the orchestrator builds it
from ``store.get_all()`` per query, so the retriever never touches the vector
store and holds no session state. Chinese text is segmented with jieba; English
words are split on ASCII word boundaries. Scoring is the classic BM25 formula
with fixed k1=1.5, b=0.75. A score of 0 means the chunk is not returned.

See docs/spec-p9-hybrid.md (task 7) and CONTEXT.md "会话级 BM25".
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter

import jieba

from rag_mcp_server.core.types import Chunk, RetrievalResult, image_refs_metadata
from rag_mcp_server.core.validation import validate_positive_int

BM25_K1 = 1.5
BM25_B = 0.75


def _is_punctuation_only(token: str) -> bool:
    """True if every char is a Unicode punctuation (Po/Ps/Pe/Pd/Pc/Pi/Pf).

    CJK ideographs are category Lo (Letter, other) and are never filtered;
    full-width/half-width punctuation (，。、？！；：（） etc.) are P categories.
    """
    return bool(token) and all(unicodedata.category(ch).startswith("P") for ch in token)


def tokenize(text: str) -> list[str]:
    """Split text into BM25 terms.

    Chinese is segmented by jieba into words; ASCII tokens are split on word
    boundaries and lowercased. Punctuation (ASCII and CJK/full-width) and
    whitespace are dropped. A bare term (no separator) is kept whole so "MCP"
    stays one token, not M/C/P.
    """
    parts = jieba.lcut(text)
    tokens: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped or stripped.isspace() or _is_punctuation_only(stripped):
            continue
        if stripped.isascii():
            tokens.append(stripped.lower())
        else:
            tokens.append(stripped)
    return tokens


class SparseBM25:
    """BM25 keyword retriever over a fixed in-memory corpus.

    k1/b are fixed module constants (BM25_K1 / BM25_B); a corpus of chunks
    that tokenize to zero terms is treated as empty (avgdl 0 -> no scoring).
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

        n_docs = len(chunks)
        doc_freq: Counter[str] = Counter()
        term_counts: list[Counter] = []
        doc_lens: list[int] = []
        for chunk in chunks:
            terms = tokenize(chunk.text)
            term_counts.append(Counter(terms))
            doc_lens.append(len(terms))
            doc_freq.update(set(terms))

        self._doc_freq = doc_freq
        self._term_counts = term_counts
        self._doc_lens = doc_lens
        self._avgdl = sum(doc_lens) / n_docs if n_docs else 0.0

    def search(self, query_text: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return top_k chunks ranked by BM25 score, most relevant first."""
        validate_positive_int("top_k", top_k)
        query_terms = tokenize(query_text)
        if not query_terms:
            return []

        scored: list[tuple[float, Chunk]] = []
        for i, chunk in enumerate(self._chunks):
            score = self._score(query_terms, i)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                score=score,
                text=chunk.text,
                sources=["sparse"],
                metadata=image_refs_metadata(chunk.metadata),
            )
            for score, chunk in scored[:top_k]
        ]

    def _score(self, query_terms: list[str], index: int) -> float:
        n_docs = len(self._chunks)
        if n_docs == 0 or self._avgdl == 0:
            # empty corpus, or a corpus whose chunks all tokenize to zero terms
            # (no term frequencies to score against) -> nothing can match
            return 0.0
        term_counts = self._term_counts[index]
        doc_len = self._doc_lens[index]
        total = 0.0
        for term in query_terms:
            freq = term_counts.get(term, 0)
            if freq == 0:
                continue
            df = self._doc_freq[term]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / self._avgdl))
            total += idf * (freq * (BM25_K1 + 1.0)) / denom
        return total
