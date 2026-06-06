"""BM25 sparse retriever with Reciprocal Rank Fusion blending.

BM25 rewards term frequency while penalising documents that are much longer
than average — it consistently outperforms raw TF-IDF and complements dense
vector search on exact-match, acronym, and entity queries.

Reciprocal Rank Fusion (RRF) is the fusion strategy: each candidate is scored
as Σ 1/(k+rank_i) across its rank in each sub-ranker.  RRF is parameter-robust
and consistently beats linear score blending (Cormack, Clarke, Buettcher 2009).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..store import Chunk, ScoredChunk, VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """Pre-computed BM25 index over a corpus of chunks.

    Build once at ingest/load time; query many times with ``score``.
    """

    chunks: list["Chunk"]
    avgdl: float
    doc_freqs: dict[str, int]   # term → number of docs containing it
    term_freqs: list[dict[str, int]]  # per-chunk term frequencies
    n: int                      # total documents

    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, chunks: list["Chunk"], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        term_freqs: list[dict[str, int]] = []
        doc_freqs: dict[str, int] = Counter()
        total_len = 0

        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            tf = Counter(tokens)
            term_freqs.append(tf)
            for tok in tf:
                doc_freqs[tok] += 1
            total_len += len(tokens)

        avgdl = total_len / max(1, len(chunks))
        idx = cls(
            chunks=list(chunks),
            avgdl=avgdl,
            doc_freqs=dict(doc_freqs),
            term_freqs=term_freqs,
            n=len(chunks),
            k1=k1,
            b=b,
        )
        return idx

    def score(self, query: str) -> list[float]:
        """Return a BM25 score for every chunk in the index."""
        q_tokens = _tokenize(query)
        scores = [0.0] * self.n
        for tok in q_tokens:
            df = self.doc_freqs.get(tok, 0)
            if df == 0:
                continue
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(tok, 0)
                if f == 0:
                    continue
                dl = sum(tf.values())
                numer = f * (self.k1 + 1)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * numer / denom
        return scores

    def search(self, query: str, k: int) -> "list[ScoredChunk]":
        from ..store import ScoredChunk

        scores = self.score(query)
        if not scores:
            return []
        k = min(k, len(scores))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredChunk(self.chunks[i], scores[i]) for i in top]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

RRF_K = 60  # standard constant from the original paper


def reciprocal_rank_fusion(
    ranked_lists: list[list["ScoredChunk"]],
    k: int,
    rrf_k: int = RRF_K,
) -> "list[ScoredChunk]":
    """Fuse multiple ranked lists of ScoredChunk via RRF.

    Each chunk_id accumulates 1/(rrf_k + rank) from every list it appears in.
    Returns the top-k by fused score.
    """
    from ..store import ScoredChunk

    fused: dict[str, float] = {}
    chunk_map: dict[str, "Chunk"] = {}

    for ranked in ranked_lists:
        for rank, sc in enumerate(ranked, start=1):
            cid = sc.chunk.id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunk_map[cid] = sc.chunk

    top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    return [ScoredChunk(chunk_map[cid], score) for cid, score in top]
