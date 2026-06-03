"""Query-time retrieval: embed the query, search the store, optionally rerank.

The reranker here is a light lexical-overlap booster blended with the vector
score. It cheaply rewards chunks that literally contain query terms — a common
production trick to fix pure-vector misses — without a second model call.
"""
from __future__ import annotations

import re

from ..embeddings import Embedder, get_embedder
from ..store import ScoredChunk, VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _lexical_overlap(query: str, text: str) -> float:
    q = set(_TOKEN_RE.findall(query.lower()))
    if not q:
        return 0.0
    t = set(_TOKEN_RE.findall(text.lower()))
    return len(q & t) / len(q)


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        rerank: bool = True,
        rerank_weight: float = 0.25,
    ) -> list[ScoredChunk]:
        query_vec = self.embedder.embed([query])[0]
        # over-fetch so the reranker has candidates to reorder
        pool = self.store.search(query_vec, k=k * 3 if rerank else k)
        if not rerank:
            return pool[:k]

        for sc in pool:
            lex = _lexical_overlap(query, sc.chunk.text)
            sc.score = (1 - rerank_weight) * sc.score + rerank_weight * lex
        pool.sort(key=lambda s: s.score, reverse=True)
        return pool[:k]
