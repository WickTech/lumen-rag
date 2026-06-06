"""Query-time retrieval: embed the query, search the store, optionally rerank.

Three retrieval modes:
  ``vector``  — dense cosine search only (original behaviour).
  ``bm25``    — sparse BM25 keyword search only.
  ``hybrid``  — Reciprocal Rank Fusion of both lists (default).

The legacy ``rerank`` parameter is preserved for backwards compatibility: when
mode is ``vector`` and rerank is True, the light lexical-overlap booster from
v0.1 is applied on top of the vector hits.
"""
from __future__ import annotations

import re
from typing import Literal

from ..embeddings import Embedder, get_embedder
from ..store import ScoredChunk, VectorStore
from .bm25 import BM25Index, reciprocal_rank_fusion

_TOKEN_RE = re.compile(r"[a-z0-9]+")

RetrievalMode = Literal["vector", "bm25", "hybrid"]


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
        self._bm25: BM25Index | None = None

    def _get_bm25(self) -> BM25Index:
        """Build (or return cached) BM25 index from the current store chunks."""
        if self._bm25 is None or len(self._bm25.chunks) != len(self.store._chunks):
            self._bm25 = BM25Index.build(self.store._chunks)
        return self._bm25

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode = "hybrid",
        rerank: bool = True,
        rerank_weight: float = 0.25,
    ) -> list[ScoredChunk]:
        """Retrieve top-k chunks for *query*.

        Args:
            query: Natural-language question.
            k: Number of results to return.
            mode: ``"vector"`` | ``"bm25"`` | ``"hybrid"`` (default).
            rerank: When mode is ``"vector"``, blend lexical overlap into scores.
            rerank_weight: Weight of the lexical component in vector+lexical blend.
        """
        if len(self.store) == 0:
            return []

        if mode == "bm25":
            return self._get_bm25().search(query, k)

        if mode == "hybrid":
            fetch = k * 3
            query_vec = self.embedder.embed([query])[0]
            dense_list = self.store.search(query_vec, k=fetch)
            sparse_list = self._get_bm25().search(query, fetch)
            return reciprocal_rank_fusion([dense_list, sparse_list], k=k)

        # --- vector (default legacy path) ---
        query_vec = self.embedder.embed([query])[0]
        pool = self.store.search(query_vec, k=k * 3 if rerank else k)
        if not rerank:
            return pool[:k]

        for sc in pool:
            lex = _lexical_overlap(query, sc.chunk.text)
            sc.score = (1 - rerank_weight) * sc.score + rerank_weight * lex
        pool.sort(key=lambda s: s.score, reverse=True)
        return pool[:k]
