"""Query-time retrieval: embed the query, search the store, optionally rerank.

Three retrieval modes:
  ``vector``  — dense cosine search only (original behaviour).
  ``bm25``    — sparse BM25 keyword search only.
  ``hybrid``  — Reciprocal Rank Fusion of both lists (default).

The legacy ``rerank`` parameter is preserved for backwards compatibility: when
mode is ``vector`` and rerank is True, the light lexical-overlap booster from
v0.1 is applied on top of the vector hits.

When a cross-encoder reranker is configured (``LUMEN_RERANKER``), any mode
over-fetches ``k * LUMEN_RERANK_POOL`` candidates and the cross-encoder picks
the final top-k. See ``rerank.py``.

``retrieve_with_confidence`` also returns a relevance confidence for the query,
which the engine compares against ``LUMEN_ABSTAIN_THRESHOLD`` to refuse questions
the corpus cannot answer. RRF and BM25 scores are rank- or corpus-relative, so
they are not used: the signal is the cross-encoder score when a reranker is
configured, otherwise the best dense cosine similarity in the store.
"""
from __future__ import annotations

import math
import re
from typing import Literal

import numpy as np

from .. import tracing
from ..config import settings
from ..embeddings import Embedder, embed_query, get_embedder
from ..store import ScoredChunk, Store
from .bm25 import BM25Index, reciprocal_rank_fusion
from .decompose import split_question
from .rerank import Reranker, get_reranker

_TOKEN_RE = re.compile(r"[a-z0-9]+")

RetrievalMode = Literal["vector", "bm25", "hybrid"]

# How deep a sub-question looks for a document the whole question missed.
SUB_QUERY_DEPTH = 3

_FROM_SETTINGS = object()  # sentinel: build the reranker from settings


def _lexical_overlap(query: str, text: str) -> float:
    q = set(_TOKEN_RE.findall(query.lower()))
    if not q:
        return 0.0
    t = set(_TOKEN_RE.findall(text.lower()))
    return len(q & t) / len(q)


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder | None = None,
        reranker: Reranker | None | object = _FROM_SETTINGS,
        decompose: bool | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        # Pass reranker=None to force-disable even when LUMEN_RERANKER is set.
        self.reranker: Reranker | None = (
            get_reranker() if reranker is _FROM_SETTINGS else reranker  # type: ignore[assignment]
        )
        self._bm25: BM25Index | None = None
        # None: follow LUMEN_DECOMPOSE at query time.
        self._decompose = decompose

    @property
    def decompose(self) -> bool:
        return settings.decompose if self._decompose is None else self._decompose

    def _get_bm25(self) -> BM25Index:
        """Build (or return cached) BM25 index from the current store chunks."""
        if self._bm25 is None or len(self._bm25.chunks) != len(self.store):
            self._bm25 = BM25Index.build(self.store.chunks())
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
        with tracing.span(
            "lumen.retrieve",
            **{
                "lumen.mode": mode,
                "lumen.k": k,
                "lumen.reranker": self.reranker is not None,
                "lumen.store": type(self.store).__name__,
            },
        ) as span:
            query_vec = None if mode == "bm25" or len(self.store) == 0 else embed_query(self.embedder, query)
            results = self._retrieve(query, k, mode, rerank, rerank_weight, query_vec)
            tracing.set_attr(span, "lumen.results", len(results))
            return results

    def retrieve_with_confidence(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode = "hybrid",
        rerank: bool = True,
        rerank_weight: float = 0.25,
    ) -> tuple[list[ScoredChunk], float]:
        """``retrieve`` plus a confidence in [0, 1]-ish that the corpus answers *query*.

        With a cross-encoder reranker: the sigmoid of the best reranked score.
        Without one: the highest cosine similarity between the query and any stored
        chunk, whatever the retrieval mode. The query is embedded once for both.
        An empty store has confidence 0.
        """
        with tracing.span(
            "lumen.retrieve",
            **{
                "lumen.mode": mode,
                "lumen.k": k,
                "lumen.reranker": self.reranker is not None,
                "lumen.store": type(self.store).__name__,
            },
        ) as span:
            if len(self.store) == 0:
                return [], 0.0
            needs_vec = mode != "bm25" or self.reranker is None
            query_vec = embed_query(self.embedder, query) if needs_vec else None
            results = self._retrieve(query, k, mode, rerank, rerank_weight, query_vec)
            if self.reranker is not None:
                confidence = _sigmoid(results[0].score) if results else 0.0
            else:
                top = self.store.search(query_vec, k=1)
                confidence = top[0].score if top else 0.0
            tracing.set_attr(span, "lumen.results", len(results))
            tracing.set_attr(span, "lumen.confidence", round(confidence, 4))
            return results, confidence

    def _retrieve(
        self,
        query: str,
        k: int,
        mode: RetrievalMode,
        rerank: bool,
        rerank_weight: float,
        query_vec: np.ndarray | None,
    ) -> list[ScoredChunk]:
        if len(self.store) == 0:
            return []

        if self.reranker is not None:
            pool_size = k * max(1, settings.rerank_pool_factor)
            # With a reranker, sub-question hits join the pool instead of displacing
            # anything: the cross-encoder judges them against the whole question.
            pool = self._decomposed(query, pool_size, mode, rerank, rerank_weight, query_vec, union=True)
            return self.reranker.rerank(query, pool, k)

        return self._decomposed(query, k, mode, rerank, rerank_weight, query_vec)

    def _decomposed(
        self,
        query: str,
        k: int,
        mode: RetrievalMode,
        rerank: bool,
        rerank_weight: float,
        query_vec: np.ndarray | None,
        union: bool = False,
    ) -> list[ScoredChunk]:
        """First-stage retrieval, plus sub-question results when decomposition is on.

        The whole question's ranking is kept. Each sub-question then offers its best
        result from a document not already listed, which replaces the lowest-ranked
        original result only if it scores higher. Two simpler fusions were measured
        first and did worse: equal-weight RRF of all the lists let a sub-question that
        lost its context ("how is it handled afterwards") reshuffle everything, and
        always reserving a slot evicted correct rank-5 results for weak sub-query hits.
        """
        base = self._first_stage(query, k, mode, rerank, rerank_weight, query_vec)
        subs = split_question(query) if self.decompose else [query]
        if len(subs) == 1:
            return base
        with tracing.span("lumen.decompose", **{"lumen.subqueries": len(subs)}):
            docs = {sc.chunk.doc_id for sc in base}
            extras: list[ScoredChunk] = []
            for sub in subs:
                for sc in self._first_stage(sub, SUB_QUERY_DEPTH, mode, rerank, rerank_weight, None):
                    if sc.chunk.doc_id not in docs:
                        docs.add(sc.chunk.doc_id)
                        extras.append(sc)
                        break
            if union:
                return base + extras
            results = list(base)
            victim = len(results) - 1
            for extra in extras:
                if victim < 1:  # the whole question's best hit always stays
                    break
                if extra.score > results[victim].score:
                    results[victim] = extra
                    victim -= 1
            return results

    def _first_stage(
        self,
        query: str,
        k: int,
        mode: RetrievalMode,
        rerank: bool,
        rerank_weight: float,
        query_vec: np.ndarray | None,
    ) -> list[ScoredChunk]:
        if mode == "bm25":
            return self._get_bm25().search(query, k)

        if query_vec is None:
            query_vec = embed_query(self.embedder, query)

        if mode == "hybrid":
            fetch = k * 3
            dense_list = self.store.search(query_vec, k=fetch)
            sparse_list = self._get_bm25().search(query, fetch)
            return reciprocal_rank_fusion([dense_list, sparse_list], k=k)

        # --- vector (default legacy path) ---
        pool = self.store.search(query_vec, k=k * 3 if rerank else k)
        if not rerank:
            return pool[:k]

        for sc in pool:
            lex = _lexical_overlap(query, sc.chunk.text)
            sc.score = (1 - rerank_weight) * sc.score + rerank_weight * lex
        pool.sort(key=lambda s: s.score, reverse=True)
        return pool[:k]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
