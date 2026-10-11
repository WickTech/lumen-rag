"""Cross-encoder reranking.

First-stage retrieval (vector / BM25 / hybrid) is cheap but scores query and
passage independently. A cross-encoder reads them *together*, which is far
more precise, but too slow to run over the whole corpus — so the retriever
over-fetches a candidate pool and the reranker reorders it.

Backed by fastembed's ONNX ``TextCrossEncoder`` (no torch), default model
``BAAI/bge-reranker-base``. Optional dependency: ``pip install 'lumen-rag[rerank]'``.
Enabled by setting ``LUMEN_RERANKER`` to a model name; empty means off.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from ..config import settings
from ..store import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        ...


class CrossEncoderReranker:
    """Reorders candidates by cross-encoder relevance score (higher is better)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as e:
            raise ImportError(
                "fastembed is required for cross-encoder reranking. "
                "Install it with: pip install 'lumen-rag[rerank]'"
            ) from e

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = list(self._model.rerank(query, [c.chunk.text for c in candidates]))
        rescored = [
            ScoredChunk(chunk=c.chunk, score=float(s)) for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored[:k]


@lru_cache(maxsize=4)
def _load(model_name: str) -> CrossEncoderReranker:
    # Model load is the expensive part (ONNX session + weights); share it across
    # the many short-lived Retriever instances the API and eval paths create.
    return CrossEncoderReranker(model_name)


def get_reranker() -> Reranker | None:
    """Return the configured reranker, or None when ``LUMEN_RERANKER`` is unset."""
    if not settings.reranker_model:
        return None
    return _load(settings.reranker_model)
