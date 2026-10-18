"""High-level facade tying ingestion, retrieval, and answering together."""
from __future__ import annotations

import asyncio
from pathlib import Path

from . import tracing
from .config import settings
from .embeddings import get_embedder
from .ingestion import ingest_documents
from .llm import Answer, answer, answer_async
from .retrieval import RetrievalMode, Retriever
from .store import Store, VectorStore, get_store


class RagEngine:
    def __init__(self, store: Store | None = None) -> None:
        self.embedder = get_embedder()
        # Explicit None check: an empty store is falsy (``__len__`` == 0), so ``store or ...``
        # would silently replace a freshly created pgvector store with an in-memory one.
        self.store = store if store is not None else get_store(self.embedder.dim)
        self.retriever = Retriever(self.store, self.embedder)

    def add_documents(self, documents: list[dict], **kwargs) -> int:
        ingest_documents(documents, store=self.store, embedder=self.embedder, **kwargs)
        self.retriever._bm25 = None  # invalidate cached BM25 index
        return len(self.store)

    def query(self, question: str, k: int = 5, mode: RetrievalMode = "hybrid") -> Answer:
        with tracing.span("lumen.query", **self._query_attrs(question, k, mode)) as span:
            chunks = self.retriever.retrieve(question, k=k, mode=mode)
            result = answer(question, chunks)
            tracing.set_attr(span, "lumen.chunks", len(chunks))
            return result

    async def aquery(self, question: str, k: int = 5, mode: RetrievalMode = "hybrid") -> Answer:
        """``query`` for use inside an event loop.

        Retrieval (CPU, or a database / embedding-API round trip) runs in a worker
        thread; the model call is awaited on the async client. Neither blocks the loop.
        """
        with tracing.span("lumen.query", **self._query_attrs(question, k, mode)) as span:
            # to_thread copies the context, so retrieval spans stay children of lumen.query.
            chunks = await asyncio.to_thread(self.retriever.retrieve, question, k=k, mode=mode)
            result = await answer_async(question, chunks)
            tracing.set_attr(span, "lumen.chunks", len(chunks))
            return result

    @staticmethod
    def _query_attrs(question: str, k: int, mode: str) -> dict:
        return {
            "lumen.k": k,
            "lumen.mode": mode,
            "lumen.question.chars": len(question),
            "lumen.question.text": tracing.content_attr(question),
        }

    # --- persistence -------------------------------------------------------
    def save(self, directory: str | Path | None = None) -> None:
        self.store.save(directory or settings.index_dir)

    @classmethod
    def load(cls, directory: str | Path | None = None) -> RagEngine:
        if settings.store.lower() != "memory":
            # Database-backed stores persist themselves: reconnect instead of reading files.
            return cls()
        store = VectorStore.load(directory or settings.index_dir)
        return cls(store=store)
