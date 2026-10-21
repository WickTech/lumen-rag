"""High-level facade tying ingestion, retrieval, and answering together."""
from __future__ import annotations

import asyncio
from pathlib import Path

from . import tracing
from .config import settings
from .embeddings import get_embedder
from .ingestion import ingest_documents
from .llm import Answer, abstain, answer, answer_async
from .retrieval import RetrievalMode, Retriever
from .store import ScoredChunk, Store, VectorStore, get_store


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

    def retrieve(
        self, question: str, k: int = 5, mode: RetrievalMode = "hybrid"
    ) -> tuple[list[ScoredChunk], float | None, bool]:
        """Retrieve context and decide whether to abstain.

        Returns ``(chunks, confidence, abstained)``. With ``LUMEN_ABSTAIN_THRESHOLD``
        unset this is plain retrieval: confidence is None and it never abstains.
        """
        threshold = settings.abstain_threshold
        if threshold is None:
            return self.retriever.retrieve(question, k=k, mode=mode), None, False
        chunks, confidence = self.retriever.retrieve_with_confidence(question, k=k, mode=mode)
        return chunks, confidence, confidence < threshold

    def query(self, question: str, k: int = 5, mode: RetrievalMode = "hybrid") -> Answer:
        with tracing.span("lumen.query", **self._query_attrs(question, k, mode)) as span:
            chunks, confidence, abstained = self.retrieve(question, k=k, mode=mode)
            result = abstain(confidence) if abstained else answer(question, chunks)
            result.confidence = confidence
            tracing.set_attr(span, "lumen.chunks", len(chunks))
            tracing.set_attr(span, "lumen.abstained", abstained)
            return result

    async def aquery(self, question: str, k: int = 5, mode: RetrievalMode = "hybrid") -> Answer:
        """``query`` for use inside an event loop.

        Retrieval (CPU, or a database / embedding-API round trip) runs in a worker
        thread; the model call is awaited on the async client. Neither blocks the loop.
        """
        with tracing.span("lumen.query", **self._query_attrs(question, k, mode)) as span:
            # to_thread copies the context, so retrieval spans stay children of lumen.query.
            chunks, confidence, abstained = await asyncio.to_thread(self.retrieve, question, k, mode)
            result = abstain(confidence) if abstained else await answer_async(question, chunks)
            result.confidence = confidence
            tracing.set_attr(span, "lumen.chunks", len(chunks))
            tracing.set_attr(span, "lumen.abstained", abstained)
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
        engine = cls(store=store)
        if store.dim != engine.embedder.dim:
            if len(store):
                raise RuntimeError(
                    f"The index in {directory or settings.index_dir} has {store.dim}-dim vectors but the "
                    f"configured embedder ({settings.embedder}) produces {engine.embedder.dim}-dim ones. "
                    "Re-ingest with this embedder (lumen ingest) or set LUMEN_EMBEDDER to the one used."
                )
            engine = cls()  # empty saved index: start fresh at the right dim
        return engine
