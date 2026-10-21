"""High-level facade tying ingestion, retrieval, and answering together."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from . import tracing
from .config import settings
from .embeddings import get_embedder
from .ingestion import ingest_documents
from .llm import Answer, abstain, answer, answer_async
from .reader import Reading, get_reader
from .retrieval import RetrievalMode, Retriever
from .store import ScoredChunk, Store, VectorStore, get_store


@dataclass
class Retrieval:
    chunks: list[ScoredChunk]
    # Retrieval confidence; None unless LUMEN_ABSTAIN_THRESHOLD is set.
    confidence: float | None
    # Below the retrieval or the reader threshold: refuse instead of answering.
    abstained: bool
    # The reader's verdict; None unless LUMEN_READER is set.
    reading: Reading | None = None

    def refusal(self) -> Answer:
        result = abstain(self.confidence)
        result.answerability = self.reading.answerability if self.reading else None
        return result


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

    def retrieve(self, question: str, k: int = 5, mode: RetrievalMode = "hybrid") -> Retrieval:
        """Retrieve context and decide whether to abstain.

        Two optional gates, cheapest first. ``LUMEN_ABSTAIN_THRESHOLD`` refuses when
        the question is not similar enough to anything indexed; ``LUMEN_READER`` reads
        the retrieved passages and refuses when none of them contains an answer.
        With neither set this is plain retrieval and never abstains.
        """
        threshold = settings.abstain_threshold
        if threshold is None:
            chunks, confidence = self.retriever.retrieve(question, k=k, mode=mode), None
        else:
            chunks, confidence = self.retriever.retrieve_with_confidence(question, k=k, mode=mode)
            if confidence < threshold:
                return Retrieval(chunks, confidence, True)
        reader = get_reader()
        if reader is None or not chunks:
            return Retrieval(chunks, confidence, False)
        with tracing.span("lumen.read", **{"lumen.reader": reader.model_id}) as span:
            reading = reader.read(question, chunks)
            tracing.set_attr(span, "lumen.answerability", round(reading.answerability, 3))
        return Retrieval(chunks, confidence, reading.answerability < settings.reader_threshold, reading)

    @staticmethod
    def _answer_reading(r: Retrieval) -> Reading | None:
        # The reader always gates; its span shapes the answer only when asked to.
        return r.reading if settings.reader_answers else None

    def _finish(self, r: Retrieval, result: Answer, span) -> Answer:
        result.confidence = r.confidence
        if r.reading is not None:
            result.answerability = r.reading.answerability
        tracing.set_attr(span, "lumen.chunks", len(r.chunks))
        tracing.set_attr(span, "lumen.abstained", r.abstained)
        return result

    def answer_with_context(
        self, question: str, k: int = 5, mode: RetrievalMode = "hybrid"
    ) -> tuple[list[ScoredChunk], Answer]:
        """``query`` that also returns the retrieved chunks (for evaluation)."""
        with tracing.span("lumen.query", **self._query_attrs(question, k, mode)) as span:
            r = self.retrieve(question, k=k, mode=mode)
            result = r.refusal() if r.abstained else answer(question, r.chunks, reading=self._answer_reading(r))
            return r.chunks, self._finish(r, result, span)

    def query(self, question: str, k: int = 5, mode: RetrievalMode = "hybrid") -> Answer:
        return self.answer_with_context(question, k=k, mode=mode)[1]

    async def aquery(self, question: str, k: int = 5, mode: RetrievalMode = "hybrid") -> Answer:
        """``query`` for use inside an event loop.

        Retrieval (CPU, or a database / embedding-API round trip) and reading run in a
        worker thread; the model call is awaited on the async client. Neither blocks the loop.
        """
        with tracing.span("lumen.query", **self._query_attrs(question, k, mode)) as span:
            # to_thread copies the context, so retrieval spans stay children of lumen.query.
            r = await asyncio.to_thread(self.retrieve, question, k, mode)
            result = (
                r.refusal()
                if r.abstained
                else await answer_async(question, r.chunks, reading=self._answer_reading(r))
            )
            return self._finish(r, result, span)

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
