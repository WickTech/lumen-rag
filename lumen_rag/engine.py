"""High-level facade tying ingestion, retrieval, and answering together."""
from __future__ import annotations

from pathlib import Path

from .config import settings
from .embeddings import get_embedder
from .ingestion import ingest_documents
from .llm import Answer, answer
from .retrieval import Retriever, RetrievalMode
from .store import VectorStore


class RagEngine:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.embedder = get_embedder()
        self.store = store or VectorStore(dim=self.embedder.dim)
        self.retriever = Retriever(self.store, self.embedder)

    def add_documents(self, documents: list[dict], **kwargs) -> int:
        ingest_documents(documents, store=self.store, embedder=self.embedder, **kwargs)
        self.retriever._bm25 = None  # invalidate cached BM25 index
        return len(self.store)

    def query(self, question: str, k: int = 5, mode: RetrievalMode = "hybrid") -> Answer:
        chunks = self.retriever.retrieve(question, k=k, mode=mode)
        return answer(question, chunks)

    # --- persistence -------------------------------------------------------
    def save(self, directory: str | Path | None = None) -> None:
        self.store.save(directory or settings.index_dir)

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "RagEngine":
        store = VectorStore.load(directory or settings.index_dir)
        return cls(store=store)
