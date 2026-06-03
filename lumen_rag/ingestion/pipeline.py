"""Document → chunks → embeddings → vector store."""
from __future__ import annotations

import uuid
from pathlib import Path

from ..embeddings import Embedder, get_embedder
from ..store import Chunk, VectorStore
from .chunker import chunk_text


def ingest_documents(
    documents: list[dict],
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    chunk_size: int = 120,
    overlap: int = 20,
) -> VectorStore:
    """Ingest a list of ``{"id", "text", "metadata"?}`` dicts into a store."""
    embedder = embedder or get_embedder()
    # NB: use an explicit None check — an empty VectorStore is falsy (__len__ == 0),
    # so `store or VectorStore(...)` would discard a passed-in empty store.
    if store is None:
        store = VectorStore(dim=embedder.dim)

    chunks: list[Chunk] = []
    for doc in documents:
        doc_id = str(doc.get("id") or uuid.uuid4())
        for piece in chunk_text(doc["text"], chunk_size, overlap):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=piece,
                    doc_id=doc_id,
                    metadata=dict(doc.get("metadata", {})),
                )
            )

    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        store.add(chunks, vectors)
    return store


def ingest_paths(paths: list[str | Path], **kwargs) -> VectorStore:
    """Ingest .txt / .md files from disk."""
    docs = []
    for p in paths:
        path = Path(p)
        docs.append(
            {"id": path.stem, "text": path.read_text(encoding="utf-8"),
             "metadata": {"source": str(path)}}
        )
    return ingest_documents(docs, **kwargs)
