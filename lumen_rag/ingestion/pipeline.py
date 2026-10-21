"""Document → chunks → embeddings → vector store."""
from __future__ import annotations

import uuid
from pathlib import Path

from ..config import settings
from ..embeddings import Embedder, get_embedder
from ..store import Chunk, Store, VectorStore
from .chunker import chunk_text
from .context import chunk_headers, contextual_text
from .loaders import load_file


def ingest_documents(
    documents: list[dict],
    store: Store | None = None,
    embedder: Embedder | None = None,
    chunk_size: int = 120,
    overlap: int = 20,
    contextual: bool | None = None,
) -> Store:
    """Ingest a list of ``{"id", "text", "metadata"?}`` dicts into a store.

    With ``contextual`` (default: ``LUMEN_CONTEXTUAL``) each chunk gets a header naming
    its document and section, stored in ``metadata["context"]`` and embedded with it.
    """
    contextual = settings.contextual if contextual is None else contextual
    embedder = embedder or get_embedder()
    # NB: use an explicit None check — an empty VectorStore is falsy (__len__ == 0),
    # so `store or VectorStore(...)` would discard a passed-in empty store.
    if store is None:
        store = VectorStore(dim=embedder.dim)

    chunks: list[Chunk] = []
    for doc in documents:
        doc_id = str(doc.get("id") or uuid.uuid4())
        pieces = chunk_text(doc["text"], chunk_size, overlap)
        headers = chunk_headers({**doc, "id": doc_id}, pieces) if contextual else [None] * len(pieces)
        for piece, header in zip(pieces, headers, strict=True):
            metadata = dict(doc.get("metadata", {}))
            if header:
                metadata["context"] = header
            chunks.append(Chunk(id=str(uuid.uuid4()), text=piece, doc_id=doc_id, metadata=metadata))

    if chunks:
        vectors = embedder.embed([contextual_text(c.text, c.metadata) for c in chunks])
        store.add(chunks, vectors)
    return store


def ingest_paths(paths: list[str | Path], **kwargs) -> Store:
    """Ingest files from disk. Supports .txt, .md, .pdf, .docx, .html, .htm."""
    docs = [load_file(p) for p in paths]
    return ingest_documents(docs, **kwargs)
