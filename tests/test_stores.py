"""The Store interface: backend selection, decoupling, and adapter validation (no database)."""
import sys

import numpy as np
import pytest

from lumen_rag import config
from lumen_rag.embeddings import get_embedder
from lumen_rag.engine import RagEngine
from lumen_rag.pgvector_store import PgVectorStore, _vector_literal
from lumen_rag.retrieval import Retriever
from lumen_rag.store import Chunk, ScoredChunk, VectorStore, get_store


class MinimalStore:
    """Implements only the public ``Store`` interface: no ``_chunks``, no numpy matrix."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._rows: list[tuple[Chunk, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, chunks, vectors) -> None:
        self._rows.extend(zip(chunks, vectors, strict=True))

    def search(self, query_vec, k: int = 5):
        scored = [ScoredChunk(c, float(v @ query_vec)) for c, v in self._rows]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:k]

    def chunks(self) -> list[Chunk]:
        return [c for c, _ in self._rows]

    def save(self, directory) -> None:
        pass


DOCS = [
    {"id": "leave", "text": "Employees receive thirty working days of annual paid leave."},
    {"id": "sick", "text": "Sick days are separate from vacation and are paid in full."},
    {"id": "robots", "text": "The H200 robot lifts thirty five kilograms to the top shelf."},
]


def test_memory_store_exposes_chunks_and_dim():
    store = VectorStore(dim=4)
    chunk = Chunk(id="a", text="hello", doc_id="d", metadata={})
    store.add([chunk], np.eye(4, dtype=np.float32)[:1])
    assert store.chunks() == [chunk]
    assert store.dim == 4 and len(store) == 1


def test_retriever_works_with_any_store_that_implements_the_interface():
    engine_store = MinimalStore(dim=get_embedder().dim)
    from lumen_rag.ingestion.pipeline import ingest_documents

    ingest_documents(DOCS, store=engine_store, embedder=get_embedder(), chunk_size=50, overlap=0)
    retriever = Retriever(engine_store, get_embedder(), reranker=None)
    for mode in ("vector", "bm25", "hybrid"):
        top = retriever.retrieve("how many days of annual leave", k=1, mode=mode)
        assert top[0].chunk.doc_id == "leave", mode


def test_engine_keeps_an_empty_store_it_was_given():
    """An empty store is falsy; the engine must not swap it for a fresh in-memory one."""
    custom = MinimalStore(dim=get_embedder().dim)
    assert len(custom) == 0
    engine = RagEngine(store=custom)
    assert engine.store is custom
    engine.add_documents(DOCS, chunk_size=50, overlap=0)
    assert len(custom) == 3
    assert engine.query("annual paid leave", k=1).citations[0]["doc_id"] == "leave"


def test_get_store_selects_the_backend(monkeypatch):
    monkeypatch.setattr(config.settings, "store", "memory", raising=False)
    assert isinstance(get_store(8), VectorStore)

    monkeypatch.setattr(config.settings, "store", "nope", raising=False)
    with pytest.raises(ValueError, match="unknown LUMEN_STORE"):
        get_store(8)

    monkeypatch.setattr(config.settings, "store", "pgvector", raising=False)
    monkeypatch.setattr(config.settings, "pg_dsn", "", raising=False)
    with pytest.raises(RuntimeError, match="LUMEN_PG_DSN"):
        get_store(8)


def test_engine_load_reconnects_instead_of_reading_files_for_database_stores(monkeypatch, tmp_path):
    sentinel = MinimalStore(dim=get_embedder().dim)
    monkeypatch.setattr(config.settings, "store", "pgvector", raising=False)
    monkeypatch.setattr("lumen_rag.engine.get_store", lambda dim: sentinel)
    engine = RagEngine.load(tmp_path / "does-not-exist")
    assert engine.store is sentinel


@pytest.mark.parametrize("table", ["Robert'; DROP TABLE x;--", "has space", "1abc", "", "a" * 41, "UPPER"])
def test_pgvector_rejects_unsafe_table_names_before_connecting(table):
    with pytest.raises(ValueError, match="invalid table name"):
        PgVectorStore("postgresql://nowhere", dim=8, table=table)


def test_pgvector_validates_other_arguments_before_connecting():
    with pytest.raises(ValueError, match="namespace"):
        PgVectorStore("postgresql://nowhere", dim=8, namespace="")
    with pytest.raises(ValueError, match="index"):
        PgVectorStore("postgresql://nowhere", dim=8, index="ivfflat")
    with pytest.raises(ValueError, match="dim"):
        PgVectorStore("postgresql://nowhere", dim=0)


def test_pgvector_explains_the_missing_driver(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)  # makes `import psycopg` raise ImportError
    with pytest.raises(ImportError, match=r"lumen-rag\[pgvector\]"):
        PgVectorStore("postgresql://nowhere", dim=8)


def test_vector_literal_is_pgvector_text_form():
    assert _vector_literal(np.array([0.5, -1.0, 0.0], dtype=np.float32)) == "[0.5,-1.0,0.0]"
