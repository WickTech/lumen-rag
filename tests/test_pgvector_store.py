"""pgvector adapter against a real Postgres. Skipped unless LUMEN_TEST_PG_DSN is set.

Locally:  docker compose up -d db
          LUMEN_TEST_PG_DSN=postgresql://lumen:lumen@localhost:5432/lumen pytest -m pgvector
CI runs these against a pgvector service container.
"""
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

from lumen_rag import config
from lumen_rag.embeddings import get_embedder
from lumen_rag.engine import RagEngine
from lumen_rag.eval import evaluate
from lumen_rag.eval.harness import load_cases
from lumen_rag.ingestion.loaders import load_file
from lumen_rag.ingestion.pipeline import ingest_documents
from lumen_rag.retrieval import Retriever
from lumen_rag.store import Chunk, VectorStore

DSN = os.environ.get("LUMEN_TEST_PG_DSN", "")
pytestmark = [
    pytest.mark.pgvector,
    pytest.mark.skipif(not DSN, reason="LUMEN_TEST_PG_DSN not set"),
]

DATA = Path(__file__).parent.parent / "data" / "eval_v2"


@pytest.fixture
def table():
    """A private table per test so runs never collide; dropped afterwards."""
    import psycopg

    name = f"lumen_test_{uuid.uuid4().hex[:12]}"
    yield name
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")


def _store(table, **kwargs):
    from lumen_rag.pgvector_store import PgVectorStore

    kwargs.setdefault("dim", get_embedder().dim)
    return PgVectorStore(DSN, table=table, **kwargs)


def _unit(rng, n, dim):
    m = rng.normal(size=(n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _chunks(prefix, n):
    return [Chunk(id=f"{prefix}{i}", text=f"text {prefix}{i}", doc_id=f"{prefix}-doc", metadata={"i": i}) for i in range(n)]


def test_search_matches_the_in_memory_store_exactly(table):
    rng = np.random.default_rng(0)
    dim = 16
    chunks, vectors = _chunks("c", 40), _unit(rng, 40, dim)
    memory = VectorStore(dim=dim)
    memory.add(chunks, vectors)
    with _store(table, dim=dim) as pg:
        pg.add(chunks, vectors)
        assert len(pg) == 40
        for q in _unit(rng, 10, dim):
            want, got = memory.search(q, k=5), pg.search(q, k=5)
            assert [s.chunk.id for s in got] == [s.chunk.id for s in want]
            assert [s.score for s in got] == pytest.approx([s.score for s in want], abs=1e-5)


def test_chunks_keep_insertion_order_and_metadata(table):
    rng = np.random.default_rng(1)
    chunks = _chunks("c", 5)
    with _store(table, dim=8) as pg:
        pg.add(chunks[:3], _unit(rng, 3, 8))
        pg.add(chunks[3:], _unit(rng, 2, 8))
        assert pg.chunks() == chunks


def test_namespaces_are_isolated_on_a_shared_table(table):
    rng = np.random.default_rng(2)
    with _store(table, dim=8, namespace="team-a") as a, _store(table, dim=8, namespace="team-b") as b:
        a.add(_chunks("a", 3), _unit(rng, 3, 8))
        b.add(_chunks("b", 5), _unit(rng, 5, 8))
        assert (len(a), len(b)) == (3, 5)
        assert {c.doc_id for c in a.chunks()} == {"a-doc"}
        assert {c.doc_id for c in b.chunks()} == {"b-doc"}
        query = _unit(rng, 1, 8)[0]
        assert {s.chunk.doc_id for s in a.search(query, k=10)} == {"a-doc"}
        assert len(b.search(query, k=10)) == 5
        assert a.clear() == 3
        assert (len(a), len(b)) == (0, 5)


def test_data_persists_across_connections(table):
    rng = np.random.default_rng(3)
    chunks = _chunks("p", 4)
    with _store(table, dim=8, namespace="keep") as first:
        first.add(chunks, _unit(rng, 4, 8))
    with _store(table, dim=8, namespace="keep") as second:
        assert second.chunks() == chunks


def test_re_adding_a_chunk_id_replaces_it(table):
    rng = np.random.default_rng(4)
    with _store(table, dim=8) as pg:
        pg.add([Chunk("x", "old", "d", {})], _unit(rng, 1, 8))
        pg.add([Chunk("x", "new", "d", {"v": 2})], _unit(rng, 1, 8))
        assert len(pg) == 1
        assert pg.chunks()[0].text == "new" and pg.chunks()[0].metadata == {"v": 2}


def test_dimension_mismatches_are_rejected(table):
    rng = np.random.default_rng(5)
    with _store(table, dim=8) as pg:
        with pytest.raises(ValueError, match="expected dim 8"):
            pg.add(_chunks("c", 1), _unit(rng, 1, 4))
        with pytest.raises(ValueError, match="length mismatch"):
            pg.add(_chunks("c", 2), _unit(rng, 1, 8))
    with pytest.raises(ValueError, match="8-dimensional"):
        _store(table, dim=16)


def test_hnsw_index_can_be_created_and_searched(table):
    rng = np.random.default_rng(6)
    with _store(table, dim=8, index="hnsw") as pg:
        pg.add(_chunks("h", 20), _unit(rng, 20, 8))
        assert len(pg.search(_unit(rng, 1, 8)[0], k=3)) == 3


def test_empty_store_search_returns_nothing(table):
    with _store(table, dim=8) as pg:
        assert pg.search(np.ones(8, dtype=np.float32) / np.sqrt(8), k=5) == []
        assert pg.chunks() == [] and len(pg) == 0


def test_eval_v2_metrics_match_the_in_memory_store(table):
    """The M3 acceptance check: same documents, same embedder, same retrieval quality."""
    embedder = get_embedder()
    docs = [load_file(p) for p in sorted((DATA / "docs").glob("*.md"))]
    cases = [c for c in load_cases(DATA / "eval.jsonl") if c.answerable]

    memory = VectorStore(dim=embedder.dim)
    ingest_documents(docs, store=memory, embedder=embedder, chunk_size=120, overlap=20)
    with _store(table, dim=embedder.dim) as pg:
        ingest_documents(docs, store=pg, embedder=embedder, chunk_size=120, overlap=20)
        assert len(pg) == len(memory)
        mem_retriever = Retriever(memory, embedder, reranker=None)
        pg_retriever = Retriever(pg, embedder, reranker=None)
        for mode in ("vector", "bm25", "hybrid"):
            want = evaluate(mem_retriever, cases, k=5, mode=mode).as_dict()
            got = evaluate(pg_retriever, cases, k=5, mode=mode).as_dict()
            for key in ("recall@k", "mrr", "ndcg@k", "hit_rate"):
                assert got[key] == pytest.approx(want[key], abs=0.005), (mode, key)


def test_engine_runs_end_to_end_on_pgvector(table, monkeypatch):
    monkeypatch.setattr(config.settings, "store", "pgvector", raising=False)
    monkeypatch.setattr(config.settings, "pg_dsn", DSN, raising=False)
    monkeypatch.setattr(config.settings, "pg_table", table, raising=False)
    monkeypatch.setattr(config.settings, "namespace", "engine-test", raising=False)
    engine = RagEngine()
    try:
        engine.add_documents(
            [{"id": "leave", "text": "Employees receive thirty working days of annual paid leave."},
             {"id": "robots", "text": "The H200 robot lifts thirty five kilograms."}],
            chunk_size=50, overlap=0,
        )
        assert engine.query("how many days of annual leave", k=1).citations[0]["doc_id"] == "leave"
        # A second engine on the same namespace sees the data without re-ingesting.
        reopened = RagEngine.load()
        assert len(reopened.store) == 2
        reopened.store.close()
    finally:
        engine.store.close()
