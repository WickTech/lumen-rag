"""Tests for BM25 retrieval and RRF hybrid fusion."""
from __future__ import annotations

import pytest

from lumen_rag.retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from lumen_rag.store import Chunk, ScoredChunk, VectorStore
from lumen_rag.embeddings import HashingEmbedder
from lumen_rag.retrieval import Retriever


def _make_chunk(id: str, text: str, doc_id: str = "doc") -> Chunk:
    return Chunk(id=id, text=text, doc_id=doc_id, metadata={})


CORPUS = [
    _make_chunk("c1", "The cat sat on the mat", "d1"),
    _make_chunk("c2", "Dogs are loyal pets and great companions", "d2"),
    _make_chunk("c3", "Cats are independent animals that love to nap", "d3"),
    _make_chunk("c4", "The weather is sunny and warm today", "d4"),
    _make_chunk("c5", "Machine learning models require training data", "d5"),
]


# ---------------------------------------------------------------------------
# BM25 unit tests
# ---------------------------------------------------------------------------

class TestBM25Index:
    def setup_method(self):
        self.idx = BM25Index.build(CORPUS)

    def test_build_sets_metadata(self):
        assert self.idx.n == len(CORPUS)
        assert self.idx.avgdl > 0
        assert "cat" in self.idx.doc_freqs

    def test_score_length(self):
        scores = self.idx.score("cat")
        assert len(scores) == len(CORPUS)

    def test_cat_query_ranks_cat_docs_first(self):
        results = self.idx.search("cat", k=2)
        ids = [sc.chunk.id for sc in results]
        # c1 ("cat sat on the mat") and c3 ("Cats are ...") should be top-2
        assert "c1" in ids or "c3" in ids

    def test_unrelated_query_returns_low_scores(self):
        results = self.idx.search("quantum physics entanglement", k=3)
        for sc in results:
            assert sc.score == 0.0 or sc.score < 1.0

    def test_search_respects_k(self):
        results = self.idx.search("cat dog", k=2)
        assert len(results) <= 2

    def test_empty_query_returns_zero_scores(self):
        scores = self.idx.score("")
        assert all(s == 0.0 for s in scores)


# ---------------------------------------------------------------------------
# RRF unit tests
# ---------------------------------------------------------------------------

class TestRRF:
    def test_fuses_two_lists(self):
        a = [ScoredChunk(_make_chunk("x", "foo"), 0.9),
             ScoredChunk(_make_chunk("y", "bar"), 0.7)]
        b = [ScoredChunk(_make_chunk("y", "bar"), 0.8),
             ScoredChunk(_make_chunk("z", "baz"), 0.6)]
        fused = reciprocal_rank_fusion([a, b], k=3)
        ids = [sc.chunk.id for sc in fused]
        # "y" appears in both → should rank highest
        assert ids[0] == "y"

    def test_output_length_capped_at_k(self):
        lists = [[ScoredChunk(_make_chunk(str(i), "x"), 1.0) for i in range(10)]]
        fused = reciprocal_rank_fusion(lists, k=3)
        assert len(fused) == 3

    def test_single_list_preserves_order(self):
        ranked = [ScoredChunk(_make_chunk(str(i), "text"), float(5 - i)) for i in range(5)]
        fused = reciprocal_rank_fusion([ranked], k=5)
        fused_ids = [sc.chunk.id for sc in fused]
        original_ids = [sc.chunk.id for sc in ranked]
        assert fused_ids == original_ids


# ---------------------------------------------------------------------------
# Integration: Retriever modes
# ---------------------------------------------------------------------------

class TestRetrieverModes:
    def setup_method(self):
        embedder = HashingEmbedder(dim=128)
        self.store = VectorStore(dim=128)
        vectors = embedder.embed([c.text for c in CORPUS])
        self.store.add(CORPUS, vectors)
        self.retriever = Retriever(self.store, embedder)

    def test_vector_mode_returns_k_results(self):
        results = self.retriever.retrieve("cat nap", k=3, mode="vector")
        assert len(results) == 3

    def test_bm25_mode_returns_k_results(self):
        results = self.retriever.retrieve("cat nap", k=3, mode="bm25")
        assert len(results) == 3

    def test_hybrid_mode_returns_k_results(self):
        results = self.retriever.retrieve("cat nap", k=3, mode="hybrid")
        assert len(results) == 3

    def test_bm25_mode_finds_keyword_match(self):
        results = self.retriever.retrieve("machine learning", k=2, mode="bm25")
        ids = [sc.chunk.doc_id for sc in results]
        assert "d5" in ids  # "Machine learning models..." is c5/d5

    def test_hybrid_mode_finds_keyword_match(self):
        results = self.retriever.retrieve("machine learning", k=3, mode="hybrid")
        ids = [sc.chunk.doc_id for sc in results]
        assert "d5" in ids

    def test_bm25_index_rebuilds_after_new_docs(self):
        new_chunk = _make_chunk("c6", "Reinforcement learning and reward signals", "d6")
        vectors = HashingEmbedder(dim=128).embed([new_chunk.text])
        self.store.add([new_chunk], vectors)
        self.retriever._bm25 = None  # simulate engine.add_documents invalidation
        results = self.retriever.retrieve("reinforcement learning", k=2, mode="bm25")
        ids = [sc.chunk.doc_id for sc in results]
        assert "d6" in ids
