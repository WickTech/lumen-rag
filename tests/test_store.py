import numpy as np

from lumen_rag.embeddings import HashingEmbedder
from lumen_rag.store import Chunk, VectorStore


def _store_with(texts):
    emb = HashingEmbedder(dim=256)
    vecs = emb.embed(texts)
    chunks = [Chunk(id=str(i), text=t, doc_id=f"d{i}", metadata={}) for i, t in enumerate(texts)]
    store = VectorStore(dim=256)
    store.add(chunks, vecs)
    return emb, store


def test_search_returns_most_similar_first():
    emb, store = _store_with(
        ["the cat sat on the mat", "stock market crashed today", "a dog ran in the park"]
    )
    q = emb.embed(["where did the cat sit"])[0]
    results = store.search(q, k=3)
    assert results[0].chunk.text == "the cat sat on the mat"
    # scores must be sorted descending
    assert results[0].score >= results[1].score >= results[2].score


def test_dim_mismatch_raises():
    store = VectorStore(dim=8)
    import pytest

    with pytest.raises(ValueError):
        store.add([Chunk("1", "x", "d", {})], np.zeros((1, 4), dtype=np.float32))


def test_save_and_load_roundtrip(tmp_path):
    _, store = _store_with(["alpha beta", "gamma delta"])
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 2
    assert loaded.dim == store.dim
