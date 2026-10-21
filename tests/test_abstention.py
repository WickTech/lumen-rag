"""Embedder selection, confidence-gated abstention, and its calibration metrics."""
import importlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen_rag import config, llm
from lumen_rag.embeddings import HashingEmbedder, get_embedder
from lumen_rag.engine import RagEngine
from lumen_rag.eval.abstention import ScoredCase, auroc, best_threshold, calibrate, outcome, split
from lumen_rag.retrieval import Retriever
from lumen_rag.store import VectorStore

api = importlib.import_module("lumen_rag.api.app")

DOCS = [
    {"id": "leave", "text": "Employees receive thirty working days of annual paid leave."},
    {"id": "robots", "text": "The H200 robot lifts thirty five kilograms to the top shelf."},
]


@pytest.fixture
def engine():
    eng = RagEngine()
    eng.add_documents(DOCS, chunk_size=50, overlap=0)
    return eng


# --- embedder selection -------------------------------------------------------


def test_auto_embedder_is_hashing_offline():
    assert isinstance(get_embedder(), HashingEmbedder)


def test_openai_embedder_without_key_fails_clearly(monkeypatch):
    monkeypatch.setattr(config.settings, "embedder", "openai")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_embedder()


def test_unknown_embedder_rejected(monkeypatch):
    monkeypatch.setattr(config.settings, "embedder", "word2vec")
    with pytest.raises(ValueError, match="LUMEN_EMBEDDER"):
        get_embedder()


def test_fastembed_embedder_is_normalised(monkeypatch):
    pytest.importorskip("fastembed")
    import os

    if not os.getenv("LUMEN_TEST_FASTEMBED"):
        pytest.skip("downloads a model; set LUMEN_TEST_FASTEMBED=1 to run")
    monkeypatch.setattr(config.settings, "embedder", "fastembed")
    emb = get_embedder()
    vecs = emb.embed(["annual leave", "robot payload"])
    assert vecs.shape == (2, emb.dim)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert emb.embed_query("annual leave").shape == (emb.dim,)


def test_loading_index_with_other_dim_fails_clearly(tmp_path, monkeypatch):
    store = VectorStore(dim=8)
    from lumen_rag.store import Chunk

    store.add([Chunk(id="c", doc_id="d", text="x", metadata={})], np.ones((1, 8), dtype=np.float32))
    store.save(tmp_path)
    with pytest.raises(RuntimeError, match="Re-ingest"):
        RagEngine.load(tmp_path)


# --- confidence and abstention ---------------------------------------------------


def test_confidence_is_top_dense_cosine(engine):
    results, confidence = engine.retriever.retrieve_with_confidence("annual paid leave", k=1, mode="bm25")
    q = HashingEmbedder().embed(["annual paid leave"])[0]
    assert confidence == pytest.approx(engine.store.search(q, k=1)[0].score)
    assert results[0].chunk.doc_id == "leave"


def test_empty_store_has_zero_confidence():
    retriever = Retriever(VectorStore(dim=512), HashingEmbedder(), reranker=None)
    assert retriever.retrieve_with_confidence("anything") == ([], 0.0)


def test_threshold_unset_keeps_old_behaviour(engine, monkeypatch):
    called = []
    monkeypatch.setattr(engine.retriever, "retrieve_with_confidence", lambda *a, **k: called.append(1))
    result = engine.query("annual paid leave")
    assert not called
    assert not result.abstained and result.confidence is None
    assert result.citations


def test_below_threshold_abstains_without_calling_the_model(engine, monkeypatch):
    monkeypatch.setattr(config.settings, "abstain_threshold", 0.99)
    monkeypatch.setattr(llm, "answer", lambda *a, **k: pytest.fail("model must not be called"))
    result = engine.query("what is the capital of peru")
    assert result.abstained
    assert result.text == llm.ABSTAIN_TEXT
    assert result.citations == []
    assert 0.0 <= result.confidence < 0.99


def test_above_threshold_answers(engine, monkeypatch):
    monkeypatch.setattr(config.settings, "abstain_threshold", 0.01)
    result = engine.query("annual paid leave")
    assert not result.abstained
    assert result.confidence > 0.01
    assert result.citations


async def test_aquery_abstains(engine, monkeypatch):
    monkeypatch.setattr(config.settings, "abstain_threshold", 0.99)
    result = await engine.aquery("what is the capital of peru")
    assert result.abstained


def test_api_reports_abstention(engine, monkeypatch):
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setattr(api, "enforce", lambda *a, **k: None)
    client = TestClient(api.app)

    body = client.post("/query", json={"question": "annual paid leave"}).json()
    assert body["abstained"] is False and body["confidence"] is None

    monkeypatch.setattr(config.settings, "abstain_threshold", 0.99)
    body = client.post("/query", json={"question": "capital of peru"}).json()
    assert body["abstained"] is True
    assert body["citations"] == []

    with client.stream("POST", "/query/stream", json={"question": "capital of peru"}) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    assert events[0] == {"token": llm.ABSTAIN_TEXT}
    assert events[-1]["done"] and events[-1]["abstained"] is True


# --- calibration metrics ------------------------------------------------------------


def _case(q, answerable, conf, hit=True):
    return ScoredCase(q, None, answerable, conf, hit if answerable else None)


def test_auroc_perfect_and_chance():
    perfect = [_case("a", True, 0.9), _case("b", True, 0.8), _case("c", False, 0.1)]
    assert auroc(perfect) == 1.0
    tied = [_case("a", True, 0.5), _case("b", False, 0.5)]
    assert auroc(tied) == 0.5
    gated = [_case("a", True, float("-inf")), _case("b", False, float("-inf"))]
    assert auroc(gated) == 0.5


def test_outcome_rates():
    scored = [
        _case("a", True, 0.9, hit=True),
        _case("b", True, 0.2, hit=True),  # wrongly refused at 0.5
        _case("c", True, 0.7, hit=False),  # answered without evidence
        _case("d", False, 0.1),  # correctly refused
        _case("e", False, 0.6),  # wrongly answered
    ]
    o = outcome(scored, 0.5)
    assert o["refusal_rate_unanswerable"] == 0.5
    assert o["false_refusal_rate"] == pytest.approx(1 / 3)
    assert o["answered_with_evidence"] == pytest.approx(1 / 3)
    assert o["correct_outcome_rate"] == pytest.approx(2 / 5)


def test_best_threshold_separates_classes():
    scored = [_case("a", True, 0.8), _case("b", True, 0.7), _case("c", False, 0.3), _case("d", False, 0.2)]
    t = best_threshold(scored)
    assert 0.3 < t < 0.7


def test_split_is_stratified_and_deterministic():
    scored = [_case(f"q{i}", i % 4 != 0, i / 40) for i in range(40)]
    dev, test = split(scored)
    assert split(scored) == (dev, test)
    assert len(dev) == len(test) == 20
    assert sum(not s.answerable for s in dev) == sum(not s.answerable for s in test) == 5
    report = calibrate(scored)
    assert report["n_test"] == 20 and "test" in report


def test_claim_diagnostics_apply_the_threshold(engine):
    from lumen_rag.eval.claims import diagnose_claims
    from lumen_rag.eval.harness import EvalCase

    cases = [
        EvalCase("annual paid leave days", ["leave"], "thirty working days", "direct", True),
        EvalCase("what is the capital of peru", [], None, "unanswerable", False),
    ]
    never = diagnose_claims(engine.retriever, cases, k=1)
    assert never.abstained == 0
    always = diagnose_claims(engine.retriever, cases, k=1, abstain_threshold=1.01)
    assert always.abstained == 1
    assert [row["abstained"] for row in always.per_case] == [True, True]
    assert all(row["confidence"] is not None for row in always.per_case)


def test_conservative_threshold_caps_false_refusals():
    from lumen_rag.eval.abstention import conservative_threshold

    scored = [_case(f"a{i}", True, 0.5 + i / 100) for i in range(20)] + [_case("u", False, 0.1)]
    t = conservative_threshold(scored, max_false_refusal=0.05)
    assert outcome(scored, t)["false_refusal_rate"] <= 0.05
    assert outcome(scored, t)["refusal_rate_unanswerable"] == 1.0
