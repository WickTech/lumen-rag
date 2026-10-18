"""The v2 eval set must stay consistent and keep discriminating between configs."""
import json
from pathlib import Path

import pytest

from lumen_rag.embeddings import get_embedder
from lumen_rag.eval import evaluate
from lumen_rag.eval.harness import EvalCase, load_cases
from lumen_rag.eval.validate import validate_eval_set
from lumen_rag.ingestion.loaders import load_file
from lumen_rag.ingestion.pipeline import ingest_documents
from lumen_rag.retrieval import Retriever
from lumen_rag.store import VectorStore

DATA = Path(__file__).parent.parent / "data" / "eval_v2"
EVAL = DATA / "eval.jsonl"
DOCS = DATA / "docs"


@pytest.fixture(scope="module")
def retriever():
    embedder = get_embedder()
    store = VectorStore(dim=embedder.dim)
    docs = [load_file(p) for p in sorted(DOCS.glob("*.md"))]
    ingest_documents(docs, store=store, embedder=embedder, chunk_size=120, overlap=20)
    return Retriever(store, embedder, reranker=None)


def test_shipped_set_is_consistent():
    errors, stats = validate_eval_set(EVAL, DOCS)
    assert errors == []
    assert stats["n_docs"] >= 100
    assert stats["n_cases"] >= 150
    for category in ("direct", "paraphrase", "disambiguation", "multihop", "unanswerable"):
        assert stats["categories"].get(category, 0) >= 20, category


def _write(tmp_path, rows, docs):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for name, text in docs.items():
        (docs_dir / f"{name}.md").write_text(text, encoding="utf-8")
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path, docs_dir


def _row(**overrides):
    row = {
        "question": "How long?",
        "relevant_doc_ids": ["a"],
        "expected_answer": "Five days.",
        "category": "direct",
        "answerable": True,
        "evidence": [{"doc_id": "a", "span": "five days"}],
    }
    row.update(overrides)
    return row


def test_validator_accepts_a_good_set(tmp_path):
    path, docs = _write(tmp_path, [_row()], {"a": "# A\nIt lasts five days."})
    errors, _ = validate_eval_set(path, docs)
    assert errors == []


def test_validator_catches_each_kind_of_mistake(tmp_path):
    rows = [
        _row(question="unknown doc", relevant_doc_ids=["zzz"], evidence=[{"doc_id": "zzz", "span": "x"}]),
        _row(question="bad span", evidence=[{"doc_id": "a", "span": "ten days"}]),
        _row(question="dup"),
        _row(question="dup"),
        _row(question="labelled unanswerable", category="unanswerable", answerable=False),
        _row(question="bad category", category="trivia"),
        _row(question="multihop with one doc", category="multihop"),
    ]
    path, docs = _write(tmp_path, rows, {"a": "# A\nIt lasts five days."})
    errors, _ = validate_eval_set(path, docs)
    text = "\n".join(errors)
    assert "not found in" in text
    assert "not found verbatim" in text
    assert "duplicate question" in text
    assert "unanswerable case must have no ids" in text
    assert "unknown category" in text
    assert "multihop case needs at least 2" in text


def test_load_cases_reads_category_and_answerable(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text(
        json.dumps({"question": "q", "relevant_doc_ids": ["a"], "category": "paraphrase"})
        + "\n"
        + json.dumps({"question": "u", "relevant_doc_ids": [], "answerable": False})
        + "\n",
        encoding="utf-8",
    )
    first, second = load_cases(path)
    assert (first.category, first.answerable) == ("paraphrase", True)
    assert second.answerable is False and second.category is None


def test_unanswerable_cases_are_counted_not_scored(retriever):
    cases = [
        EvalCase("How many days of annual paid leave do EU employees get?", ["vacation-eu"], category="direct"),
        EvalCase("Who won the 1998 World Cup?", [], category="unanswerable", answerable=False),
    ]
    report = evaluate(retriever, cases, k=5)
    assert report.n_cases == 1
    assert report.n_unanswerable == 1
    assert set(report.by_category) == {"direct"}
    assert report.hit_rate == 1.0


def test_evaluate_applies_the_retrieval_mode(retriever):
    cases = [c for c in load_cases(EVAL) if c.answerable]
    vector = evaluate(retriever, cases, k=5, mode="vector")
    bm25 = evaluate(retriever, cases, k=5, mode="bm25")
    # If the mode were ignored these would be identical.
    assert vector.as_dict() != bm25.as_dict()


def test_v2_discriminates_between_configurations(retriever):
    """M0 acceptance: configurations must be clearly separable on the v2 set."""
    cases = [c for c in load_cases(EVAL) if c.answerable]
    mrr = {m: evaluate(retriever, cases, k=5, mode=m).mrr for m in ("vector", "bm25", "hybrid")}
    spread = max(mrr.values()) - min(mrr.values())
    assert spread >= 0.1, mrr


def test_hard_categories_are_harder_than_direct(retriever):
    cases = [c for c in load_cases(EVAL) if c.answerable]
    report = evaluate(retriever, cases, k=5, mode="vector")
    by = report.by_category
    assert by["direct"]["mrr"] > by["paraphrase"]["mrr"]
    assert by["direct"]["mrr"] > by["multihop"]["mrr"]
