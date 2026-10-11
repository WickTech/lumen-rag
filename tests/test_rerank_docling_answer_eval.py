"""Reranker wiring, Docling loader routing, and the answer-eval harness.

All offline: the cross-encoder, Docling converter, and DeepEval judge are
replaced with fakes so these tests exercise *our* logic, not the libraries.
"""
import json
import sys
import types

import pytest

from lumen_rag import config
from lumen_rag.engine import RagEngine
from lumen_rag.eval.answer_quality import (
    AnswerEvalReport,
    AnswerSample,
    MetricResult,
    evaluate_answers,
)
from lumen_rag.eval.harness import EvalCase, load_cases
from lumen_rag.ingestion import loaders
from lumen_rag.retrieval import Retriever, rerank
from lumen_rag.store import ScoredChunk

DOCS = [
    {"id": "deploys", "text": "We deploy to production every weekday at 4pm using blue-green."},
    {"id": "oncall", "text": "On-call rotations last one week and run Monday to Monday."},
    {"id": "review", "text": "Billing changes require two approvals including a senior engineer."},
    {"id": "backups", "text": "Database backups run nightly and are kept for thirty days."},
]


class _PreferDocReranker:
    """Fake cross-encoder: puts one doc first regardless of first-stage order."""

    def __init__(self, preferred: str) -> None:
        self.preferred = preferred
        self.pool_sizes: list[int] = []

    def rerank(self, query, candidates, k):
        self.pool_sizes.append(len(candidates))
        rescored = [
            ScoredChunk(c.chunk, 1.0 if c.chunk.doc_id == self.preferred else 0.0)
            for c in candidates
        ]
        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored[:k]


# --- reranker ---------------------------------------------------------------


def test_reranker_disabled_by_default():
    assert rerank.get_reranker() is None
    engine = RagEngine()
    assert engine.retriever.reranker is None


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_reranker_reorders_overfetched_pool(mode, monkeypatch):
    monkeypatch.setattr(config.settings, "rerank_pool_factor", 4)
    engine = RagEngine()
    engine.add_documents(DOCS)
    fake = _PreferDocReranker("backups")
    retriever = Retriever(engine.store, engine.embedder, reranker=fake)

    results = retriever.retrieve("When do we deploy?", k=1, mode=mode)

    assert [r.chunk.doc_id for r in results] == ["backups"]
    # First stage was asked for k * pool_factor; the store only has 4 chunks.
    assert fake.pool_sizes == [4]


def test_explicit_none_overrides_configured_reranker(monkeypatch):
    engine = RagEngine()  # built while reranking is off
    monkeypatch.setattr(config.settings, "reranker_model", "BAAI/bge-reranker-base")
    calls = []
    monkeypatch.setattr(rerank, "_load", lambda name: calls.append(name) or "loaded")
    assert Retriever(engine.store, engine.embedder, reranker=None).reranker is None
    assert Retriever(engine.store, engine.embedder).reranker == "loaded"
    assert calls == ["BAAI/bge-reranker-base"]


def test_cross_encoder_reranker_sorts_by_model_score(monkeypatch):
    class FakeTextCrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def rerank(self, query, documents):
            return [float("approvals" in d) for d in documents]

    mod = types.ModuleType("fastembed.rerank.cross_encoder")
    mod.TextCrossEncoder = FakeTextCrossEncoder
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", mod)

    engine = RagEngine()
    engine.add_documents(DOCS)
    pool = [ScoredChunk(c, 0.5) for c in engine.store._chunks]
    out = rerank.CrossEncoderReranker("fake").rerank("approvals?", pool, k=2)

    assert out[0].chunk.doc_id == "review"
    assert out[0].score == 1.0
    assert len(out) == 2
    assert rerank.CrossEncoderReranker("fake").rerank("q", [], k=3) == []


# --- docling ----------------------------------------------------------------


def _install_fake_docling(monkeypatch, markdown: str):
    converted = []

    class FakeConverter:
        def convert(self, source):
            converted.append(source)
            doc = types.SimpleNamespace(export_to_markdown=lambda: markdown)
            return types.SimpleNamespace(document=doc)

    mod = types.ModuleType("docling.document_converter")
    mod.DocumentConverter = FakeConverter
    monkeypatch.setitem(sys.modules, "docling", types.ModuleType("docling"))
    monkeypatch.setitem(sys.modules, "docling.document_converter", mod)
    loaders._docling_converter.cache_clear()
    return converted


def test_docling_routes_supported_types_when_enabled(tmp_path, monkeypatch):
    converted = _install_fake_docling(monkeypatch, "| col |\n|---|\n| cell |")
    monkeypatch.setattr(config.settings, "use_docling", True)
    page = tmp_path / "report.html"
    page.write_text("<p>ignored by fake</p>", encoding="utf-8")

    doc = loaders.load_file(page)

    assert doc["id"] == "report"
    assert doc["text"].startswith("| col |")
    assert converted == [str(page)]
    loaders._docling_converter.cache_clear()


def test_docling_not_used_for_plain_text_or_when_disabled(tmp_path, monkeypatch):
    converted = _install_fake_docling(monkeypatch, "from docling")
    note = tmp_path / "note.md"
    note.write_text("plain markdown", encoding="utf-8")
    page = tmp_path / "page.html"
    page.write_text("<p>stdlib parser</p>", encoding="utf-8")

    monkeypatch.setattr(config.settings, "use_docling", True)
    assert loaders.load_file(note)["text"] == "plain markdown"
    monkeypatch.setattr(config.settings, "use_docling", False)
    assert loaders.load_file(page)["text"] == "stdlib parser"
    assert converted == []
    loaders._docling_converter.cache_clear()


def test_pptx_without_docling_gives_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "docling", None)  # make the import fail
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    loaders._docling_converter.cache_clear()
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(b"not really a pptx")
    with pytest.raises(ImportError, match=r"lumen-rag\[docling\]"):
        loaders.load_file(deck)
    loaders._docling_converter.cache_clear()


# --- answer eval ------------------------------------------------------------


class _FakeScorer:
    def __init__(self):
        self.samples: list[AnswerSample] = []

    def score(self, sample):
        self.samples.append(sample)
        out = {
            "faithfulness": MetricResult(0.9, "grounded"),
            "answer_relevancy": MetricResult(0.4 if "on-call" in sample.question else 0.8),
        }
        if sample.expected_answer:
            out["contextual_recall"] = MetricResult(0.0, error="RateLimitError: slow down")
        return out


def test_evaluate_answers_aggregates_and_isolates_errors():
    engine = RagEngine()
    engine.add_documents(DOCS)
    cases = [
        EvalCase("When do we deploy to production?", ["deploys"]),
        EvalCase("How long is an on-call rotation?", ["oncall"], expected_answer="One week."),
    ]
    scorer = _FakeScorer()

    report = evaluate_answers(engine, cases, k=2, scorer=scorer, threshold=0.5)

    assert isinstance(report, AnswerEvalReport)
    assert report.n_cases == 2
    assert report.means["faithfulness"] == pytest.approx(0.9)
    assert report.means["answer_relevancy"] == pytest.approx(0.6)
    assert report.pass_rates["answer_relevancy"] == pytest.approx(0.5)
    # The errored metric is reported but excluded from aggregates.
    assert report.errors == 1
    assert "contextual_recall" not in report.means
    assert report.per_case[1]["metrics"]["contextual_recall"]["error"].startswith("RateLimit")
    # The scorer saw real retrieved context and the reference answer.
    assert scorer.samples[0].retrieval_context
    assert scorer.samples[1].expected_answer == "One week."
    json.dumps(report.as_dict())  # serialisable for the CLI
    assert "faithfulness" in report.pretty()


def test_evaluate_answers_refuses_offline_without_scorer():
    engine = RagEngine()
    engine.add_documents(DOCS)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        evaluate_answers(engine, [EvalCase("q", [])])


def test_load_cases_reads_optional_expected_answer(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"question": "a?", "relevant_doc_ids": ["x"]}\n'
        '{"question": "b?", "relevant_doc_ids": [], "expected_answer": "B."}\n',
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert cases[0].expected_answer is None
    assert cases[1].expected_answer == "B."
