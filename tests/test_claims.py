"""Claim-level diagnostics: alignment, labelling, and failure attribution."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lumen_rag import config
from lumen_rag.cli import app
from lumen_rag.embeddings import get_embedder
from lumen_rag.engine import RagEngine
from lumen_rag.eval.claims import (
    content_tokens,
    diagnose_claims,
    is_abstention,
    lexical_support,
    split_claims,
)
from lumen_rag.eval.harness import EvalCase, load_cases
from lumen_rag.ingestion.loaders import load_file
from lumen_rag.ingestion.pipeline import ingest_documents
from lumen_rag.llm import Answer
from lumen_rag.retrieval import Retriever
from lumen_rag.store import Chunk, ScoredChunk, VectorStore

DATA = Path(__file__).parent.parent / "data" / "eval_v2"


def _chunk(doc_id: str, text: str) -> ScoredChunk:
    return ScoredChunk(Chunk(id=f"{doc_id}-c", text=text, doc_id=doc_id, metadata={}), score=1.0)


class FakeRetriever:
    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, question, k=5, mode="hybrid"):
        return self.chunks[:k]


GOLD = _chunk("leave", "Employees receive 30 working days of annual paid leave.")
OTHER = _chunk("sick", "Sick days are separate and there are 10 paid sick days per year.")


def _answer(text: str):
    return lambda question, chunks: Answer(text, [])


def test_split_claims_strips_markers_and_citations():
    text = "(offline extractive answer) Employees get 30 days of leave [1]. Unused days expire in March [2]. Ok."
    claims = split_claims(text)
    assert claims == ["Employees get 30 days of leave.", "Unused days expire in March."]


def test_content_tokens_fold_plurals_keep_numbers_drop_stopwords():
    assert content_tokens("The 30 working days are paid") == {"30", "working", "day", "paid"}
    assert content_tokens("") == set()


def test_lexical_support_is_the_share_of_claim_words_in_the_chunk():
    assert lexical_support("30 working days of leave", "Employees receive 30 working days of annual paid leave") == 1.0
    assert lexical_support("30 working days of leave", "Sick days are separate") == pytest.approx(0.25)
    assert lexical_support("the of and", "anything") == 0.0


def test_abstention_detects_short_declines_only():
    assert is_abstention("I don't know.")
    assert is_abstention("I don't have any indexed context to answer that.")
    long_extractive = "(offline extractive answer) " + "The Starter plan does not have webhooks. " * 10
    assert not is_abstention(long_extractive)


def test_claim_labels_supported_distractor_unsupported():
    text = (
        "Employees receive 30 working days of annual paid leave. "
        "There are 10 paid sick days per year. "
        "The company provides free lunch every Friday afternoon."
    )
    report = diagnose_claims(
        FakeRetriever([GOLD, OTHER]),
        [EvalCase("How many leave days?", ["leave"], expected_answer="30 working days", category="direct")],
        answer_fn=_answer(text),
    )
    labels = [c["label"] for c in report.per_case[0]["claims"]]
    assert labels == ["supported", "distractor", "unsupported"]
    assert report.claim_counts == {"supported": 1, "distractor": 1, "unsupported": 1}
    d = report.as_dict()
    assert d["claim_precision"] == pytest.approx(1 / 3, abs=1e-4)
    assert d["faithfulness"] == pytest.approx(2 / 3, abs=1e-4)
    assert report.per_case[0]["claims"][0]["source_doc"] == "leave"
    assert report.per_case[0]["claims"][2]["source_doc"] is None


def test_outcome_ok_retrieval_miss_and_generation_error():
    cases = [
        EvalCase("q ok", ["leave"], expected_answer="30 working days", category="direct"),
        EvalCase("q miss", ["gone"], expected_answer="forty two", category="direct"),
        EvalCase("q gen", ["leave"], expected_answer="completely different answer words", category="paraphrase"),
    ]
    report = diagnose_claims(
        FakeRetriever([GOLD, OTHER]), cases, answer_fn=_answer("Employees receive 30 working days of annual paid leave.")
    )
    outcomes = [row["outcome"] for row in report.per_case]
    # ok: expected words covered. miss: gold doc 'gone' was never retrieved.
    # gen: gold retrieved ('leave') but the answer does not contain the expected answer.
    assert outcomes == ["ok", "retrieval-miss", "generation-error"]
    assert report.outcomes == {"ok": 1, "retrieval-miss": 1, "generation-error": 1}
    d = report.as_dict()
    assert d["answer_accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert d["retrieval_miss_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert report.by_category["direct"]["n"] == 2
    assert report.by_category["paraphrase"]["generation_error"] == 1.0


def test_unanswerable_cases_are_abstained_or_hallucinated():
    cases = [EvalCase("who knows?", [], category="unanswerable", answerable=False)] * 2
    answers = iter([Answer("I don't know.", []), Answer("Employees receive 30 working days of annual paid leave.", [])])
    report = diagnose_claims(FakeRetriever([GOLD]), cases, answer_fn=lambda q, c: next(answers))
    assert [row["outcome"] for row in report.per_case] == ["abstained", "hallucinated"]
    assert report.n_cases == 0 and report.n_unanswerable == 2
    assert report.as_dict()["abstain_rate"] == 0.5
    # An unanswerable answer's claims can only be grounded in non-gold evidence.
    assert report.per_case[1]["claims"][0]["label"] == "distractor"


def test_a_custom_aligner_replaces_the_lexical_one():
    report = diagnose_claims(
        FakeRetriever([GOLD]),
        [EvalCase("q", ["leave"], expected_answer="30", category="direct")],
        answer_fn=_answer("Totally unrelated words appear here."),
        aligner=lambda claim, chunk: 1.0,
    )
    assert report.per_case[0]["claims"][0]["label"] == "supported"


@pytest.fixture(scope="module")
def v2_retriever():
    embedder = get_embedder()
    store = VectorStore(dim=embedder.dim)
    ingest_documents(
        [load_file(p) for p in sorted((DATA / "docs").glob("*.md"))],
        store=store,
        embedder=embedder,
        chunk_size=120,
        overlap=20,
    )
    return Retriever(store, embedder, reranker=None)


def test_v2_attribution_blames_retrieval_more_than_generation(v2_retriever):
    cases = load_cases(DATA / "eval.jsonl")
    vector = diagnose_claims(v2_retriever, cases, mode="vector").as_dict()
    bm25 = diagnose_claims(v2_retriever, cases, mode="bm25").as_dict()
    # The offline extractive answerer only repeats retrieved text, so it is faithful by construction.
    assert vector["unsupported_rate"] == 0.0 and vector["faithfulness"] == 1.0
    # Failures here are mostly retrieval failures, and better retrieval means more correct answers.
    assert vector["retrieval_miss_rate"] > vector["generation_error_rate"]
    assert bm25["answer_accuracy"] > vector["answer_accuracy"]
    # The extractive answerer never declines, so every unanswerable question is hallucinated.
    assert vector["abstain_rate"] == 0.0


def test_cli_claims_flag_runs_offline(tmp_path, monkeypatch, v2_retriever):
    engine = RagEngine(store=v2_retriever.store)
    index = tmp_path / "idx"
    engine.save(index)
    monkeypatch.setattr(config.settings, "index_dir", str(index), raising=False)
    out = tmp_path / "per_case.json"
    result = CliRunner().invoke(
        app, ["eval-answers", str(DATA / "eval.jsonl"), "--claims", "--mode", "bm25", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "Claim diagnostics" in result.output
    summary = json.loads(next(ln for ln in result.output.splitlines() if ln.startswith("{")))
    assert summary["mode"] == "bm25" and summary["n_cases"] == 153
    rows = json.loads(out.read_text())
    assert len(rows) == 176 and {"question", "outcome", "claims"} <= set(rows[0])
