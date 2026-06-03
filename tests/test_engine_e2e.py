"""End-to-end smoke test of the full RAG loop, running fully offline."""
from lumen_rag.engine import RagEngine
from lumen_rag.eval import evaluate
from lumen_rag.eval.harness import EvalCase
from lumen_rag.retrieval import Retriever


DOCS = [
    {"id": "deploys", "text": "We deploy to production every weekday at 4pm using blue-green."},
    {"id": "oncall", "text": "On-call rotations last one week and run Monday to Monday."},
    {"id": "review", "text": "Billing changes require two approvals including a senior engineer."},
]


def test_full_loop_retrieves_and_answers():
    engine = RagEngine()
    engine.add_documents(DOCS)
    result = engine.query("When do we deploy?", k=2)
    assert result.citations
    assert result.citations[0]["doc_id"] == "deploys"


def test_eval_harness_scores_relevant_docs():
    engine = RagEngine()
    engine.add_documents(DOCS)
    cases = [
        EvalCase("When do we deploy to production?", ["deploys"]),
        EvalCase("How long is an on-call rotation?", ["oncall"]),
    ]
    report = evaluate(Retriever(engine.store, engine.embedder), cases, k=2)
    assert report.n_cases == 2
    assert report.hit_rate > 0.0
