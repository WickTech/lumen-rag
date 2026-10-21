"""Rule-based query decomposition and how the retriever folds sub-questions in."""
from lumen_rag.embeddings import HashingEmbedder
from lumen_rag.ingestion.pipeline import ingest_documents
from lumen_rag.retrieval import Retriever
from lumen_rag.retrieval.decompose import split_question
from lumen_rag.store import VectorStore


def test_splits_compound_question_at_the_second_question_word():
    q = "How much can I claim per day for meals on a trip abroad, and how soon must I submit the claim?"
    assert split_question(q) == [
        "How much can I claim per day for meals on a trip abroad",
        "how soon must I submit the claim",
    ]


def test_single_questions_and_short_fragments_stay_whole():
    assert split_question("What does TEVV stand for?") == ["What does TEVV stand for?"]
    # "and how?" is too short to search with, so nothing is split off.
    assert split_question("Is the grant taxable, and how?") == ["Is the grant taxable, and how?"]


DOCS = [
    {"id": "meals", "text": "Meals on trips abroad are paid at a per diem of 65 euros a day."},
    {"id": "deadline", "text": "Expense claims must be submitted within 30 days of the trip ending."},
    {"id": "parking", "text": "Parking spaces are allocated by lottery every quarter."},
]


def _retriever(decompose):
    emb = HashingEmbedder()
    store = VectorStore(emb.dim)
    ingest_documents(DOCS, store=store, embedder=emb, chunk_size=50, overlap=0)
    return Retriever(store, emb, reranker=None, decompose=decompose)


def test_off_by_default_and_never_changes_single_questions():
    q = "What is the per diem for meals abroad?"
    assert [r.chunk.doc_id for r in _retriever(None).retrieve(q, k=2, mode="vector")] == [
        r.chunk.doc_id for r in _retriever(True).retrieve(q, k=2, mode="vector")
    ]


def test_sub_question_hit_replaces_only_a_weaker_tail_result():
    q = "What is the per diem for meals abroad, and when must expense claims be submitted?"
    base = _retriever(False).retrieve(q, k=2, mode="vector")
    dec = _retriever(True).retrieve(q, k=2, mode="vector")
    assert dec[0].chunk.doc_id == base[0].chunk.doc_id  # the whole question's best hit stays first
    assert {r.chunk.doc_id for r in dec} == {"meals", "deadline"}
    assert len(dec) == 2
