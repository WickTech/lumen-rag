"""Contextual chunk headers: titles, heading paths, and where the header is used."""
from lumen_rag import config
from lumen_rag.embeddings import HashingEmbedder
from lumen_rag.engine import RagEngine
from lumen_rag.ingestion.chunker import chunk_text
from lumen_rag.ingestion.context import chunk_headers, contextual_text, document_title
from lumen_rag.ingestion.pipeline import ingest_documents
from lumen_rag.retrieval.bm25 import BM25Index
from lumen_rag.store import VectorStore

PDF_LIKE = """Example Framework
Contents
1. Framing Risk 4
2. Audience 9
1. Framing Risk
Risk is the composite measure of probability and impact. Organizations differ.
1.1 Risk Tolerance
Tolerance is highly contextual. It changes with the use case.
1. Be risk-based, resource-efficient, and voluntary.
2. Audience
Many actors use the framework across the lifecycle.
Appendix A:
Descriptions of Actor Tasks
Design tasks are performed by data scientists.
"""


def test_title_from_metadata_then_markdown_heading_then_id():
    assert document_title({"id": "x", "text": "body", "metadata": {"title": "Real Title"}}) == "Real Title"
    assert document_title({"id": "x", "text": "# Pro plan\nThe Pro plan costs..."}) == "Pro plan"
    assert document_title({"id": "plan-pro", "text": "No heading here."}) == "plan pro"


def test_headers_follow_numbered_sections_and_skip_toc_and_list_items():
    doc = {"id": "rmf", "text": PDF_LIKE, "metadata": {"title": "Example Framework"}}
    chunks = chunk_text(PDF_LIKE, chunk_size=12, overlap=0)
    by_chunk = dict(zip(chunks, chunk_headers(doc, chunks), strict=True))
    header = next(h for c, h in by_chunk.items() if "highly contextual" in c)
    assert header == "Document: Example Framework. Section: 1. Framing Risk > 1.1 Risk Tolerance."
    assert by_chunk[chunks[0]] == "Document: Example Framework."  # contents page: no section yet
    actors = next(h for c, h in by_chunk.items() if "Many actors" in c)
    assert actors.endswith("Section: 2. Audience.")  # the list item "1. Be risk-based..." is not a heading
    design = next(h for c, h in by_chunk.items() if "data scientists" in c)
    assert design.endswith("Section: Appendix A: Descriptions of Actor Tasks.")


def test_markdown_title_is_not_repeated_as_a_section():
    doc = {"id": "p", "text": "# Pro plan\nCosts 7,900 euros.\n## Billing\nAnnual billing gives two months free."}
    chunks = chunk_text(doc["text"], chunk_size=4, overlap=0)
    headers = chunk_headers(doc, chunks)
    assert headers[0] == "Document: Pro plan."
    assert headers[-1] == "Document: Pro plan. Section: Billing."


def test_header_is_embedded_and_indexed_but_not_shown():
    store = VectorStore(512)
    doc = {"id": "plan-pro", "text": "# Pro plan\nThe price is 7,900 euros.\nAnnual billing gives two months free."}
    ingest_documents([doc], store=store, embedder=HashingEmbedder(), chunk_size=5, overlap=0, contextual=True)
    last = store.chunks()[-1]
    assert "Pro plan" not in last.text and last.metadata["context"] == "Document: Pro plan."
    assert contextual_text(last.text, last.metadata).startswith("Document: Pro plan.\n")
    # BM25 finds the billing chunk by the document title it never mentions.
    hits = BM25Index.build(store.chunks()).search("pro plan annual billing", 1)
    assert hits[0].chunk.id == last.id


def test_off_by_default(monkeypatch):
    eng = RagEngine()
    eng.add_documents([{"id": "d", "text": "# Title\nSome text here."}])
    assert all("context" not in c.metadata for c in eng.store.chunks())
    monkeypatch.setattr(config.settings, "contextual", True)
    eng2 = RagEngine()
    eng2.add_documents([{"id": "d", "text": "# Title\nSome text here."}])
    assert eng2.store.chunks()[0].metadata["context"] == "Document: Title."
    assert eng2.query("some text").citations[0]["context"] == "Document: Title."
