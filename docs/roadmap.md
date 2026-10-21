# Roadmap

What is done and what is next. The reasoning and status behind it is in [RAG_GAP_ANALYSIS.md](RAG_GAP_ANALYSIS.md). Back to the [README](../README.md).

## 🗺️ Roadmap

- [x] **BM25 hybrid search** — sparse + dense fusion via Reciprocal Rank Fusion; `vector | bm25 | hybrid` mode per query
- [x] **PDF / DOCX / HTML ingestion** — document loaders with clean text extraction
- [x] **Streaming answers** — server-sent events on `POST /query/stream`
- [x] **Eval regression guard** — `scripts/check_eval.py` in CI; build fails if recall drops
- [x] **PR eval report** — sticky PR comment with metric deltas on the v2 eval set; fails on a drop over 0.01
- [x] **pgvector adapter** with namespaces behind a `Store` interface (Qdrant / Pinecone still open)
- [ ] **Qdrant / Pinecone adapters** — swap the in-memory store for a production vector DB without touching engine callers
- [x] **Cross-encoder reranker** — second-stage BGE reranking via fastembed (`LUMEN_RERANKER`)
- [x] **Layout-aware ingestion** — Docling backend for PDF/DOCX/PPTX/HTML (`LUMEN_DOCLING=1`)
- [x] **Answer-quality eval** — DeepEval faithfulness / relevancy metrics (`lumen eval-answers`)
- [x] **Claim-level diagnostics** — `lumen eval-answers --claims` attributes wrong answers to retrieval vs generation, offline
- [x] **Multi-tenant namespaces** — `LUMEN_NAMESPACE` isolates collections per user/team (pgvector backend)
- [x] **Async FastAPI endpoints** — non-blocking model calls on a pooled async client, blocking work in worker threads (async DB and embedder clients are still open)
- [x] **Real embedder benchmark** — local bge-small (fastembed) next to the hashing baseline; OpenAI row pending
- [x] **Abstention** — confidence-gated refusal, calibrated on held-out questions, measured end to end
- [x] **Answerability check beyond similarity** — an extractive SQuAD 2.0 reader refuses near misses that similarity lets through
- [x] **Query decomposition** for multi-hop questions: measured, no gain with a real embedder (see [benchmarks](benchmarks.md)); kept opt-in
- [ ] **LLM query rewriting** for implicit multi-hop questions (needs an API key with credit)
- [x] **Contextual chunk headers** — offline title + section headers before embedding and BM25; helps keyword search, small effect on dense (see [benchmarks](benchmarks.md))
- [ ] **LLM-written chunk context** — Anthropic's full version, needs an API key
- [ ] **Public benchmark** — BEIR SciFact alongside the in-repo sets
- [ ] **Fine-tuned embedding adapter** — LoRA adapter trained on domain Q&A pairs to improve recall
