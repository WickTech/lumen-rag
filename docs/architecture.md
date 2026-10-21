# Architecture

Components, data flow, storage backends, the async API and tracing. Back to the [README](../README.md).

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Ingest
        F[Files — txt/md/html/pdf/docx] --> LD[Document loaders]
        LD --> C[Sentence-aware chunker]
        C --> E1[Embedder]
        E1 --> VS[(Vector store)]
        C --> BM25[(BM25 index)]
    end
    subgraph Query
        Q[Question] --> E2[Embedder]
        E2 --> DS[Cosine search]
        VS --> DS
        Q --> SS[BM25 search]
        BM25 --> SS
        DS --> RRF[Reciprocal Rank Fusion]
        SS --> RRF
        RRF --> L[LLM answerer]
        L --> A[Answer + citations]
    end
    subgraph Evaluate
        QS[Labelled Q&A set] --> H[Eval harness]
        H --> RRF
        H --> M[recall · precision · MRR · nDCG · hit-rate]
        M --> G[Regression guard — CI fails if scores drop]
    end
```

```
lumen_rag/
├── ingestion/
│   ├── loaders.py      txt / md / html / pdf / docx → {"id","text","metadata"} (PDF title kept)
│   ├── chunker.py      sentence-aware chunking with overlap
│   ├── context.py      contextual chunk headers: "Document: <title>. Section: <headings>."
│   └── pipeline.py     docs → chunks → embeddings → store
├── store.py            Store interface + persistable in-memory cosine store (.lumen_index/)
├── pgvector_store.py   Postgres + pgvector store with namespaces
├── embeddings.py       HashingEmbedder (offline) · FastEmbedEmbedder (local bge-small) · OpenAIEmbedder
├── retrieval/
│   ├── bm25.py         BM25Index + reciprocal_rank_fusion()
│   ├── rerank.py       cross-encoder reranker (ONNX via fastembed)
│   ├── decompose.py    rule-based query decomposition (opt-in, measured no gain)
│   └── retriever.py    Retriever(mode=vector|bm25|hybrid), retrieve_with_confidence()
├── reader.py           extractive SQuAD 2.0 reader (ONNX): answerability gate
├── engine.py           RagEngine: retrieve → abstention gates → answer
├── llm.py              grounded answerer with citations; offline extractive fallback
├── eval/
│   ├── metrics.py      recall · precision · MRR · nDCG · hit-rate
│   ├── harness.py      labelled-set evaluation, per-category breakdowns
│   ├── abstention.py   confidence sweeps, AUROC, dev/test threshold calibration
│   ├── claims.py       claim-level diagnostics (retrieval vs generation failures)
│   ├── evidence.py     evidence-span grading for retrieval inside one document
│   └── answer_quality.py  DeepEval judging (needs an API key)
├── tracing.py          optional OpenTelemetry spans
├── api/                FastAPI: /ingest /ingest/upload /query /query/stream /eval /health /stats
│   └── static/         interactive demo UI, served at /
└── cli.py              lumen ingest | ask | eval | eval-answers | serve
scripts/
├── benchmark.py        v1/v2 benchmarks: --embedder --rerank --abstain --reader --contextual --decompose
├── real_pdf_check.py   end-to-end and retrieval checks on the NIST AI RMF PDF
├── check_eval.py       eval regression guard (enforces score thresholds)
├── eval_report.py      PR bot: v2 metrics for head and base, markdown diff, fails on regression
├── validate_eval_set.py  eval set consistency checks
└── loadtest.py         blocking vs non-blocking API load test
```

## ✨ Features

- **Sentence-aware chunking** with configurable size + overlap.
- **Exact cosine vector search** in a tiny, persistable store (swap for pgvector/Qdrant without touching callers).
- **BM25 + Reciprocal Rank Fusion** — true sparse/dense hybrid retrieval; choose `vector`, `bm25`, or `hybrid` mode per query.
- **Cited answers** — every response carries numbered source citations.
- **Streaming answers** — Server-Sent Events on `POST /query/stream`.
- **Multi-format ingestion** — `.txt`, `.md`, `.html`/`.htm` (no extra deps), `.pdf` (`pip install lumen-rag[pdf]`), `.docx` (`pip install lumen-rag[docx]`).
- **Interactive demo UI** — upload docs, ask questions, and watch eval metrics render live at `/` (served by the FastAPI app, no separate frontend build).
- **Pluggable embedders** — `LUMEN_EMBEDDER=fastembed` runs `BAAI/bge-small-en-v1.5` locally (ONNX, CPU, no torch, `pip install lumen-rag[embed]`); `openai` uses the hosted API; `hashing` is the dependency-free default that keeps CI offline.
- **Abstention** — set `LUMEN_ABSTAIN_THRESHOLD` and the engine refuses questions whose retrieval confidence is too low, without calling the model; responses carry `abstained` and `confidence`. Calibrated on held-out questions; see [Abstention](abstention.md).
- **Cross-encoder reranking** — optional second stage (`LUMEN_RERANKER=BAAI/bge-reranker-base`, `pip install lumen-rag[rerank]`): any mode over-fetches `k × LUMEN_RERANK_POOL` candidates and a BGE cross-encoder (ONNX via fastembed, no torch) picks the final top-k. On the bundled eval set at k=3: MRR 0.974 → 1.000, nDCG 0.968 → 0.980.
- **Layout-aware parsing** — optional Docling backend (`LUMEN_DOCLING=1`, `pip install lumen-rag[docling]`) for PDF/DOCX/PPTX/HTML: reading order, multi-column pages, and tables exported as Markdown.
- **📊 Evaluation harness** — recall@k, precision@k, MRR, nDCG@k, hit-rate over a labelled question set.
- **Claim-level diagnostics** — `lumen eval-answers --claims` labels every answer claim supported / distractor / unsupported and attributes wrong answers to retrieval or generation, offline.
- **Answer-quality eval** — `lumen eval-answers data/eval.jsonl` judges generated answers with [DeepEval](https://github.com/confident-ai/deepeval) (faithfulness, answer relevancy, contextual relevancy; plus contextual precision/recall for cases with an `expected_answer`). Needs `OPENAI_API_KEY` and `pip install lumen-rag[answer-eval]`.
- **Answerability reader** — `LUMEN_READER`: an extractive SQuAD 2.0 model checks a retrieved passage actually contains an answer and refuses near misses that similarity lets through; see [abstention](abstention.md).
- **Contextual chunk headers** — `LUMEN_CONTEXTUAL=1`: document title and section path embedded and indexed with every chunk; see [benchmarks](benchmarks.md#contextual-chunk-headers).
- **Query decomposition** — `LUMEN_DECOMPOSE=1`: splits compound questions; measured as no gain, kept opt-in.
- **Eval regression guard** — `scripts/check_eval.py` enforces minimum score thresholds in CI; the build fails if retrieval quality drops.
- **Three interfaces** — Python API, Typer CLI, and a FastAPI server.

## 🗄️ Storage backends

The engine talks to a small `Store` interface (`add`, `search`, `chunks`, `save`), so the
vector index is swappable without touching retrieval or ingestion.

| Backend | Select with | Persistence | Notes |
|---|---|---|---|
| in-memory (default) | `LUMEN_STORE=memory` | files in `LUMEN_INDEX_DIR` | exact cosine search, zero dependencies |
| Postgres + pgvector | `LUMEN_STORE=pgvector` | the database | `pip install 'lumen-rag[pgvector]'`, needs `LUMEN_PG_DSN` |

```bash
docker compose up -d db                       # Postgres 16 with pgvector (dev credentials)
pip install -e ".[pgvector]"
export LUMEN_STORE=pgvector LUMEN_PG_DSN=postgresql://lumen:lumen@localhost:5432/lumen
lumen ingest data/eval_v2/docs                # rows are committed as they are added
lumen eval data/eval_v2/eval.jsonl            # same metrics as the in-memory store
```

- **Namespaces.** `LUMEN_NAMESPACE` (default `default`) scopes every read and write. Several
  namespaces share one table and never see each other's chunks, so one deployment can serve
  several users or teams. `PgVectorStore.clear()` deletes only the current namespace.
- **Exact by default.** pgvector scans exactly, so results match the in-memory store (the test
  suite checks the v2 metrics agree within 0.005). `index="hnsw"` adds an approximate index for
  large collections; it can change results and interacts with the namespace filter.
- **Re-ingesting a chunk id replaces it** in Postgres; the in-memory store would keep both copies.
- **BM25 still runs in the app**: hybrid and BM25 modes load the namespace's chunks to build the
  sparse index, which is fine for tens of thousands of chunks. Moving BM25 into Postgres full-text
  search is the next step for larger collections.
- **Qdrant and Pinecone are not implemented.** The interface is small enough that an adapter is
  one file, but nothing beyond pgvector has been written or tested.
- The database tests (`pytest -m pgvector`) need `LUMEN_TEST_PG_DSN`; CI runs them against a
  `pgvector/pgvector:pg16` service container and fails if any are skipped.

## ⚡ Async API and tracing

**The API no longer blocks its event loop.** `/query` and `/query/stream` await the model on a shared,
connection-pooled async client, and the blocking parts (retrieval, which can be CPU work or a database /
embedding-API round trip; upload parsing; indexing) run in worker threads. Before this, `/query/stream`
iterated a *synchronous* SDK stream inside an `async def`, so with an API key one slow model call froze every
other request, `/health` included, and `answer()` built a new OpenAI client (and TLS handshake) per request.

`python scripts/loadtest.py` measures it against the real app with a **simulated** 100 ms model and offline
retrieval (so it shows how the server copes while waiting, not how fast any real model is):

| Variant | requests/s | p50 latency | p95 latency | loop lag max |
|---|---|---|---|---|
| non-blocking model call | 153.5 | 119 ms | 126 ms | 28 ms |
| blocking model call in the loop | 9.8 | 1125 ms | 2026 ms | 2033 ms |

60 requests, 20 in flight, one run on a laptop; treat the ratio as indicative. "Loop lag" is the longest
time a 10 ms heartbeat task could not run: it is what a blocking call costs every other client.

### Tracing (OpenTelemetry, optional)

```bash
pip install 'lumen-rag[tracing]'
LUMEN_TRACING=console lumen serve                      # print spans to stdout
LUMEN_TRACING=otlp OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 lumen serve
```

`otlp` speaks OTLP/HTTP, which Langfuse, Arize Phoenix, Jaeger and Grafana Tempo accept; point it at yours with the
standard `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` / `OTEL_SERVICE_NAME` variables. If your
application already installs a global OpenTelemetry provider, the spans appear without setting `LUMEN_TRACING`.
Without the packages or a provider, tracing costs a couple of function calls and records nothing.

| Span | Attributes |
|---|---|
| `lumen.query` | `lumen.mode`, `lumen.k`, `lumen.question.chars`, `lumen.chunks` |
| `lumen.retrieve` (child) | `lumen.mode`, `lumen.k`, `lumen.reranker`, `lumen.store`, `lumen.results` |
| `lumen.answer` (child) | `lumen.offline`, `lumen.context_chunks`, `lumen.answer.chars`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` |

A real run (`LUMEN_TRACING=console`, offline, BM25):

```
lumen.retrieve  parent=lumen.query  {"lumen.mode": "bm25", "lumen.k": 1, "lumen.reranker": false, "lumen.store": "VectorStore", "lumen.results": 1}
lumen.answer    parent=lumen.query  {"lumen.offline": true, "lumen.context_chunks": 1, "lumen.answer.chars": 91}
lumen.query     (root)              {"lumen.k": 1, "lumen.mode": "bm25", "lumen.question.chars": 29, "lumen.chunks": 1}
```

**Privacy:** only lengths are recorded. Question and answer text appear on spans only with `LUMEN_TRACE_CONTENT=1`,
because other people's documents flow through this engine and you may not want them in your trace backend.

What is and is not verified: the span tree, attributes, error recording, streaming spans (including a client that
disconnects mid-stream) and the OTLP wire path (spans posted as protobuf to `/v1/traces` on a local stub collector)
are covered by tests. It has **not** been run against a hosted Langfuse or Phoenix instance, and token usage is only
recorded for non-streaming answers.
