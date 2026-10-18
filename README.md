# 🔦 Lumen RAG

> A small, **transparent, and *evaluated*** Retrieval-Augmented Generation engine. Ingest documents, retrieve with vector search, answer with citations — and **measure retrieval quality** with a proper IR metric suite.

[![CI](https://github.com/WickTech/lumen-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/WickTech/lumen-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Most RAG demos stop at "it answered my question." Lumen treats retrieval as the
engineering problem it actually is: it ships an **evaluation harness** so you can
prove that recall@5 went *up* when you change your chunking, embeddings, or
reranking — instead of vibes.

> 🔌 **Runs 100% offline.** With no API key, Lumen uses a deterministic hashing
> embedder and an extractive answerer, so the full pipeline — including the eval
> suite — runs in CI with zero secrets. Add `OPENAI_API_KEY` for real embeddings
> and generated answers.

![Lumen RAG demo: load the sample corpus, ask a question, get a cited answer, run the built-in retrieval eval](https://raw.githubusercontent.com/WickTech/lumen-rag/master/docs/demo.gif)

*The bundled demo UI (`lumen serve`, then open `/demo/`), recorded offline: no API key, no network.*

---

## ✅ Current Status

| | |
|---|---|
| **Tests** | 41/41 passing — chunker, vector store, BM25, RRF, loaders, IR metrics, end-to-end |
| **CI** | GitHub Actions: lint (ruff) → pytest → eval regression guard → v2 set validation; a separate job comments retrieval metric deltas on every PR |
| **Python** | 3.10 and 3.12 tested |
| **Offline** | Full pipeline runs with zero API keys or network access |
| **Deployment** | `Dockerfile` ready; FastAPI server on port 8000; [hosted demo on HF Spaces](https://huggingface.co/spaces/WickTech/lumen-rag) |

---

## 🎮 Live demo

**[Try it hosted on Hugging Face Spaces →](https://huggingface.co/spaces/WickTech/lumen-rag)**

Upload your own `.txt`/`.md`/`.html`/`.pdf`/`.docx` files (or load the bundled
sample corpus), ask questions against them, and watch retrieval quality
(recall@k, MRR, nDCG@k) render live from the eval harness — no signup, no API
key, runs entirely on the offline hashing embedder.

Run it locally instead:

```bash
lumen serve && open http://localhost:8000
```

---

## ✨ Features

- **Sentence-aware chunking** with configurable size + overlap.
- **Exact cosine vector search** in a tiny, persistable store (swap for pgvector/Qdrant without touching callers).
- **BM25 + Reciprocal Rank Fusion** — true sparse/dense hybrid retrieval; choose `vector`, `bm25`, or `hybrid` mode per query.
- **Cited answers** — every response carries numbered source citations.
- **Streaming answers** — Server-Sent Events on `POST /query/stream`.
- **Multi-format ingestion** — `.txt`, `.md`, `.html`/`.htm` (no extra deps), `.pdf` (`pip install lumen-rag[pdf]`), `.docx` (`pip install lumen-rag[docx]`).
- **Interactive demo UI** — upload docs, ask questions, and watch eval metrics render live at `/` (served by the FastAPI app, no separate frontend build).
- **Cross-encoder reranking** — optional second stage (`LUMEN_RERANKER=BAAI/bge-reranker-base`, `pip install lumen-rag[rerank]`): any mode over-fetches `k × LUMEN_RERANK_POOL` candidates and a BGE cross-encoder (ONNX via fastembed, no torch) picks the final top-k. On the bundled eval set at k=3: MRR 0.974 → 1.000, nDCG 0.968 → 0.980.
- **Layout-aware parsing** — optional Docling backend (`LUMEN_DOCLING=1`, `pip install lumen-rag[docling]`) for PDF/DOCX/PPTX/HTML: reading order, multi-column pages, and tables exported as Markdown.
- **📊 Evaluation harness** — recall@k, precision@k, MRR, nDCG@k, hit-rate over a labelled question set.
- **Claim-level diagnostics** — `lumen eval-answers --claims` labels every answer claim supported / distractor / unsupported and attributes wrong answers to retrieval or generation, offline.
- **Answer-quality eval** — `lumen eval-answers data/eval.jsonl` judges generated answers with [DeepEval](https://github.com/confident-ai/deepeval) (faithfulness, answer relevancy, contextual relevancy; plus contextual precision/recall for cases with an `expected_answer`). Needs `OPENAI_API_KEY` and `pip install lumen-rag[answer-eval]`.
- **Eval regression guard** — `scripts/check_eval.py` enforces minimum score thresholds in CI; the build fails if retrieval quality drops.
- **Three interfaces** — Python API, Typer CLI, and a FastAPI server.

---

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
│   ├── loaders.py   txt / md / html / pdf / docx → {"id","text","metadata"}
│   ├── chunker.py   sentence-aware chunking with overlap
│   └── pipeline.py  docs → chunks → embeddings → vector store
├── store.py         persistable cosine vector store (.lumen_index/)
├── retrieval/
│   ├── bm25.py      BM25Index + reciprocal_rank_fusion()
│   └── retriever.py Retriever(mode=vector|bm25|hybrid)
├── embeddings.py    HashingEmbedder (offline) + OpenAIEmbedder
├── llm.py           grounded answerer with citations; offline extractive fallback
├── eval/            ⭐ IR metrics + evaluation harness
├── api/             FastAPI — /ingest /ingest/upload /query /query/stream /eval /health /stats
│   └── static/      interactive demo UI, served at /
├── cli.py           lumen ingest | ask | eval | serve
└── scripts/
    ├── check_eval.py  eval regression guard (enforces score thresholds)
    ├── eval_report.py PR bot: v2 metrics for head and base, markdown diff, fails on regression
    └── validate_eval_set.py  eval set consistency checks
```

---

## 🚀 Quick start

**Install.** Not on PyPI yet: the first release is prepared (see [`docs/RELEASING.md`](https://github.com/WickTech/lumen-rag/blob/master/docs/RELEASING.md))
but unpublished, so until then install from GitHub. The `pip install lumen-rag[extra]` commands elsewhere in
this README start working once it is on PyPI; before that, add the extra to the git URL.

```bash
pip install "lumen-rag @ git+https://github.com/WickTech/lumen-rag"                    # core, runs fully offline
pip install "lumen-rag[pgvector,tracing] @ git+https://github.com/WickTech/lumen-rag"  # with extras
```

**Or work from a checkout** (needed for the bundled sample data and eval sets):

```bash
git clone https://github.com/WickTech/lumen-rag && cd lumen-rag
pip install -e ".[dev]"            # add ,openai for real embeddings

# Index the sample corpus and ask a question (works offline)
lumen ingest data/docs                                 # picks up .txt, .md, .html, .pdf, .docx
lumen ask "How many approvals does a billing change need?"
lumen ask "When do we deploy?" --mode bm25             # vector | bm25 | hybrid

# Measure retrieval quality and enforce score thresholds
python scripts/check_eval.py data/eval.jsonl --k 3 --mode hybrid
```

Example eval output:

```
  Retrieval eval — 19 cases @ k=3
  ----------------------------------
  recall@k       0.9737
  precision@k    0.3333
  mrr            0.9737
  ndcg@k         0.9677
  hit_rate       1.0000
```

### As a server

```bash
lumen serve            # or: docker build -t lumen . && docker run -p 8000:8000 lumen
curl localhost:8000/health
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"documents":[{"id":"d1","text":"We deploy at 4pm on weekdays."}]}'
curl -X POST localhost:8000/query  -H 'content-type: application/json' \
  -d '{"question":"When do we deploy?","k":3,"mode":"hybrid"}'
```

Interactive docs at `http://localhost:8000/docs`.

### As a library

```python
from lumen_rag.engine import RagEngine

engine = RagEngine()
engine.add_documents([{"id": "policy", "text": "Engineers get 20 vacation days."}])
print(engine.query("How much vacation do I get?").text)
```

---

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

---

## 🧪 Testing

```bash
pytest -q                                                   # 70 tests, all offline
ruff check lumen_rag tests                                  # lint
python scripts/check_eval.py data/eval.jsonl --k 3          # eval regression guard (v1 smoke set)
python scripts/validate_eval_set.py                         # eval set v2 consistency
python scripts/eval_report.py run --root . --out head.json  # v2 metrics, same as the PR bot
```

CI runs on Python 3.10 & 3.12: lint → pytest → `check_eval.py` (enforces recall@k ≥ 0.80, hit\_rate ≥ 0.80, mrr ≥ 0.70). The build fails if scores drop.

### PR eval report

Every pull request that touches `lumen_rag/`, `data/eval_v2/` or the report script gets a
sticky comment with the retrieval metrics (recall@5, MRR, nDCG@5, plus MRR per question
category) for vector, BM25 and hybrid search on the [v2 eval set](https://github.com/WickTech/lumen-rag/blob/master/data/eval_v2/README.md),
measured on the PR head **and** on its base commit, with the deltas. The
[`Eval report`](https://github.com/WickTech/lumen-rag/blob/master/github/workflows/eval-report.yml) job fails if any tracked metric drops
by more than 0.01. Retrieval is deterministic offline (hashing embedder, no reranker), so a
drop is a real change, not noise. If a PR changes the eval set itself the numbers are shown
but marked not comparable and never block the merge. Fork PRs cannot receive the comment
(read-only token); the report is in the job summary instead.

---

## 🧭 Where Lumen fits

Lumen is a small RAG *engine* that ships with its own evaluation. It is not a replacement for the large
evaluation libraries, and `lumen eval-answers` actually uses one of them (DeepEval) as its LLM judge.

| | Lumen RAG | [Ragas](https://github.com/explodinggradients/ragas) | [DeepEval](https://github.com/confident-ai/deepeval) |
|---|---|---|---|
| What it is | RAG engine + built-in evaluation | RAG evaluation metrics library | LLM / RAG evaluation framework, pytest-style |
| Brings its own retrieval pipeline | Yes (swappable store, retriever, answerer) | No, you supply the pipeline | No, you supply the pipeline |
| Metric breadth | Small: recall / MRR / nDCG, claim-level attribution, optional DeepEval judging | RAG metrics (faithfulness, answer relevance, context precision / recall), synthetic test data | 50+ metrics across RAG, agents, multi-turn, safety |
| Needs an LLM judge | Not for retrieval metrics or `--claims`; yes for `eval-answers` | LLM-based metrics need a judge model | Most metrics are LLM-judged |
| Works with no API key at all | Yes, by default (hashing embedder, extractive answerer) | Depends on the metric and the judge model | Depends on the metric and the judge model |
| Regression gate in CI | Threshold script + a PR comment with metric deltas, in this repo | Not a built-in feature I found | pytest integration |
| Hosted dashboard | No; exports OpenTelemetry traces instead | No | Optional (Confident AI) |

How to read it: pick Ragas or DeepEval when you already have a pipeline and want many LLM-judged metrics.
Pick Lumen when you want a pipeline you can read end to end, deterministic offline evaluation that gates a
pull request, and a way to tell *retrieval* failures from *generation* failures (see claim-level diagnostics
below). The other two columns summarise their public documentation as reported by secondary sources in
October 2026; check their docs for current details.

---

## 📊 Benchmarks — what chunking and retrieval mode actually buy you

Measured with `scripts/benchmark.py`, offline hashing embedder, no reranker, k=5.
Two eval sets ship with the repo:

- **v1** — `data/docs` (7 files) + `data/eval.jsonl` (19 questions). A fast smoke
  test, but **saturated**: k=5 returns 5 of the 7 documents, so recall is ~1.0
  for any configuration and tells you nothing.
- **v2** — `data/eval_v2` (120 short docs, 176 questions in five categories:
  direct, paraphrase, disambiguation, multihop, unanswerable). Built so that
  configurations can be told apart; see [`data/eval_v2/README.md`](https://github.com/WickTech/lumen-rag/blob/master/data/eval_v2/README.md).

Reproduce with `python scripts/benchmark.py` (v1) or `python scripts/benchmark.py --set v2`.

### v2 (discriminating)

| Configuration | recall@5 | precision@5 | MRR | nDCG@5 | hit rate |
|---|---|---|---|---|---|
| vector-only | 0.74 | 0.17 | 0.69 | 0.69 | 0.78 |
| BM25 only | **0.90** | 0.21 | **0.88** | **0.87** | **0.94** |
| hybrid (BM25 + RRF) | 0.82 | 0.19 | 0.73 | 0.74 | 0.86 |

MRR by question category (153 answerable questions; the 23 unanswerable ones are counted, not scored):

| Configuration | direct | disambiguation | multihop | paraphrase |
|---|---|---|---|---|
| vector-only | 0.84 | 0.77 | 0.67 | 0.54 |
| BM25 only | 0.98 | 0.89 | 0.86 | 0.80 |
| hybrid (BM25 + RRF) | 0.84 | 0.78 | 0.76 | 0.59 |

Paraphrased and multi-hop questions are the hard cases for the offline embedder.
Hybrid lands between its two inputs here: the hashing embedder is a weak,
bag-of-words dense signal, so fusing it with BM25 pulls the ranking *down* from
BM25 alone. That is a property of this offline embedder, not of hybrid retrieval
in general; re-run with `OPENAI_API_KEY` set to see how a real embedder changes it.

### v1 (smoke test)

| Configuration | recall@5 | precision@5 | MRR | nDCG@5 | hit rate |
|---|---|---|---|---|---|
| naive (1 chunk/doc, vector-only) | 1.00 | 0.21 | 0.96 | 0.98 | 1.00 |
| + sentence chunking (vector-only) | 0.97 | 0.20 | **1.00** | 0.98 | 1.00 |
| BM25 only (sentence chunking) | 0.97 | 0.20 | 0.96 | 0.96 | 1.00 |
| hybrid (BM25 + RRF, sentence chunking) | 0.97 | 0.20 | 0.97 | 0.97 | 1.00 |

> **Correction (2026-10-18):** earlier versions of this table (and
> [`docs/case-study.md`](https://github.com/WickTech/lumen-rag/blob/master/docs/case-study.md)) reported naive MRR 0.93 and "+ hybrid"
> equal to chunked-vector. `evaluate()` did not apply the retrieval mode, so every
> row had in fact been measured in hybrid mode and the "vector-only" labels were
> wrong. `evaluate()` now takes `mode=`, and the numbers above are the real ones.

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

---

## 🔬 Claim-level diagnostics — where do wrong answers come from?

Retrieval metrics say whether the right *documents* came back; answer judges say whether an
answer *reads* well. Neither says why an answer is wrong. `lumen eval-answers --claims`
splits every answer into claims, aligns each claim with the retrieved chunks, and attributes
failures to the retriever or the generator. It needs no API key and no judge model.

```bash
lumen ingest data/eval_v2/docs
lumen eval-answers data/eval_v2/eval.jsonl --claims --mode bm25 --out per_case.json
```

Per **claim**: `supported` (grounded in a retrieved chunk from a gold document), `distractor`
(grounded in retrieved *non-gold* text: faithful, but the wrong evidence) or `unsupported`
(not grounded in anything retrieved). Per **answerable question**: `ok`, `retrieval-miss`
(answer wrong and a gold document was not retrieved) or `generation-error` (answer wrong
although every gold document was retrieved). Per **unanswerable question**: `abstained` or
`hallucinated`.

Results on the [v2 set](https://github.com/WickTech/lumen-rag/blob/master/data/eval_v2/README.md), offline extractive answerer, k=5:

| Retrieval mode | answer ok | retrieval-miss | generation-error | claim precision | faithfulness |
|---|---|---|---|---|---|
| vector-only | 0.73 | 0.23 | 0.05 | 0.33 | 1.00 |
| BM25 only | **0.88** | 0.09 | 0.03 | 0.45 | 1.00 |
| hybrid (BM25 + RRF) | 0.78 | 0.16 | 0.06 | 0.37 | 1.00 |

Reading it:

- **Most wrong answers are retrieval failures.** Under vector-only search 23% of questions fail
  because the gold document was not retrieved, against 5% where it was retrieved but the answer
  still missed it. Better retrieval (BM25) is what lifts answer accuracy from 0.73 to 0.88.
- **A `generation-error` here means the answerer ignored evidence it had.** The offline answerer
  pastes only the top 2 chunks, so a gold document retrieved at rank 3 to 5 still produces a wrong
  answer.
- **`unsupported` is 0 and faithfulness is 1.00 by construction**: an extractive answerer only
  repeats retrieved text. These claim rates become informative with a generative answerer
  (`OPENAI_API_KEY`), where fabricated claims can appear.
- **Claim precision is mostly answer noise**: the share of answer sentences that come from the gold
  document rather than from the other retrieved chunks it was stitched together with.
- **The offline answerer never declines**, so 0 of 23 unanswerable questions are `abstained`. An
  abstain-below-a-score rule is an obvious next step, and this report will measure it.

Alignment is a deterministic lexical overlap (share of a claim's content words found in a chunk,
default threshold `--tau 0.6`). It cannot see paraphrase or negation, so with a generative
answerer treat the rates as an approximation. `diagnose_claims(..., aligner=fn)` accepts any
`(claim, chunk_text) -> score` function, so an NLI model or an LLM judge can replace it.

---

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
- [ ] **Fine-tuned embedding adapter** — LoRA adapter trained on domain Q&A pairs to improve recall

---

## 📈 What this demonstrates

- Treating retrieval quality as a **measurable, regression-tested** property.
- Clean ports-and-adapters design: embedder, store, reranker, and answerer are each swappable.
- Pragmatic offline-first engineering so the project is reproducible without secrets.
- Shipping the same core through three surfaces (library / CLI / HTTP).

## 📄 License

MIT — see [LICENSE](https://github.com/WickTech/lumen-rag/blob/master/LICENSE).
