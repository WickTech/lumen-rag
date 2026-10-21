# 🔦 Lumen RAG

> A small, transparent and **evaluated** Retrieval-Augmented Generation engine: ingest documents, retrieve, answer with citations, refuse what the documents don't say, and **measure** every step.

[![CI](https://github.com/WickTech/lumen-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/WickTech/lumen-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Spaces-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/WickTech/lumen-rag)

![Lumen RAG demo: load the sample corpus, ask a question, get a cited answer, run the built-in retrieval eval](https://raw.githubusercontent.com/WickTech/lumen-rag/master/docs/demo.gif)

*The demo UI, as served by the [hosted demo](https://huggingface.co/spaces/WickTech/lumen-rag) and by `lumen serve`. Recorded on the offline hashing embedder; the hosted demo runs bge-small and also shows retrieval confidence and refusals.*

## Table of contents

- [About the project](#about-the-project)
- [Results at a glance](#results-at-a-glance)
- [Features](#features)
- [Installation and running](#installation-and-running)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

## About the project

**What.** Lumen is a RAG engine you can read end to end: loaders for txt, md, html, pdf and docx,
sentence-aware chunking, dense, BM25 or hybrid retrieval, optional cross-encoder reranking, cited
answers (extractive offline, or generated with an OpenAI-compatible model), and two gates that
refuse questions the documents cannot answer. It ships as a Python library, a CLI, a FastAPI server
with a demo UI, and a [hosted demo](https://huggingface.co/spaces/WickTech/lumen-rag).

**Why.** Most RAG demos stop at "it answered my question." Lumen treats retrieval as an engineering
problem with numbers attached: a labelled evaluation set, a CI gate that fails a pull request when
retrieval gets worse, and published before/after results for every change, including the changes
that did not help.

**How.** Everything runs on CPU with no API key. Embeddings, reranking and the answerability reader
are ONNX models (no torch), and the test suite uses a deterministic hashing embedder so CI needs no
secrets or network. Thresholds are chosen on one half of the eval set and reported on the other, so
the published numbers are not tuned on their own test data.

## Results at a glance

On eval set v2 (120 documents, 216 questions, 63 of them unanswerable); full tables in
[benchmarks](docs/benchmarks.md) and [abstention](docs/abstention.md).

| Change | Before | After |
|---|---|---|
| Real embedder (bge-small vs hashing), vector MRR | 0.69 | **0.95** |
| + MiniLM cross-encoder rerank, MRR | 0.95 | **0.97** (≈ +470 ms/query) |
| Abstention: unanswerable questions refused at 3% false refusals | 0% | 26% (similarity) → **42%** (reader) |
| NIST AI RMF PDF, right outcome on 10 questions | 6/10 | **10/10** (both gates) |
| Contextual headers, BM25 MRR with small chunks | 0.81 | **0.85** |
| Query decomposition, multi-hop recall@5 | 0.90 | 0.90 (no gain, documented) |

## Features

- **Retrieval:** dense, BM25 or hybrid (Reciprocal Rank Fusion) per query; optional cross-encoder reranking.
- **Embedders:** local `BAAI/bge-small-en-v1.5`, OpenAI, or an offline hashing embedder for tests.
- **Abstention:** a similarity gate, then an extractive SQuAD 2.0 reader that catches near misses; every answer reports its scores.
- **Ingestion:** txt, md, html, pdf, docx (optional Docling for layout); optional contextual chunk headers.
- **Evaluation:** recall, precision, MRR, nDCG, hit-rate; per-category breakdowns; claim-level diagnostics that blame retrieval or generation; DeepEval judging with an API key.
- **Quality gates:** CI fails on an eval regression and comments metric deltas on every pull request.
- **Serving:** FastAPI with streaming (SSE), uploads, rate limits and an admin-gated reset; demo UI at `/`; Docker; Postgres + pgvector with namespaces; OpenTelemetry tracing.

Architecture and the full feature list: [docs/architecture.md](docs/architecture.md).

## Installation and running

**Prerequisites:** Python 3.10 or newer and git. Docker is optional (pgvector, container runs).

1. **Clone the repository**

   ```bash
   git clone https://github.com/WickTech/lumen-rag && cd lumen-rag
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

3. **Install the package with the extras you need**

   ```bash
   pip install -e ".[dev]"                  # core + test tools; runs fully offline
   pip install -e ".[dev,embed,reader,pdf]" # + local embeddings, answerability reader, PDF loading
   ```

   Other extras: `rerank`, `docx`, `docling`, `openai`, `answer-eval`, `pgvector`, `tracing`
   (see [configuration](docs/configuration.md)). The first use of `embed`, `reader` or `rerank`
   downloads the ONNX model from Hugging Face once.

4. **Check that everything works**

   ```bash
   pytest -q            # offline; some tests skip without Postgres or model downloads
   ```

5. **Run it**

   ```bash
   lumen serve          # demo UI at http://localhost:8000, API docs at /docs
   ```

   Or with Docker: `docker build -t lumen . && docker run -p 8000:8000 lumen`.

Not on PyPI yet; to install without cloning, use
`pip install "lumen-rag[embed] @ git+https://github.com/WickTech/lumen-rag"`.

## Usage

**Command line.** Index a folder, then ask questions:

```bash
lumen ingest data/docs
lumen ask "How many approvals does a billing change need?"
lumen ask "When do we deploy?" --mode bm25          # vector | bm25 | hybrid
lumen eval data/eval.jsonl --k 3                    # retrieval metrics on a labelled set
```

**Real embeddings and refusals**, all local:

```bash
export LUMEN_EMBEDDER=fastembed LUMEN_ABSTAIN_THRESHOLD=0.66 \
       LUMEN_READER=onnx-community/roberta-base-squad2-ONNX LUMEN_READER_THRESHOLD=-3.5
lumen ingest data/docs
lumen ask "What is our parental leave policy in Brazil?" --mode vector
# I can't find this in the indexed documents, so I won't guess.
# Retrieval confidence: 0.608 (abstained)
```

**HTTP API.**

```bash
lumen serve
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"documents":[{"id":"d1","text":"We deploy at 4pm on weekdays."}]}'
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"When do we deploy?","k":3,"mode":"hybrid"}'
```

Responses carry the answer, numbered citations, and `abstained`, `confidence` and `answerability`
when the gates are enabled. `POST /query/stream` streams the answer; `POST /ingest/upload` takes files.

**Python.**

```python
from lumen_rag.engine import RagEngine

engine = RagEngine()
engine.add_documents([{"id": "policy", "text": "Engineers get 20 vacation days."}])
answer = engine.query("How much vacation do I get?")
print(answer.text, answer.citations)
```

**Measure a change.**

```bash
python scripts/benchmark.py --set v2                               # offline baseline
python scripts/benchmark.py --set v2 --embedder fastembed --abstain # real embedder + abstention
python scripts/real_pdf_check.py                                   # end-to-end on the NIST AI RMF PDF
```

All settings are environment variables; see [docs/configuration.md](docs/configuration.md).

## Documentation

| Document | Contents |
|---|---|
| [Benchmarks](docs/benchmarks.md) | Embedders, retrieval modes, reranking, contextual headers, query decomposition |
| [Abstention](docs/abstention.md) | Similarity gate, answerability reader, calibration, NIST PDF check |
| [Evaluation and testing](docs/evaluation.md) | Test suite, CI gates, PR eval report, claim-level diagnostics, comparison with Ragas and DeepEval |
| [Architecture](docs/architecture.md) | Data flow, code layout, storage backends, async API, tracing |
| [Configuration](docs/configuration.md) | Every environment variable and extra |
| [Roadmap](docs/roadmap.md) and [gap analysis](docs/RAG_GAP_ANALYSIS.md) | What is done, what is next, and why |
| [Case study](docs/case-study.md) | The engineering story behind the numbers |
| [Changelog](CHANGELOG.md) and [releasing](docs/RELEASING.md) | Release history and process |

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first: changes ship
with tests that run offline, and with before/after numbers when they touch retrieval.

## Credits

Built by [Rishabh Verma](https://github.com/WickTech). Lumen stands on:

- **Models:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (embeddings),
  [deepset roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2) via its
  [ONNX export](https://huggingface.co/onnx-community/roberta-base-squad2-ONNX) (answerability reader),
  [ms-marco-MiniLM-L-6-v2](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2) and BAAI bge-reranker (reranking).
- **Libraries:** [fastembed](https://github.com/qdrant/fastembed), [ONNX Runtime](https://onnxruntime.ai/),
  [FastAPI](https://fastapi.tiangolo.com/), [Gradio](https://www.gradio.app/), [pgvector](https://github.com/pgvector/pgvector),
  [pypdf](https://github.com/py-pdf/pypdf), [Docling](https://github.com/docling-project/docling),
  [DeepEval](https://github.com/confident-ai/deepeval), [OpenTelemetry](https://opentelemetry.io/).
- **Ideas and data:** Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval);
  Barnett et al., [Seven Failure Points When Engineering a RAG System](https://arxiv.org/abs/2401.05856);
  the [NIST AI Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1) (public domain) as a real test document.

## License

MIT; see [LICENSE](LICENSE).
