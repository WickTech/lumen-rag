# Configuration

Every setting is an environment variable, read once at startup by `lumen_rag/config.py` (a `.env`
file in the working directory is loaded too; [`.env.example`](../.env.example) lists them with
comments). Everything new is **opt-in**: leaving a variable unset keeps the default behaviour.
Back to the [README](../README.md).

## Models and providers

| Variable | Default | What it does |
|---|---|---|
| `LUMEN_EMBEDDER` | `auto` | `hashing` (offline, no model), `fastembed` (local `BAAI/bge-small-en-v1.5`, needs the `embed` extra), `openai`, or `auto` (OpenAI when `OPENAI_API_KEY` is set, else hashing). Re-ingest after changing it. |
| `LUMEN_FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Model for `LUMEN_EMBEDDER=fastembed`. |
| `OPENAI_API_KEY` | — | Enables hosted embeddings (`auto`/`openai`) and LLM-generated answers. Without it answers are extractive. |
| `OPENAI_BASE_URL` | — | Any OpenAI-compatible endpoint. |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model. |
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model for generated answers. |

## Retrieval

| Variable | Default | What it does |
|---|---|---|
| `LUMEN_RERANKER` | — | Cross-encoder for second-stage reranking, e.g. `BAAI/bge-reranker-base` (`rerank` extra). |
| `LUMEN_RERANK_POOL` | `4` | Candidates fed to the reranker: `k × this`. |
| `LUMEN_CONTEXTUAL` | — | `1` embeds and BM25-indexes each chunk with a document/section header. Helps keyword search; shifts abstention scores ([benchmarks](benchmarks.md#contextual-chunk-headers)). |
| `LUMEN_DECOMPOSE` | — | `1` splits compound questions into sub-queries. Measured no gain ([benchmarks](benchmarks.md)). |
| `LUMEN_DOCLING` | — | `1` parses PDF/DOCX/PPTX/HTML with Docling (`docling` extra). |

## Abstention

Calibrated values for bge-small in vector mode are in [abstention](abstention.md); recalibrate for
another embedder or corpus with `scripts/benchmark.py --abstain`.

| Variable | Default | What it does |
|---|---|---|
| `LUMEN_ABSTAIN_THRESHOLD` | — | Refuse when retrieval confidence is below this (e.g. `0.66`). |
| `LUMEN_READER` | — | Extractive answerability reader, e.g. `onnx-community/roberta-base-squad2-ONNX` (`reader` extra). |
| `LUMEN_READER_THRESHOLD` | `0` | Refuse when the reader's answerability is below this (e.g. `-3.5`). |
| `LUMEN_READER_FILE` | `onnx/model_quantized.onnx` | ONNX weights file inside the reader repository. |
| `LUMEN_READER_ANSWERS` | — | `1` answers offline with the reader's sentence instead of the top passages. |

## Storage

| Variable | Default | What it does |
|---|---|---|
| `LUMEN_STORE` | `memory` | `memory` (files in `LUMEN_INDEX_DIR`) or `pgvector` (`pgvector` extra). |
| `LUMEN_INDEX_DIR` | `.lumen_index` | Where the in-memory store is persisted. |
| `LUMEN_PG_DSN` | — | `postgresql://…` connection string for `pgvector`. |
| `LUMEN_PG_TABLE` | `lumen_chunks` | Table for `pgvector`. |
| `LUMEN_NAMESPACE` | `default` | Isolates collections on a shared `pgvector` table. |

## Server and observability

| Variable | Default | What it does |
|---|---|---|
| `ADMIN_KEY` | — | When set, `/ingest`, `/ingest/upload`, `/ingest/sample` and `/reset` require it in the `X-Admin-Key` header (401 otherwise). Unset: open locally, refused with 403 when `SPACE_ID` is set (Hugging Face Spaces). `GET /health` reports `ingest_locked`. |
| `LUMEN_SEED_SAMPLE` | on if `SPACE_ID` is set | `1` loads the bundled sample corpus at startup when the index is empty. |
| `LUMEN_MAX_UPLOAD_BYTES` | `10485760` | Per-file cap for `/ingest/upload` (413 past it). |
| `LUMEN_MAX_UPLOAD_FILES` | `20` | Files per `/ingest/upload` request (413 past it). |
| `LUMEN_MIN_CHUNK_SIZE` / `LUMEN_MAX_CHUNK_SIZE` | `20` / `1000` | Allowed `chunk_size` (words) on the ingest routes; `overlap` must be `0..chunk_size-1` (422 otherwise). |
| `LUMEN_TRACING` | — | `console` or `otlp` OpenTelemetry export (`tracing` extra). |
| `LUMEN_TRACE_CONTENT` | — | `1` also records question and answer text on spans (off by default for privacy). |

## Optional extras

`pip install -e ".[<extra>]"`: `embed`, `reader`, `rerank`, `pdf`, `docx`, `loaders`, `docling`,
`openai`, `answer-eval`, `pgvector`, `tracing`, `dev`. None of them pulls in torch; models run on ONNX.
