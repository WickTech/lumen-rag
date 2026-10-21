# Changelog

All notable changes. Dates are 2026. Versioning follows [SemVer](https://semver.org/); the project is
pre-1.0, so minor versions may change behaviour.

## [Unreleased]

### Added
- **Answerability reader** (`LUMEN_READER`, `reader` extra): an extractive SQuAD 2.0 model on ONNX
  reads the retrieved passages and refuses when none contains an answer, after the similarity gate.
  Responses carry `answerability`; `LUMEN_READER_ANSWERS=1` answers with the span's sentence.
- `scripts/benchmark.py --reader` compares answerability signals, with AUROC on near-miss and
  off-topic questions separately; the 40 near-miss rows in eval set v2 are tagged `near_miss`.
- `scripts/real_pdf_check.py`: end-to-end check on the NIST AI RMF 1.0 PDF.
- `lumen eval-answers --claims` now runs the engine's own path, so both gates are measured.
- **Contextual chunk headers** (`LUMEN_CONTEXTUAL=1`, `benchmark.py --contextual`): each chunk is
  embedded and BM25-indexed with "Document: <title>. Section: <heading path>.", built offline from
  markdown, numbered and appendix headings; citations carry it as `context`. PDF loading keeps the
  PDF's title. Measured on v2 with small chunks and on a new NIST AI RMF evidence-span set
  (`data/nist_rmf`, `lumen_rag.eval.evidence`, `real_pdf_check.py --retrieval`).
- **Query decomposition** (`LUMEN_DECOMPOSE=1`, `benchmark.py --decompose`): compound questions are
  split into sub-questions whose best new documents can join the results. Measured: no gain on eval
  set v2 with bge-small, a loss in hybrid mode; off by default and documented as a negative result.
- **Pluggable embedders**: `LUMEN_EMBEDDER=auto|hashing|fastembed|openai`. `fastembed` runs
  `BAAI/bge-small-en-v1.5` locally (ONNX, CPU, no API key; `embed` extra). `auto` keeps the old behaviour.
- **Abstention**: with `LUMEN_ABSTAIN_THRESHOLD` set, the engine refuses questions whose retrieval
  confidence is below it, without calling the model. `/query`, `/query/stream` and `lumen ask` report
  `abstained` and `confidence`. Off by default.
- `lumen_rag.eval.abstention`: confidence sweep, AUROC, and threshold calibration on a held-out split.
  `lumen eval-answers --claims --abstain-threshold` measures the effect on answers.
- `scripts/benchmark.py --embedder --rerank --abstain`, with per-query latency.
- 40 near-miss unanswerable questions in eval set v2 (now 216 questions, 63 unanswerable).

### Changed
- The HF Space demo uses the bge-small embedder, dense retrieval by default, and abstention.

### Fixed
- Loading a saved index with an embedder of a different dimension now fails with a clear message.
- The test suite no longer picks up an `OPENAI_API_KEY` from a developer `.env` during collection.

## [0.1.0] - unreleased

First tagged release. The engine existed before; this release adds the evaluation and
operations work below and fixes the benchmark numbers that earlier docs reported.

### Added
- **Eval set v2** (`data/eval_v2`): 120 documents and 176 questions in five categories (direct,
  paraphrase, disambiguation, multihop, unanswerable), each answerable row with a verbatim evidence span,
  plus `lumen_rag.eval.validate` and `scripts/validate_eval_set.py`. (#3)
- **Per-category metrics and unanswerable handling** in the eval harness; `evaluate(..., mode=)`. (#3)
- **PR eval report**: a workflow that measures the v2 set on the PR head and base and comments the metric
  deltas, failing on a drop over 0.01. (#4)
- **Claim-level diagnostics**: `lumen eval-answers --claims` labels answer claims supported / distractor /
  unsupported and attributes wrong answers to retrieval or generation, offline. (#5)
- **Storage backends**: a `Store` interface, a Postgres + pgvector store with per-namespace isolation
  (`LUMEN_STORE`, `LUMEN_PG_DSN`, `LUMEN_NAMESPACE`), `docker-compose.yml`, and a CI job that runs the
  database tests against a real Postgres. (#6)
- **Non-blocking API** and pooled async model clients; `RagEngine.aquery`, `answer_async`,
  `answer_stream_async`; `scripts/loadtest.py`. (#7)
- **Optional OpenTelemetry tracing** (`LUMEN_TRACING=console|otlp`, `tracing` extra); question and answer
  text are recorded only with `LUMEN_TRACE_CONTENT=1`. (#7)
- New extras: `pgvector`, `tracing`.

### Fixed
- `evaluate()` ignored the retrieval mode, so every benchmark row ran in hybrid mode. The README and case
  study tables were wrong and have been re-measured. (#3)
- `RagEngine` replaced an empty store with an in-memory one (`store or ...` on a falsy object). (#6)
- `Retriever` read the store's private `_chunks`; it now uses the `Store` interface. (#6)
- `/query/stream` iterated a synchronous SDK stream and retrieved on the event loop; upload parsing and
  indexing ran on the loop. All now run off it. (#7)
- The OpenAI client was rebuilt on every request; it is now created once per process. (#7)
- CI never ran on pushes to `master` (the trigger listed only `main`). (#4)

### Known limits
- The offline extractive answerer never declines and is faithful by construction, so the claim rates in
  `--claims` are informative mainly with a generative answerer.
- Qdrant and Pinecone adapters are not implemented; BM25 still builds in the application.
- Tracing is verified against a stub OTLP collector, not a hosted Langfuse or Phoenix.
- The load test uses a simulated model.
