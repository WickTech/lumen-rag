# Changelog

All notable changes. Dates are 2026. Versioning follows [SemVer](https://semver.org/); the project is
pre-1.0, so minor versions may change behaviour.

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
