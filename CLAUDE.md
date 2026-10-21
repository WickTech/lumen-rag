# Lumen RAG: notes for working in this repo

A small RAG engine with built-in evaluation. `README.md` is a short overview; detail lives in
`docs/` (benchmarks, abstention, evaluation, architecture, configuration, roadmap) and
`CONTRIBUTING.md`. `docs/RAG_GAP_ANALYSIS.md` tracks the roadmap and its status.

## Commands

- Tests: `uv run --extra dev pytest -q` (offline; skips need Postgres, OpenTelemetry or model downloads).
- Lint: `uv run --extra dev ruff check .` (ruff is pinned; CI fails on any finding).
- `uv run` rewrites `uv.lock` when extras are resolved ad hoc: `git checkout uv.lock` afterwards unless
  dependencies changed on purpose (then `uv lock` and commit it).
- Benchmarks:
  - `python scripts/benchmark.py --set v2 --abstain` (hashing, what CI uses)
  - `uv run --extra embed --extra rerank --extra reader python scripts/benchmark.py --set v2 --embedder fastembed --rerank --abstain --reader onnx-community/roberta-base-squad2-ONNX` (~15 min on CPU)
  - Results are committed as `benchmark_results*.json`; the tables in `docs/benchmarks.md` and
    `docs/abstention.md` are copied from the printed output (the README only has a summary table).
- Real-document check: `python scripts/real_pdf_check.py` (NIST AI RMF PDF; needs embed, reader, pdf extras).
- Eval set consistency: `python scripts/validate_eval_set.py`.

## Conventions

- Every change ships with measured before/after numbers in the README, including results that
  go against the change. Thresholds are chosen on the dev half of eval set v2 and reported on the
  held-out half (`lumen_rag/eval/abstention.py`).
- New behaviour is opt-in through `LUMEN_*` settings in `lumen_rag/config.py`, documented in
  `.env.example`; unset keeps the old behaviour and the existing tests pass unchanged.
- Optional dependencies are lazy imports behind extras (`embed`, `rerank`, `reader`, `pdf`, ...),
  with an ImportError naming the extra. No torch: models run on ONNX.
- Tests stay offline and hermetic: `tests/conftest.py` resets settings at import time, because
  `lumen_rag.api.app` builds an engine during collection. Real-model tests are skipped unless an
  env flag (`LUMEN_TEST_FASTEMBED`, `LUMEN_TEST_READER`) is set.

## Gotchas

- A `.env` in the parent checkout (`/home/rishi/workspace/lumen-rag/.env`) is found by
  `load_dotenv()`; it holds an OpenAI key with no credit, so `--embedder openai` fails with 429.
- An expired local HF login makes public model downloads fail with 401: set
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.
- HF Space (`deploy/hf-space/`): ZeroGPU only accepts apps launched through Gradio's
  `launch()`; `gradio_app.py` passes the FastAPI routes in via `app_kwargs`. Deploy with
  `deploy/hf-space/push.sh` (ships `git archive HEAD`, so deploy from the commit you mean),
  then check `hf spaces info WickTech/lumen-rag --expand runtime` and smoke-test the live URL.
