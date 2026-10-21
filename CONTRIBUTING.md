# Contributing to Lumen RAG

Thanks for taking a look. Lumen is small on purpose, and its one firm rule is that **every change
ships with measured numbers**, including numbers that go against the change.

## Development setup

```bash
git clone https://github.com/WickTech/lumen-rag && cd lumen-rag
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -e ".[dev]"                             # add embed, reader, rerank, pdf as needed
pytest -q && ruff check .
```

With [uv](https://docs.astral.sh/uv/): `uv run --extra dev pytest -q` and `uv run --extra dev ruff check .`.
`uv run` re-resolves `uv.lock` when you pass extras ad hoc; revert that file unless you changed
dependencies on purpose (then run `uv lock` and commit it).

## Before you open a pull request

1. `pytest -q` passes and stays offline. Tests must not need network, API keys or model downloads;
   real-model tests are skipped unless an env flag (`LUMEN_TEST_FASTEMBED`, `LUMEN_TEST_READER`) is set.
2. `ruff check .` is clean (the ruff version is pinned; CI fails on any finding).
3. If you touched retrieval, run `python scripts/benchmark.py --set v2` and put the before/after
   numbers in the PR. CI also posts the v2 metric deltas as a PR comment and fails on a drop over 0.01.
4. If you changed an eval set, run `python scripts/validate_eval_set.py`.
5. Add a line to [`CHANGELOG.md`](CHANGELOG.md).

## Conventions

- **New behaviour is opt-in** through a `LUMEN_*` setting in `lumen_rag/config.py`, documented in
  `.env.example` and [`docs/configuration.md`](docs/configuration.md). Unset keeps the old behaviour,
  and the existing tests pass unchanged.
- **Optional dependencies are lazy imports** behind an extra, with an `ImportError` that names the
  extra. No torch: models run on ONNX.
- **Thresholds are calibrated honestly**: chosen on the dev half of eval set v2 and reported on the
  held-out half (`lumen_rag/eval/abstention.py`).
- **Report what did not work.** [Query decomposition](docs/benchmarks.md) is kept in the repo as a
  documented negative result.

More detail: [evaluation and testing](docs/evaluation.md), [architecture](docs/architecture.md),
[roadmap](docs/roadmap.md).
