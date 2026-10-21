# Evaluation and testing

The test suite, the CI gates, claim-level answer diagnostics, and how Lumen compares with the large evaluation libraries. Back to the [README](../README.md).

## 🧪 Testing

```bash
pytest -q                                                   # 138 pass offline, 13 skip without optional services
ruff check .                                                # lint
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

| Embedder | Retrieval mode | answer ok | retrieval-miss | generation-error | claim precision | faithfulness |
|---|---|---|---|---|---|---|
| hashing | vector-only | 0.73 | 0.23 | 0.05 | 0.27 | 1.00 |
| hashing | BM25 only | 0.88 | 0.09 | 0.03 | 0.36 | 1.00 |
| hashing | hybrid (BM25 + RRF) | 0.78 | 0.16 | 0.06 | 0.30 | 1.00 |
| bge-small | vector-only | **0.94** | 0.01 | 0.05 | 0.39 | 1.00 |
| bge-small | hybrid (BM25 + RRF) | 0.92 | 0.02 | 0.06 | 0.39 | 1.00 |

Reading it:

- **Most wrong answers were retrieval failures, and a real embedder removes them.** With the hashing
  embedder, 23% of questions fail because the gold document was not retrieved, against 5% where it
  was retrieved but the answer still missed it. bge-small cuts retrieval misses to 1% and lifts
  answer accuracy from 0.73 to 0.94; what is left is generation error.
- **A `generation-error` here means the answerer ignored evidence it had.** The offline answerer
  pastes only the top 2 chunks, so a gold document retrieved at rank 3 to 5 still produces a wrong
  answer.
- **`unsupported` is 0 and faithfulness is 1.00 by construction**: an extractive answerer only
  repeats retrieved text. These claim rates become informative with a generative answerer
  (`OPENAI_API_KEY`), where fabricated claims can appear.
- **Claim precision is mostly answer noise**: the share of answer sentences that come from the gold
  document rather than from the other retrieved chunks it was stitched together with.
- **The answerer itself never declines**, so without abstention 0 of 63 unanswerable questions are
  `abstained`. With `--abstain-threshold 0.664` (bge-small, vector) 17 are; see
  [Abstention](abstention.md).

Alignment is a deterministic lexical overlap (share of a claim's content words found in a chunk,
default threshold `--tau 0.6`). It cannot see paraphrase or negation, so with a generative
answerer treat the rates as an approximation. `diagnose_claims(..., aligner=fn)` accepts any
`(claim, chunk_text) -> score` function, so an NLI model or an LLM judge can replace it.

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
