# Case study: what chunking and hybrid retrieval actually buy you

Lumen ships an eval harness (`lumen_rag/eval/`) specifically so claims like
"chunking helps" or "hybrid retrieval helps" don't have to stay vibes. This
is the measurement behind the numbers in the README.

## Setup

- **Corpus**: `data/docs/` — 7 markdown documents (~2,250 words total), each
  covering multiple unrelated handbook topics per file (e.g. `deploys-ops.md`
  bundles deployments, on-call, incident response, disaster recovery, plus
  three unrelated filler sections). This mirrors a real internal wiki page:
  long, multi-topic, with the answer to any given question living in one
  paragraph out of several.
- **Eval set**: `data/eval.jsonl` — 19 labelled questions, each with the
  correct source document id(s).
- **Embedder**: the offline deterministic `HashingEmbedder` (bag-of-words
  feature hashing, L2-normalised) — no API key, fully reproducible, what CI
  and the hosted demo run by default.
- **Metric**: `lumen_rag.eval.harness.evaluate`, doc-level (chunk hits are
  collapsed to unique parent documents before scoring), k=5.
- **Reproduce**: `python scripts/benchmark.py`

## Results

> **Correction (2026-10-18).** The first version of this case study reported naive
> MRR 0.93 and hybrid equal to chunked-vector. `evaluate()` ignored the retrieval
> mode, so all three rows were actually hybrid-mode runs. The harness now applies
> `mode=`, and the tables below are re-measured. The v1 set is also tiny and
> saturated (7 documents, k=5), so the conclusions that matter come from the v2 set.

### v1 smoke test (7 docs, 19 questions)

| Configuration | recall@5 | precision@5 | MRR | nDCG@5 | hit rate |
|---|---|---|---|---|---|
| naive, 1 chunk per doc, vector-only | 1.00 | 0.21 | 0.96 | 0.98 | 1.00 |
| + sentence-aware chunking, vector-only | 0.97 | 0.20 | **1.00** | 0.98 | 1.00 |
| BM25 only, sentence chunking | 0.97 | 0.20 | 0.96 | 0.96 | 1.00 |
| hybrid (BM25 + RRF), sentence chunking | 0.97 | 0.20 | 0.97 | 0.97 | 1.00 |

With 7 candidate documents and k=5, recall cannot discriminate: any config
returns most of the corpus. Chunking raises MRR from 0.96 to 1.00 (the right
document is always first) while recall dips from 1.00 to 0.97 (one question loses
its second relevant chunk's parent from the top 5). Treat this set as a smoke test.

### v2 (120 docs, 216 questions, `data/eval_v2`), hashing embedder

| Configuration | recall@5 | precision@5 | MRR | nDCG@5 | hit rate |
|---|---|---|---|---|---|
| vector-only | 0.74 | 0.17 | 0.69 | 0.69 | 0.78 |
| BM25 only | 0.90 | 0.21 | 0.88 | 0.87 | 0.94 |
| hybrid (BM25 + RRF) | 0.82 | 0.19 | 0.73 | 0.74 | 0.86 |

| MRR by category | direct | disambiguation | multihop | paraphrase |
|---|---|---|---|---|
| vector-only | 0.84 | 0.77 | 0.67 | 0.54 |
| BM25 only | 0.98 | 0.89 | 0.86 | 0.80 |
| hybrid (BM25 + RRF) | 0.84 | 0.78 | 0.76 | 0.59 |

### v2 with a real embedder (`BAAI/bge-small-en-v1.5`, local CPU)

| Configuration | recall@5 | precision@5 | MRR | nDCG@5 | hit rate |
|---|---|---|---|---|---|
| vector-only | 0.97 | 0.23 | 0.95 | 0.94 | 0.99 |
| BM25 only | 0.90 | 0.21 | 0.88 | 0.87 | 0.94 |
| hybrid (BM25 + RRF) | 0.97 | 0.23 | 0.91 | 0.91 | 0.99 |
| vector-only + MiniLM rerank | 0.98 | 0.23 | 0.97 | 0.96 | 0.99 |

| MRR by category | direct | disambiguation | multihop | paraphrase |
|---|---|---|---|---|
| vector-only | 0.98 | 0.96 | 0.93 | 0.93 |
| hybrid (BM25 + RRF) | 0.99 | 0.87 | 0.95 | 0.86 |

## Reading the numbers honestly

**The v2 set separates configurations; v1 does not.** On v2 the best and worst
configurations differ by 0.19 MRR, and paraphrased and multi-hop questions are
clearly harder than direct ones, which is what a retrieval benchmark should show.

**Hybrid sits between its inputs because the dense half is weak.** The offline
`HashingEmbedder` is a bag-of-words feature hasher, so its ranking is a noisy
version of what BM25 already computes. Reciprocal Rank Fusion then mixes a good
ranking with a worse one and the result is worse than BM25 alone. This was a
statement about the offline embedder, and the real-embedder run confirms it: with
bge-small the order flips. Dense search alone is best (MRR 0.95), and fusing in
BM25 now lowers it to 0.91, mostly on disambiguation (0.96 → 0.87), where
keyword matches promote the near-duplicate distractor documents. Fusion weights
are a per-corpus decision, not a default to trust.

**Chunking is not exercised by v2.** Its documents are 60 to 110 words, shorter
than the 120-word chunk window, so chunked and unchunked runs are identical. The
chunking effect needs longer documents.

**The cross-encoder reranker is not in these tables.** It needs a model download,
so the benchmark keeps it off to stay reproducible offline.

## Takeaways for anyone building on Lumen

1. **Don't just report recall/hit-rate.** They saturate fast on small
   corpora (v1: 7 documents, k=5) and hide ranking differences. Check that your
   eval set can actually tell two configurations apart before trusting it; v2
   exists for that reason.
2. **Check that your benchmark applies the settings it claims to.** The first
   version of this study labelled rows "vector-only" while running hybrid. A test
   (`test_evaluate_applies_the_retrieval_mode`) now guards it.
3. **Chunking's benefit scales with document length and topic density**, not
   corpus size. A corpus of short, single-topic documents (like v2) won't show
   this effect; you need documents where the answer is a minority of the content.
4. **Hybrid retrieval's payoff is embedder-dependent.** With the offline hashing
   embedder, hybrid scored below BM25 alone on v2. Re-run
   `python scripts/benchmark.py --set v2` with `OPENAI_API_KEY` set to see the
   difference with a real embedder on your own corpus.
