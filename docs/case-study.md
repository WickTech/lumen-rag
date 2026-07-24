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

| Configuration | recall@5 | precision@5 | MRR | nDCG@5 | hit rate |
|---|---|---|---|---|---|
| naive — 1 chunk per doc, vector-only | 0.97 | 0.20 | 0.93 | 0.94 | 1.00 |
| + sentence-aware chunking, vector-only | 0.97 | 0.20 | **0.97** | **0.97** | 1.00 |
| + hybrid (BM25 + Reciprocal Rank Fusion) | 0.97 | 0.20 | **0.97** | **0.97** | 1.00 |

## Reading the numbers honestly

**Recall and hit-rate are already saturated** at 0.97/1.00 in the naive
config — with only 7 candidate documents and distinct enough vocabulary
per topic, the correct document almost always lands somewhere in the top 5
regardless of technique. Publishing only recall would (falsely) suggest
chunking doesn't matter here. It does — just not on that metric.

**MRR and nDCG@5 are where the effect shows up.** Both measure *where* the
correct document ranks, not just whether it's present. Moving from naive to
chunked lifts MRR from 0.93 to 0.97 — a small absolute jump that maps to a
concrete failure mode disappearing: when a whole 300–450 word multi-topic
document is embedded as a single vector, the sections irrelevant to the
query dilute the average, and the correct document occasionally ranks 2nd
or 3rd behind a partial-vocabulary-match decoy instead of 1st. Sentence-aware
chunking (120-word windows, 20-word overlap) embeds each section on its own,
so the answer-bearing chunk competes on its own signal instead of being
outvoted by the rest of the document.

**Hybrid ties chunked-vector, not because RRF doesn't work, but because the
offline `HashingEmbedder` is itself a bag-of-words signal** — term-frequency
counts, L2-normalised. That's structurally close to what BM25 computes, so
fusing the two rankers mostly agrees with itself. Hybrid's actual value
proposition — catching exact rare-term or numeric matches that a *semantic*
embedding model under-weights in favor of topical similarity — needs a real
semantic embedder (`OPENAI_API_KEY` set, `text-embedding-3-small`) and a
larger, noisier corpus to demonstrate honestly. That's flagged as a follow-up
rather than asserted with numbers we didn't measure.

## Takeaways for anyone building on Lumen

1. **Don't just report recall/hit-rate.** They saturate fast on small
   corpora and hide ranking-quality regressions. MRR/nDCG catch what recall
   misses.
2. **Chunking's benefit scales with document length and topic density**, not
   corpus size. A corpus of short, single-topic documents won't show this
   effect — you need documents where the answer is a minority of the content.
3. **Hybrid retrieval's payoff is embedder-dependent.** With a lexical/hash
   embedder it's close to redundant with vector search; with a real semantic
   embedder it complements it. Re-run `scripts/benchmark.py` with
   `OPENAI_API_KEY` set to see the difference on your own corpus.
