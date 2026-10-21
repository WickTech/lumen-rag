# Benchmarks

What embeddings, retrieval mode, reranking, contextual headers and query decomposition actually buy you, measured on the bundled eval sets. Back to the [README](../README.md).

## 📊 Benchmarks — what embeddings, retrieval mode and reranking actually buy you

Measured with `scripts/benchmark.py`, k=5. Two eval sets ship with the repo:

- **v1** — `data/docs` (7 files) + `data/eval.jsonl` (19 questions). A fast smoke
  test, but **saturated**: k=5 returns 5 of the 7 documents, so recall is ~1.0
  for any configuration and tells you nothing.
- **v2** — `data/eval_v2` (120 short docs, 216 questions in five categories:
  direct, paraphrase, disambiguation, multihop, unanswerable). Built so that
  configurations can be told apart; see [`data/eval_v2/README.md`](https://github.com/WickTech/lumen-rag/blob/master/data/eval_v2/README.md).

Reproduce:

```bash
python scripts/benchmark.py --set v2 --abstain                                # hashing embedder (CI default)
pip install -e ".[embed,rerank]"
python scripts/benchmark.py --set v2 --embedder fastembed --rerank --abstain  # bge-small + MiniLM reranker
```

### v2 with a real embedder (`BAAI/bge-small-en-v1.5`, local, CPU)

| Configuration | recall@5 | precision@5 | MRR | nDCG@5 | hit rate | ms/query |
|---|---|---|---|---|---|---|
| vector-only | 0.97 | 0.23 | 0.95 | 0.94 | 0.99 | 19 |
| BM25 only | 0.90 | 0.21 | 0.88 | 0.87 | 0.94 | 1 |
| hybrid (BM25 + RRF) | 0.97 | 0.23 | 0.91 | 0.91 | 0.99 | 15 |
| hybrid + rerank | 0.97 | 0.23 | 0.96 | 0.95 | 0.98 | 488 |
| vector-only + rerank | **0.98** | 0.23 | **0.97** | **0.96** | 0.99 | 489 |

MRR by question category (153 answerable questions; the 63 unanswerable ones are used for abstention below):

| Configuration | direct | disambiguation | multihop | paraphrase |
|---|---|---|---|---|
| vector-only | 0.98 | 0.96 | 0.93 | 0.93 |
| BM25 only | 0.98 | 0.89 | 0.86 | 0.80 |
| hybrid (BM25 + RRF) | 0.99 | 0.87 | 0.95 | 0.86 |
| hybrid + rerank | 1.00 | 0.97 | 0.96 | 0.92 |
| vector-only + rerank | 1.00 | 0.98 | 0.96 | 0.94 |

### v2 with the offline hashing embedder (what CI runs)

| Configuration | recall@5 | precision@5 | MRR | nDCG@5 | hit rate |
|---|---|---|---|---|---|
| vector-only | 0.74 | 0.17 | 0.69 | 0.69 | 0.78 |
| BM25 only | **0.90** | 0.21 | **0.88** | **0.87** | **0.94** |
| hybrid (BM25 + RRF) | 0.82 | 0.19 | 0.73 | 0.74 | 0.86 |

| Configuration | direct | disambiguation | multihop | paraphrase |
|---|---|---|---|---|
| vector-only | 0.84 | 0.77 | 0.67 | 0.54 |
| BM25 only | 0.98 | 0.89 | 0.86 | 0.80 |
| hybrid (BM25 + RRF) | 0.84 | 0.78 | 0.76 | 0.59 |

What the two tables say:

- **The embedder was the bottleneck.** Swapping the hashing embedder for bge-small lifts
  vector-only MRR from 0.69 to 0.95, and paraphrase MRR from 0.54 to 0.93: the questions that
  share no words with their document are exactly what a semantic model fixes.
- **The ranking of strategies flips.** With the weak embedder, BM25 alone wins and fusion drags it
  down. With a real one, dense search wins and *BM25 now drags the fusion down* (MRR 0.95 → 0.91,
  disambiguation 0.96 → 0.87): equal-weight RRF lets the weaker list promote near-duplicate
  distractors. "Hybrid is always better" is not true on this corpus; measure it on yours.
- **Reranking buys the last few points at 25× the latency.** A 22M-parameter cross-encoder
  (`Xenova/ms-marco-MiniLM-L-6-v2`, CPU) takes MRR from 0.95 to 0.97 and costs ~470 ms a query.
  Worth it when precision at rank 1 matters; not for an autocomplete box.
- **Contextual chunk headers help keyword search, barely move dense search.** With
  `LUMEN_CONTEXTUAL=1` each chunk is embedded and BM25-indexed with a header naming its document
  and section, e.g. `Document: Artificial Intelligence Risk Management Framework (AI RMF 1.0).
  Section: 3. AI Risks and Trustworthiness > 3.4 Accountable and Transparent.`, built offline from
  the title and the markdown, numbered or appendix headings (tables of contents and list items are
  skipped). The header is used for search only; answers and citations show the plain chunk, and
  citations name the section. See [Contextual chunk headers](#contextual-chunk-headers) below.
- **Query decomposition did not help (a negative result, kept opt-in).** Splitting compound
  questions at the second question word ("…, and how soon must I submit the claim?") and adding
  each sub-question's best new document recovers the missing hop on some multi-hop questions and
  loses it on as many: multi-hop recall@5 stays 0.90 with bge-small (0.95 with the reranker) and
  drops from 0.88 to 0.83 in hybrid mode. Equal-weight RRF of the sub-query lists did worse
  (0.88). The remaining misses are *implicit* hops ("How much bonus do I earn if checkout is
  down?" needs the severity table) that no lexical split can see; an LLM rewrite is the next thing
  to try. Reproduce with `--decompose`; enable with `LUMEN_DECOMPOSE=1`.
- **Not measured yet:** OpenAI `text-embedding-3-small`. The run is wired up
  (`--embedder openai`) but the key available when these numbers were taken had no API credit.

### Contextual chunk headers

[Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) prepends an
LLM-written context to each chunk and reports 35% fewer failed retrievals (49% with contextual BM25).
Lumen's headers are the offline, deterministic version: document title plus heading path, no model.
v2 documents fit in one 120-word chunk, so they only show the effect with 40-word chunks, where a
chunk loses its title; the NIST AI RMF set (22 questions graded by evidence spans,
[`data/nist_rmf`](../data/nist_rmf/README.md)) tests a real 48-page PDF with its section structure.

| Set | Embedder, mode | without headers | with headers |
|---|---|---|---|
| v2, 40-word chunks (MRR) | bge-small, vector | 0.93 | 0.92 |
| v2, 40-word chunks (MRR) | bge-small, hybrid | 0.90 | **0.92** (disambiguation 0.87 → 0.92) |
| v2, 40-word chunks (MRR) | BM25 | 0.81 | **0.85** |
| NIST PDF (hit@5) | bge-small, vector | 0.86 | **0.91** |
| NIST PDF (hit@5) | bge-small, hybrid | 0.91 | 0.91 |
| NIST PDF (hit@5) | BM25 | 0.82 | 0.77 |
| NIST PDF (hit@5) | hashing, hybrid | 0.55 | **0.73** |

- **Keyword search gains most on multi-document corpora**: the title gives every chunk the words
  that tell look-alike documents apart (Pro vs Starter plan). Inside one document the title is the
  same for every chunk, only the section path discriminates, and BM25 loses a question to the longer
  chunks.
- **Dense search with a good embedder barely moves** (one NIST question gained, v2 flat). The
  gains are a fraction of Anthropic's because a title and a heading path say much less than an
  LLM-written summary of where the chunk sits; that version needs an API key and is the obvious next
  step.
- **Headers move the abstention scores, so recalibrate when you turn them on.** On v2 the
  similarity threshold barely changes (0.666 vs 0.664), but on the NIST PDF, whose header is a
  long title plus section path, the end-to-end check with both gates drops from 10/10 to 8/10:
  "Which function is designed to be cross-cutting…" falls from 0.70 to 0.60 similarity and is
  wrongly refused, and one near miss rises above the reader threshold. That is why headers are off
  by default and off on the HF Space, whose thresholds were calibrated without them.
- NIST differences are one question (0.045) at a time, so read them as direction, not size.

Reproduce: `python scripts/benchmark.py --set v2 --embedder fastembed --contextual`,
`python scripts/real_pdf_check.py --retrieval`, and `LUMEN_CONTEXTUAL=1 python scripts/real_pdf_check.py`.
Enable with `LUMEN_CONTEXTUAL=1` and re-ingest.

### v1 (smoke test, hashing embedder)

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
