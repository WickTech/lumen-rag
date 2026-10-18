# Eval set v2

A harder retrieval benchmark for Lumen RAG. It exists because the original set
(`data/eval.jsonl`, 7 documents, 19 questions) is saturated: with `k=5` the
retriever returns 5 of the 7 documents, so recall is about 1.0 for any
configuration and cannot show an improvement.

## Contents

- `docs/` — 120 short documents (60 to 110 words each), one markdown file per
  document. The file name (without `.md`) is the document id.
- `eval.jsonl` — 176 labelled questions.

## Provenance and licence

All documents and questions were written for this repository about a
**fictional** company, "Halden Robotics" (warehouse picking robots). No real
company, person or customer data is used and nothing is copied from another
source, so there are no third-party licence terms. The data is released under
the repository's MIT licence.

## Question categories

| Category | Count | What it tests |
|---|---|---|
| `direct` | 42 | Wording overlaps the document; the easy baseline. |
| `paraphrase` | 51 | Same fact, different words ("holiday days" vs "paid leave"). |
| `disambiguation` | 31 | Several near-identical documents (per region, plan, model or team); only one is right. |
| `multihop` | 29 | The answer needs two documents; both must be retrieved. |
| `unanswerable` | 23 | The corpus does not contain the answer. No relevant docs; counted, not scored. |

The corpus is built from 24 topic families of about five documents each (time
off, equipment, expenses, on-call, runbooks, release process, pricing, support
terms, security, data retention, hiring, reviews, robot hardware, robot safety,
fleet API, deployment, procurement, finance, legal, marketing, facilities,
internal tools, engineering practice, customer success). Near-duplicate
documents within a family are deliberate distractors.

## Row format

```json
{"question": "...", "relevant_doc_ids": ["vacation-eu"], "expected_answer": "30 working days.",
 "category": "direct", "answerable": true,
 "evidence": [{"doc_id": "vacation-eu", "span": "receive 30 working days of annual paid leave"}]}
```

`evidence[].span` is a verbatim excerpt of the cited document, so every answerable
label can be checked mechanically. Unanswerable rows have `relevant_doc_ids: []`,
`answerable: false`, no `evidence` and no `expected_answer`.

## Using it

```bash
python scripts/validate_eval_set.py            # consistency checks (ids, spans, duplicates, categories)
python scripts/benchmark.py --set v2           # configurations x categories table
lumen ingest data/eval_v2/docs
lumen eval data/eval_v2/eval.jsonl --k 5       # one-off scorecard with per-category breakdown
lumen eval-answers data/eval_v2/eval.jsonl --claims   # why answers go wrong: retrieval vs generation
```

## Baseline (offline hashing embedder, no reranker, k=5)

| Configuration | recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| vector-only | 0.74 | 0.69 | 0.69 |
| BM25 only | 0.90 | 0.88 | 0.87 |
| hybrid (BM25 + RRF) | 0.82 | 0.73 | 0.74 |

The spread between best and worst configuration is 0.19 MRR (the M0 target was at
least 0.10). `tests/test_eval_set_v2.py` guards this.

## Known limits

- The documents are short, so chunking has no effect on this set.
- Paraphrases still share some vocabulary with the source documents, which keeps
  BM25 strong. A set written against a real semantic embedder would be harder.
- The cross-encoder reranker was not measured (it needs a model download).
- Written by one author with LLM assistance; questions and answers were checked
  mechanically (spans, ids) but not by independent annotators.

## Extending it

Add a document under `docs/`, add rows to `eval.jsonl` with a verbatim `evidence`
span, then run `python scripts/validate_eval_set.py`. Keep one fact per question
and add distractor documents when you add a new topic.
