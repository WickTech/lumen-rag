# Abstention

How Lumen refuses questions its documents cannot answer, how the thresholds were calibrated, and what it costs. Back to the [README](../README.md).

## 🙅 Abstention — refusing what the documents don't say

A RAG system that always answers will answer questions its documents cannot (the first of the
[Seven Failure Points](https://arxiv.org/abs/2401.05856)). Lumen can refuse instead:

```bash
export LUMEN_EMBEDDER=fastembed LUMEN_ABSTAIN_THRESHOLD=0.66
lumen ingest data/eval_v2/docs
lumen ask "Are hotel minibar charges reimbursable?" --mode vector
# I can't find this in the indexed documents, so I won't guess.
# Retrieval confidence: 0.627 (abstained)
lumen ask "What is the street address of the Frankfurt office?" --mode vector
# Retrieval confidence: 0.800  -> answered: a near miss (the office doc exists, the address doesn't)
```

The confidence is the best cosine similarity between the question and any stored chunk (or, with
a reranker configured, the sigmoid of the best cross-encoder score). Below the threshold the engine
returns a refusal **without calling the model**, and `/query`, `/query/stream` and `lumen ask`
report `abstained` and `confidence`. Unset, nothing changes.

**How the threshold was chosen.** 63 of the 216 v2 questions are unanswerable; 40 of them are
*near misses*, on-topic for a real document but asking for a fact it does not contain.
`lumen_rag.eval.abstention` splits the set in half (stratified, deterministic), picks thresholds on
one half and reports them on the other, so the numbers below are not tuned on their own test data.

Held-out half (107 questions, 31 unanswerable), bge-small, vector mode:

| Signal | AUROC | Operating point | Threshold | Unanswerable refused | Answerable wrongly refused | Right outcome* |
|---|---|---|---|---|---|---|
| no abstention | — | — | — | 0.00 | 0.00 | 0.70 |
| top dense cosine | 0.75 | ≤5% false refusals | 0.664 | 0.26 | 0.03 | **0.76** |
| top dense cosine | 0.75 | max Youden J | 0.740 | 0.71 | 0.29 | 0.71 |
| cross-encoder score | 0.80 | ≤5% false refusals | 0.005 | 0.19 | 0.03 | 0.75 |
| hashing cosine (CI) | 0.58 | max Youden J | 0.262 | 0.26 | 0.08 | 0.62 (0.59 without) |

\*Refused if unanswerable, or answered with a relevant document in the top 5 if answerable.

What it shows:

- **The conservative setting is a clear win**: it refuses a quarter of the unanswerable questions
  while turning away 3% of the answerable ones, and the share of right outcomes goes from 0.70 to 0.76.
- **Pushing harder does not pay.** Balancing both errors equally (Youden's J) refuses 71% of
  unanswerable questions but also 29% of answerable ones.
- **Similarity cannot see near misses.** Measured on the full set, the dense signal separates the
  original 23 unanswerable questions from answerable ones with AUROC 0.82, but the 40 near misses
  only 0.69: "What is the street address of the Frankfurt office?" scores 0.80 against the
  Frankfurt office page, which does not give one, higher than many answerable questions. Catching those needs a check that reads the passage:
  the reader below.
- **With the hashing embedder abstention is close to chance** (AUROC 0.58). Calibrate per embedder:
  thresholds do not transfer.

### The reader: a second gate that reads the passage

Similarity asks "is this question about something indexed?". An extractive reader trained on
SQuAD 2.0, which includes unanswerable questions, asks "does a retrieved passage contain the
answer?": it scores the best answer span against a "no answer" option, and Lumen refuses when
`answerability = best span − no answer` is below `LUMEN_READER_THRESHOLD`. It runs on ONNX
(`onnx-community/roberta-base-squad2-ONNX`, int8), so still no torch and no API key.

```bash
pip install -e ".[embed,reader]"
export LUMEN_EMBEDDER=fastembed LUMEN_ABSTAIN_THRESHOLD=0.66 \
       LUMEN_READER=onnx-community/roberta-base-squad2-ONNX LUMEN_READER_THRESHOLD=-3.5
python scripts/benchmark.py --set v2 --embedder fastembed --abstain --reader "$LUMEN_READER"
```

v2, bge-small, vector mode; thresholds chosen on the dev half, rates on the held-out half:

| Signal | AUROC | AUROC near-miss | AUROC off-topic | Operating point | Unanswerable refused | Answerable wrongly refused | Right outcome |
|---|---|---|---|---|---|---|---|
| retrieval similarity | 0.75 | 0.69 | 0.82 | ≤5% false refusals | 0.26 | 0.03 | 0.76 |
| tinyroberta reader (82 MB) | 0.81 | 0.76 | 0.87 | ≤5% false refusals | 0.26 | 0.07 | 0.73 |
| **roberta-base reader** (125 MB) | **0.88** | **0.80** | **0.94** | ≤5% false refusals | **0.42** | 0.03 | **0.80** |
| roberta-base reader | 0.88 | 0.80 | 0.94 | SQuAD rule (threshold 0) | 0.77 | 0.20 | 0.79 |

At the same 3% false-refusal rate the reader refuses 42% of unanswerable questions against 26% for
similarity, and right outcomes go from 0.70 (no abstention) to 0.80. Its cost is latency: 0.4 to
0.9 s a query on a laptop CPU for five short passages (1.5 to 4 s for five PDF pages' worth), against
~20 ms for the similarity gate. Running
similarity first lets clearly off-topic questions skip the reader.

**On a real document.** [`scripts/real_pdf_check.py`](../scripts/real_pdf_check.py) downloads the
[NIST AI Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1) (48-page PDF,
public domain), indexes it and asks six questions it answers, three near misses and one off-topic
question:

| Setup | Right outcome |
|---|---|
| no gates | 6/10 (answers everything) |
| similarity ≥ 0.66 | 7/10 (refuses only the off-topic question) |
| similarity ≥ 0.66, then reader ≥ −3.5 | **10/10** |

"When will version 2.0 of the AI RMF be released?" scores 0.80 on similarity (the document
discusses its review schedule and versioning) and is refused only by the reader. Ten questions are
a spot check, not a benchmark, and two margins were close (−1.7 answered, −3.7 refused).

**End to end**, with claim diagnostics on all 216 v2 questions (`lumen eval-answers --claims`,
bge-small, vector mode, offline extractive answers):

| Gates | Unanswerable refused (of 63) | Answer accuracy (answerable) | Claim precision |
|---|---|---|---|
| none | 0 | 0.94 | 0.39 |
| similarity ≥ 0.664 | 17 | 0.92 | 0.43 |
| reader ≥ −3.5 | 26 | 0.91 | 0.44 |
| similarity ≥ 0.664, then reader ≥ −3.5 | **34** | 0.88 | 0.46 |
| same, answering with the reader's sentence (`LUMEN_READER_ANSWERS=1`) | 34 | 0.68 | **0.75** |

Both gates together refuse over half of the unanswerable questions for a 6-point loss in accuracy
on answerable ones. Answering with the reader's sentence is opt-in for a reason: the answers are far
more focused (claim precision 0.75), but a single sentence cannot answer a multi-hop question (16 of
the 29 multihop questions get a wrong one-sentence answer) and the reader sometimes picks the span from a near-duplicate
distractor, e.g. the sick-leave sentence for an EU vacation question. The default keeps the reader
as a gate and answers with the top passages.
