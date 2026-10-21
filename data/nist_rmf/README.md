# NIST AI RMF retrieval set

22 questions about the [NIST AI Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1)
(NIST AI 100-1, January 2023), for measuring retrieval *inside* one long document.

- `eval.jsonl`: one row per question, `{"question", "evidence": [...]}`. `evidence` lists verbatim
  spans of the document; a retrieved chunk counts as relevant when it contains one of them.
  Scored by `lumen_rag.eval.evidence` (hit@k, MRR).
- The PDF itself is not stored here. `scripts/real_pdf_check.py --retrieval` downloads it
  from nvlpubs.nist.gov. It is a work of the US federal government and not subject to copyright
  in the United States.
- Questions were written for this repository and mostly avoid the evidence's own wording, so that
  where a chunk sits in the document (its section) matters. Each span was checked to occur in a
  120-word chunk of the extracted text.
- 22 questions is small: one question moves hit@5 by 0.045. Treat differences of one or two
  questions as noise.
