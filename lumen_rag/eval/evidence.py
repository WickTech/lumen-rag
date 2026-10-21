"""Retrieval quality on a single long document, graded by evidence spans.

Document-level relevance (``relevant_doc_ids``) cannot score retrieval inside one
document: every chunk has the same doc id. Here each question lists verbatim
evidence spans instead, and a retrieved chunk is relevant when it contains one.
Used for the NIST AI RMF set in ``data/nist_rmf``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..retrieval import Retriever
from ..retrieval.retriever import RetrievalMode

_WS = re.compile(r"\s+")


@dataclass
class EvidenceCase:
    question: str
    evidence: list[str]


def load_evidence_cases(path: str | Path) -> list[EvidenceCase]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [EvidenceCase(r["question"], list(r["evidence"])) for r in rows]


def evaluate_evidence(
    retriever: Retriever, cases: list[EvidenceCase], k: int = 5, mode: RetrievalMode = "hybrid"
) -> dict:
    """hit@k (an evidence span is in the top k) and MRR (rank of the first such chunk)."""
    hits = rr = 0.0
    per_case = []
    for case in cases:
        spans = [_WS.sub(" ", e).strip() for e in case.evidence]
        results = retriever.retrieve(case.question, k=k, mode=mode)
        rank = next(
            (i for i, r in enumerate(results, 1) if any(s in _WS.sub(" ", r.chunk.text) for s in spans)), None
        )
        hits += rank is not None
        rr += 1 / rank if rank else 0.0
        per_case.append({"question": case.question, "rank": rank})
    n = max(1, len(cases))
    return {"k": k, "n_cases": len(cases), "hit@k": hits / n, "mrr": rr / n, "per_case": per_case}
