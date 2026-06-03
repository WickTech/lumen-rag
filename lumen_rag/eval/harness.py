"""Run a labelled question set through a retriever and aggregate metrics.

This is what turns "the demo felt good" into "recall@5 is 0.82". Point it at a
JSONL of questions with known-relevant doc ids and it prints a scorecard you
can track across changes to chunking, embeddings, or reranking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..retrieval import Retriever
from . import metrics


@dataclass
class EvalCase:
    question: str
    relevant_doc_ids: list[str]


@dataclass
class EvalReport:
    k: int
    n_cases: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    per_case: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "n_cases": self.n_cases,
            "recall@k": round(self.recall_at_k, 4),
            "precision@k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@k": round(self.ndcg_at_k, 4),
            "hit_rate": round(self.hit_rate, 4),
        }

    def pretty(self) -> str:
        d = self.as_dict()
        lines = [f"  Retrieval eval — {d['n_cases']} cases @ k={d['k']}", "  " + "-" * 34]
        for key in ("recall@k", "precision@k", "mrr", "ndcg@k", "hit_rate"):
            lines.append(f"  {key:<14} {d[key]:.4f}")
        return "\n".join(lines)


def _unique_preserving_order(items) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_cases(path: str | Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cases.append(EvalCase(obj["question"], list(obj["relevant_doc_ids"])))
    return cases


def evaluate(retriever: Retriever, cases: list[EvalCase], k: int = 5) -> EvalReport:
    agg = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0, "hit": 0.0}
    per_case = []

    for case in cases:
        results = retriever.retrieve(case.question, k=k)
        # Retrieval is chunk-level but relevance is doc-level: collapse to the
        # rank-ordered list of *unique* doc ids so a doc with several retrieved
        # chunks counts once (otherwise recall/nDCG can exceed 1.0).
        retrieved_ids = _unique_preserving_order(r.chunk.doc_id for r in results)
        row = {
            "question": case.question,
            "recall": metrics.recall_at_k(retrieved_ids, case.relevant_doc_ids, k),
            "precision": metrics.precision_at_k(retrieved_ids, case.relevant_doc_ids, k),
            "mrr": metrics.mrr(retrieved_ids, case.relevant_doc_ids),
            "ndcg": metrics.ndcg_at_k(retrieved_ids, case.relevant_doc_ids, k),
            "hit": metrics.hit_rate(retrieved_ids, case.relevant_doc_ids, k),
            "retrieved": retrieved_ids,
        }
        for key in agg:
            agg[key] += row[key]
        per_case.append(row)

    n = max(1, len(cases))
    return EvalReport(
        k=k,
        n_cases=len(cases),
        recall_at_k=agg["recall"] / n,
        precision_at_k=agg["precision"] / n,
        mrr=agg["mrr"] / n,
        ndcg_at_k=agg["ndcg"] / n,
        hit_rate=agg["hit"] / n,
        per_case=per_case,
    )
