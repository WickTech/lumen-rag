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
from ..retrieval.retriever import RetrievalMode
from . import metrics


@dataclass
class EvalCase:
    question: str
    relevant_doc_ids: list[str]
    # Optional reference answer; enables contextual precision/recall in answer eval.
    expected_answer: str | None = None
    # Optional label for per-category breakdowns (e.g. direct, paraphrase, multihop).
    category: str | None = None
    # False for questions the corpus cannot answer. They have no relevant docs, so
    # they are excluded from retrieval metrics and only counted (``n_unanswerable``).
    answerable: bool = True


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
    # Answerable cases only; unanswerable ones are counted here, not scored.
    n_unanswerable: int = 0
    # category -> {"n": int, "recall@k": float, "mrr": float, "ndcg@k": float, "hit_rate": float}
    by_category: dict[str, dict] = field(default_factory=dict)

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
        if self.n_unanswerable:
            lines.append(f"  ({self.n_unanswerable} unanswerable questions not scored)")
        if self.by_category:
            lines.append("")
            lines.append(f"  {'category':<16}{'n':>4}  {'recall@k':>9}  {'mrr':>7}  {'ndcg@k':>7}")
            for name, row in sorted(self.by_category.items()):
                lines.append(
                    f"  {name:<16}{row['n']:>4}  {row['recall@k']:>9.4f}  "
                    f"{row['mrr']:>7.4f}  {row['ndcg@k']:>7.4f}"
                )
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
        cases.append(
            EvalCase(
                obj["question"],
                list(obj.get("relevant_doc_ids", [])),
                obj.get("expected_answer"),
                obj.get("category"),
                bool(obj.get("answerable", True)),
            )
        )
    return cases


def evaluate(
    retriever: Retriever,
    cases: list[EvalCase],
    k: int = 5,
    mode: RetrievalMode = "hybrid",
) -> EvalReport:
    agg = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0, "hit": 0.0}
    cat_agg: dict[str, dict[str, float]] = {}
    per_case = []
    n_unanswerable = 0

    for case in cases:
        if not case.answerable:
            # Nothing to retrieve; scoring it would divide by zero relevant docs.
            n_unanswerable += 1
            continue
        results = retriever.retrieve(case.question, k=k, mode=mode)
        # Retrieval is chunk-level but relevance is doc-level: collapse to the
        # rank-ordered list of *unique* doc ids so a doc with several retrieved
        # chunks counts once (otherwise recall/nDCG can exceed 1.0).
        retrieved_ids = _unique_preserving_order(r.chunk.doc_id for r in results)
        row = {
            "question": case.question,
            "category": case.category,
            "recall": metrics.recall_at_k(retrieved_ids, case.relevant_doc_ids, k),
            "precision": metrics.precision_at_k(retrieved_ids, case.relevant_doc_ids, k),
            "mrr": metrics.mrr(retrieved_ids, case.relevant_doc_ids),
            "ndcg": metrics.ndcg_at_k(retrieved_ids, case.relevant_doc_ids, k),
            "hit": metrics.hit_rate(retrieved_ids, case.relevant_doc_ids, k),
            "retrieved": retrieved_ids,
        }
        for key in agg:
            agg[key] += row[key]
        if case.category:
            bucket = cat_agg.setdefault(
                case.category, {"n": 0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "hit": 0.0}
            )
            bucket["n"] += 1
            for key in ("recall", "mrr", "ndcg", "hit"):
                bucket[key] += row[key]
        per_case.append(row)

    scored = len(per_case)
    n = max(1, scored)
    by_category = {
        name: {
            "n": int(b["n"]),
            "recall@k": round(b["recall"] / b["n"], 4),
            "mrr": round(b["mrr"] / b["n"], 4),
            "ndcg@k": round(b["ndcg"] / b["n"], 4),
            "hit_rate": round(b["hit"] / b["n"], 4),
        }
        for name, b in cat_agg.items()
    }
    return EvalReport(
        k=k,
        n_cases=scored,
        recall_at_k=agg["recall"] / n,
        precision_at_k=agg["precision"] / n,
        mrr=agg["mrr"] / n,
        ndcg_at_k=agg["ndcg"] / n,
        hit_rate=agg["hit"] / n,
        per_case=per_case,
        n_unanswerable=n_unanswerable,
        by_category=by_category,
    )
