"""Information-retrieval metrics for evaluating a retriever.

All functions take ``retrieved`` (an ordered list of doc ids, best first) and
``relevant`` (the set of ids that *should* be retrieved). This is the standard
way to score a RAG retriever against a labelled question set.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _rel_set(relevant: Iterable[str]) -> set[str]:
    return set(relevant)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = _rel_set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in rel)
    return hits / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = _rel_set(relevant)
    hits = sum(1 for doc in retrieved[:k] if doc in rel)
    return hits / k


def hit_rate(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant doc appears in the top-k, else 0.0."""
    rel = _rel_set(relevant)
    return 1.0 if any(doc in rel for doc in retrieved[:k]) else 0.0


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    rel = _rel_set(relevant)
    for i, doc in enumerate(retrieved, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    rel = _rel_set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(retrieved[:k], start=1)
        if doc in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
