"""Calibrate and score abstention: refusing questions the corpus cannot answer.

The engine abstains when retrieval confidence (``Retriever.retrieve_with_confidence``)
falls below ``LUMEN_ABSTAIN_THRESHOLD``. This module answers "which threshold, and
what does it cost?" on a labelled set that includes unanswerable questions:

* ``score_cases``  runs every question and records its confidence, whether it is
  answerable, and (for answerable ones) whether a relevant doc was retrieved.
* ``sweep``        shows the trade-off at each threshold: unanswerable questions
  refused vs answerable questions wrongly refused.
* ``auroc``        is the threshold-free summary: the probability that a random
  answerable question scores higher than a random unanswerable one.
* ``calibrate``    picks the threshold on one half of the set (max Youden's J) and
  reports it on the other half, so the published numbers are not tuned on their
  own test data.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

from ..reader import ExtractiveReader
from ..retrieval import Retriever
from ..retrieval.retriever import RetrievalMode
from .harness import EvalCase


@dataclass
class ScoredCase:
    question: str
    category: str | None
    answerable: bool
    confidence: float
    # Answerable cases: a relevant doc is in the top k. None for unanswerable ones.
    hit: bool | None
    near_miss: bool = False


def score_cases(
    retriever: Retriever,
    cases: list[EvalCase],
    k: int = 5,
    mode: RetrievalMode = "hybrid",
    reader: ExtractiveReader | None = None,
) -> list[ScoredCase]:
    """Confidence per case: retrieval confidence, or with ``reader`` its answerability."""
    scored = []
    for case in cases:
        if reader is None:
            results, confidence = retriever.retrieve_with_confidence(case.question, k=k, mode=mode)
        else:
            results = retriever.retrieve(case.question, k=k, mode=mode)
            confidence = reader.read(case.question, results).answerability
        hit = None
        if case.answerable:
            relevant = set(case.relevant_doc_ids)
            hit = any(r.chunk.doc_id in relevant for r in results)
        scored.append(
            ScoredCase(case.question, case.category, case.answerable, float(confidence), hit, case.near_miss)
        )
    return scored


def gate_together(first: list[ScoredCase], first_threshold: float, second: list[ScoredCase]) -> list[ScoredCase]:
    """Cases scored by ``second``, but already refused where ``first`` is below its threshold.

    Models a two-stage gate (retrieval similarity, then the reader): thresholds
    swept on the result apply to the second signal only.
    """
    out = []
    for a, b in zip(first, second, strict=True):
        conf = b.confidence if a.confidence >= first_threshold else float("-inf")
        out.append(ScoredCase(b.question, b.category, b.answerable, conf, b.hit, b.near_miss))
    return out


def outcome(scored: list[ScoredCase], threshold: float) -> dict:
    """Rates at one threshold (refuse when confidence < threshold)."""
    ans = [s for s in scored if s.answerable]
    unans = [s for s in scored if not s.answerable]
    refused_unans = sum(s.confidence < threshold for s in unans)
    refused_ans = sum(s.confidence < threshold for s in ans)
    answered_hit = sum(s.confidence >= threshold and bool(s.hit) for s in ans)
    n = len(scored)
    return {
        "threshold": round(float(threshold), 4),
        # Unanswerable questions correctly refused (abstention recall).
        "refusal_rate_unanswerable": refused_unans / len(unans) if unans else 0.0,
        # Answerable questions wrongly refused.
        "false_refusal_rate": refused_ans / len(ans) if ans else 0.0,
        # Answerable questions answered with a relevant doc in context.
        "answered_with_evidence": answered_hit / len(ans) if ans else 0.0,
        # Right outcome overall: refused if unanswerable, answered with evidence if answerable.
        "correct_outcome_rate": (refused_unans + answered_hit) / n if n else 0.0,
    }


def sweep(scored: list[ScoredCase], thresholds: list[float]) -> list[dict]:
    return [outcome(scored, t) for t in thresholds]


def auroc(scored: list[ScoredCase]) -> float:
    """P(confidence of a random answerable > that of a random unanswerable); ties count half."""
    pos = np.array([s.confidence for s in scored if s.answerable])
    neg = np.array([s.confidence for s in scored if not s.answerable])
    if not len(pos) or not len(neg):
        return float("nan")
    # Compare rather than subtract: refused-by-an-earlier-gate cases score -inf, and
    # -inf - -inf is nan, which would drop those ties instead of counting them half.
    greater = pos[:, None] > neg[None, :]
    equal = pos[:, None] == neg[None, :]
    return float((greater.sum() + 0.5 * equal.sum()) / greater.size)


def best_threshold(scored: list[ScoredCase]) -> float:
    """Threshold maximising Youden's J = refusal rate on unanswerable − false-refusal rate.

    Candidates are midpoints between consecutive distinct confidences (plus the
    ends), so the choice does not sit exactly on a training example.
    """
    values = sorted({s.confidence for s in scored})
    if not values:
        return 0.0
    candidates = [values[0] - 1e-6] + [(a + b) / 2 for a, b in pairwise(values)] + [values[-1] + 1e-6]
    best, best_j = candidates[0], -2.0
    for t in candidates:
        o = outcome(scored, t)
        j = o["refusal_rate_unanswerable"] - o["false_refusal_rate"]
        if j > best_j + 1e-12:
            best, best_j = t, j
    return float(best)


def split(scored: list[ScoredCase]) -> tuple[list[ScoredCase], list[ScoredCase]]:
    """Deterministic, stratified 50/50 dev/test split keyed on a hash of the question."""
    dev: list[ScoredCase] = []
    test: list[ScoredCase] = []
    for answerable in (True, False):
        group = sorted(
            (s for s in scored if s.answerable == answerable),
            key=lambda s: hashlib.sha256(s.question.encode()).hexdigest(),
        )
        dev.extend(group[0::2])
        test.extend(group[1::2])
    return dev, test


def conservative_threshold(scored: list[ScoredCase], max_false_refusal: float = 0.05) -> float:
    """Highest threshold whose false-refusal rate stays within ``max_false_refusal``.

    Youden's J weighs a wrong refusal and a wrong answer equally; a product usually
    tolerates far fewer refusals of answerable questions, so this is the other
    operating point worth reporting.
    """
    values = sorted({s.confidence for s in scored})
    candidates = [values[0] - 1e-6] + [(a + b) / 2 for a, b in pairwise(values)] if values else [0.0]
    best = candidates[0]
    for t in candidates:
        if outcome(scored, t)["false_refusal_rate"] <= max_false_refusal:
            best = t
    return float(best)


def calibrate(scored: list[ScoredCase], max_false_refusal: float = 0.05) -> dict:
    """Choose thresholds on the dev half and report them on the held-out half.

    Two operating points: ``threshold`` maximises Youden's J; ``conservative``
    is the highest threshold that refuses at most ``max_false_refusal`` of the
    answerable dev questions.
    """
    dev, test = split(scored)
    threshold = best_threshold(dev)
    conservative = conservative_threshold(dev, max_false_refusal)
    return {
        "threshold": round(threshold, 4),
        "auroc": round(auroc(scored), 4),
        "auroc_test": round(auroc(test), 4),
        # Full set: answerable vs each kind of unanswerable question.
        "auroc_near_miss": round(auroc([s for s in scored if s.answerable or s.near_miss]), 4),
        "auroc_off_topic": round(auroc([s for s in scored if s.answerable or not s.near_miss]), 4),
        "n_dev": len(dev),
        "n_test": len(test),
        "n_unanswerable_test": sum(not s.answerable for s in test),
        "dev": outcome(dev, threshold),
        "test": outcome(test, threshold),
        "conservative": {
            "max_false_refusal": max_false_refusal,
            "threshold": round(conservative, 4),
            "dev": outcome(dev, conservative),
            "test": outcome(test, conservative),
        },
        "no_abstention_test": outcome(test, float("-inf")),
    }


def as_rows(scored: list[ScoredCase]) -> list[dict]:
    return [asdict(s) for s in scored]
