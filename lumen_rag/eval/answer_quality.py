"""Answer-quality evaluation with an LLM judge (DeepEval).

The retrieval harness (``harness.py``) tells you whether the right *documents*
came back. This module asks the next question: given that context, is the
generated *answer* any good?

Metrics (via DeepEval, judged by an OpenAI model):
  faithfulness         — are the answer's claims supported by the retrieved context?
  answer_relevancy     — does the answer actually address the question?
  contextual_relevancy — how much of the retrieved context is on-topic?
  contextual_precision / contextual_recall — only for cases with ``expected_answer``.

Requires ``pip install 'lumen-rag[answer-eval]'`` and ``OPENAI_API_KEY`` — an
offline (extractive) answer has no generator to judge, so offline runs are refused.

Scoring is behind the small ``Scorer`` interface so the harness itself is
testable without DeepEval or network access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..config import settings
from ..engine import RagEngine
from ..llm import answer
from ..retrieval import RetrievalMode
from .harness import EvalCase


@dataclass
class AnswerSample:
    question: str
    answer: str
    retrieval_context: list[str]
    expected_answer: str | None = None


@dataclass
class MetricResult:
    score: float
    reason: str | None = None
    error: str | None = None


class Scorer(Protocol):
    def score(self, sample: AnswerSample) -> dict[str, MetricResult]:
        ...


@dataclass
class AnswerEvalReport:
    n_cases: int
    threshold: float
    means: dict[str, float]
    pass_rates: dict[str, float]
    errors: int
    per_case: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "threshold": self.threshold,
            "mean": {k: round(v, 4) for k, v in self.means.items()},
            "pass_rate": {k: round(v, 4) for k, v in self.pass_rates.items()},
            "errors": self.errors,
        }

    def pretty(self) -> str:
        lines = [
            f"  Answer eval — {self.n_cases} cases, pass ≥ {self.threshold}",
            "  " + "-" * 48,
            f"  {'metric':<22} {'mean':>8} {'pass rate':>11}",
        ]
        for name, mean in self.means.items():
            lines.append(f"  {name:<22} {mean:>8.4f} {self.pass_rates[name]:>11.2%}")
        if self.errors:
            lines.append(f"  ({self.errors} metric evaluation(s) failed — see per-case output)")
        return "\n".join(lines)


class DeepEvalScorer:
    """Runs DeepEval's RAG metrics on each sample with an OpenAI judge model."""

    def __init__(self, judge_model: str = "gpt-4o-mini", threshold: float = 0.5) -> None:
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                ContextualRelevancyMetric,
                FaithfulnessMetric,
            )
        except ImportError as e:
            raise ImportError(
                "deepeval is required for answer-quality eval. "
                "Install it with: pip install 'lumen-rag[answer-eval]'"
            ) from e

        # async_mode=False: measure() runs synchronously, one sample at a time,
        # which keeps judge-model rate limits and error attribution simple.
        opts = {"model": judge_model, "threshold": threshold, "async_mode": False}
        self._always = {
            "faithfulness": FaithfulnessMetric(**opts),
            "answer_relevancy": AnswerRelevancyMetric(**opts),
            "contextual_relevancy": ContextualRelevancyMetric(**opts),
        }
        self._with_reference = {
            "contextual_precision": ContextualPrecisionMetric(**opts),
            "contextual_recall": ContextualRecallMetric(**opts),
        }

    def score(self, sample: AnswerSample) -> dict[str, MetricResult]:
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=sample.question,
            actual_output=sample.answer,
            retrieval_context=sample.retrieval_context,
            expected_output=sample.expected_answer,
        )
        metrics = dict(self._always)
        if sample.expected_answer:
            metrics.update(self._with_reference)

        results: dict[str, MetricResult] = {}
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                results[name] = MetricResult(score=float(metric.score), reason=metric.reason)
            except Exception as e:  # one bad judge call shouldn't sink the whole run
                results[name] = MetricResult(score=0.0, error=f"{type(e).__name__}: {e}")
        return results


def evaluate_answers(
    engine: RagEngine,
    cases: list[EvalCase],
    *,
    k: int = 5,
    mode: RetrievalMode = "hybrid",
    scorer: Scorer | None = None,
    threshold: float = 0.5,
    judge_model: str = "gpt-4o-mini",
) -> AnswerEvalReport:
    """Retrieve, generate, and judge an answer for every case."""
    if scorer is None:
        if settings.offline:
            raise RuntimeError(
                "Answer-quality eval needs a real generator and judge: set OPENAI_API_KEY. "
                "(Offline mode only produces extractive answers.)"
            )
        scorer = DeepEvalScorer(judge_model=judge_model, threshold=threshold)

    totals: dict[str, float] = {}
    passes: dict[str, int] = {}
    counts: dict[str, int] = {}
    errors = 0
    per_case = []

    for case in cases:
        chunks = engine.retriever.retrieve(case.question, k=k, mode=mode)
        generated = answer(case.question, chunks)
        sample = AnswerSample(
            question=case.question,
            answer=generated.text,
            retrieval_context=[c.chunk.text for c in chunks],
            expected_answer=case.expected_answer,
        )
        results = scorer.score(sample)

        row: dict = {"question": case.question, "answer": generated.text, "metrics": {}}
        for name, res in results.items():
            row["metrics"][name] = {"score": res.score, "reason": res.reason, "error": res.error}
            if res.error:
                errors += 1
                continue  # failed evaluations don't count toward means / pass rates
            totals[name] = totals.get(name, 0.0) + res.score
            counts[name] = counts.get(name, 0) + 1
            passes[name] = passes.get(name, 0) + (res.score >= threshold)
        per_case.append(row)

    return AnswerEvalReport(
        n_cases=len(cases),
        threshold=threshold,
        means={name: totals[name] / counts[name] for name in counts},
        pass_rates={name: passes[name] / counts[name] for name in counts},
        errors=errors,
        per_case=per_case,
    )
