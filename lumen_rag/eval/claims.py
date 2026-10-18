"""Claim-level diagnostics: why did an answer go wrong?

Retrieval metrics say whether the right documents came back. Answer judges say
whether an answer reads well. Neither says *where* a bad answer came from. This
module splits each generated answer into claims, aligns every claim with the
retrieved chunks, and attributes failures to the retriever or to the generator.

Per claim (needs a labelled gold document set for the question):

  supported    grounded in a retrieved chunk from a gold document
  distractor   grounded in a retrieved chunk from a *non-gold* document: faithful
               to what was retrieved, but the evidence was the wrong evidence
  unsupported  not grounded in any retrieved chunk (fabricated or unfaithful)

Per answerable question:

  ok               the answer covers the expected answer
  retrieval-miss   the answer is wrong and not all gold documents were retrieved
  generation-error the answer is wrong although every gold document was retrieved
                   (the generator used the evidence badly or ignored it)

Per unanswerable question: ``abstained`` (the answer declines) or ``hallucinated``.

Alignment is a deterministic lexical overlap, so the whole report runs offline and
is reproducible. It is a proxy: it cannot see paraphrase or negation, so with a
generative answerer treat the rates as an approximation. ``aligner`` accepts any
``(claim, chunk_text) -> float in [0, 1]`` so an NLI model or LLM judge can be
plugged in without touching the rest.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..llm import Answer, answer
from ..retrieval.retriever import RetrievalMode, Retriever
from ..store import ScoredChunk
from .harness import EvalCase

Aligner = Callable[[str, str], float]
AnswerFn = Callable[[str, list[ScoredChunk]], Answer]

CLAIM_LABELS = ("supported", "distractor", "unsupported")
OUTCOMES = ("ok", "retrieval-miss", "generation-error")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CITATION_RE = re.compile(r"\[\d+\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9#(\"'])")
_ABSTAIN_RE = re.compile(
    r"(don't|do not|doesn't|does not) (know|have)|cannot (find|answer)|can't (find|answer)"
    r"|no (relevant )?(information|context)|not (found|mentioned|covered)",
    re.IGNORECASE,
)
_OFFLINE_PREFIX = "(offline extractive answer)"

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do",
        "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
        "me", "more", "most", "must", "my", "no", "not", "of", "on", "or", "our", "out", "over", "per",
        "so", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
        "those", "to", "up", "us", "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "will", "with", "would", "you", "your",
    }
)


def _fold(token: str) -> str:
    # Crude plural folding so "days" matches "day"; applied on both sides.
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    """Lower-cased content words (stop words dropped, plurals folded, numbers kept)."""
    tokens = _TOKEN_RE.findall(text.lower())
    return {_fold(t) for t in tokens if t.isdigit() or (t not in _STOPWORDS and len(t) > 1)}


def lexical_support(claim: str, chunk_text: str) -> float:
    """Fraction of the claim's content words that appear in the chunk."""
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & content_tokens(chunk_text)) / len(claim_tokens)


def is_abstention(text: str) -> bool:
    """True for a short answer that declines ("I don't know", "no information").

    Long answers are never abstentions: extractive answers are pasted document text,
    which can contain phrases like "does not have" for unrelated reasons.
    """
    cleaned = _CITATION_RE.sub(" ", text.replace(_OFFLINE_PREFIX, " "))
    return len(cleaned.split()) <= 30 and bool(_ABSTAIN_RE.search(cleaned))


def split_claims(text: str) -> list[str]:
    """Split an answer into sentence-level claims, without markers and citations."""
    text = text.replace(_OFFLINE_PREFIX, " ")
    text = _CITATION_RE.sub(" ", text)
    text = re.sub(r"\s+([.!?,;:])", r"\1", re.sub(r"\s+", " ", text)).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if len(content_tokens(s)) >= 2]


@dataclass
class ClaimReport:
    k: int
    mode: str
    tau: float
    n_cases: int  # answerable cases
    n_unanswerable: int
    n_claims: int
    claim_counts: dict[str, int]
    outcomes: dict[str, int]
    abstained: int  # unanswerable cases where the answer declined
    by_category: dict[str, dict] = field(default_factory=dict)
    per_case: list[dict] = field(default_factory=list)

    def _rate(self, num: float, den: float) -> float:
        return round(num / den, 4) if den else 0.0

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "mode": self.mode,
            "tau": self.tau,
            "n_cases": self.n_cases,
            "n_unanswerable": self.n_unanswerable,
            "n_claims": self.n_claims,
            "claim_precision": self._rate(self.claim_counts["supported"], self.n_claims),
            "distractor_rate": self._rate(self.claim_counts["distractor"], self.n_claims),
            "unsupported_rate": self._rate(self.claim_counts["unsupported"], self.n_claims),
            "faithfulness": self._rate(
                self.claim_counts["supported"] + self.claim_counts["distractor"], self.n_claims
            ),
            "answer_accuracy": self._rate(self.outcomes["ok"], self.n_cases),
            "retrieval_miss_rate": self._rate(self.outcomes["retrieval-miss"], self.n_cases),
            "generation_error_rate": self._rate(self.outcomes["generation-error"], self.n_cases),
            "abstain_rate": self._rate(self.abstained, self.n_unanswerable),
        }

    def pretty(self) -> str:
        d = self.as_dict()
        lines = [
            (
                f"  Claim diagnostics — {d['n_cases']} answerable questions, {d['n_claims']} claims "
                f"(k={d['k']}, mode={d['mode']}, support ≥ {d['tau']})"
            ),
            "  " + "-" * 60,
            "  Claims (share of all answer claims)",
            f"    supported      {d['claim_precision']:.4f}   grounded in retrieved gold evidence",
            f"    distractor     {d['distractor_rate']:.4f}   grounded in retrieved non-gold evidence",
            f"    unsupported    {d['unsupported_rate']:.4f}   not grounded in anything retrieved",
            f"    faithfulness   {d['faithfulness']:.4f}   supported + distractor",
            "  Answerable questions",
            f"    ok                {d['answer_accuracy']:.4f}",
            f"    retrieval-miss    {d['retrieval_miss_rate']:.4f}   wrong, gold document not retrieved",
            f"    generation-error  {d['generation_error_rate']:.4f}   wrong, gold retrieved",
        ]
        if self.n_unanswerable:
            lines += [
                "  Unanswerable questions",
                f"    abstained         {d['abstain_rate']:.4f}   ({self.abstained} of {self.n_unanswerable})",
            ]
        if self.by_category:
            lines += ["", f"  {'category':<16}{'n':>4}  {'ok':>6}  {'ret-miss':>9}  {'gen-err':>8}  {'precision':>9}"]
            for name, row in sorted(self.by_category.items()):
                lines.append(
                    f"  {name:<16}{row['n']:>4}  {row['ok']:>6.3f}  {row['retrieval_miss']:>9.3f}  "
                    f"{row['generation_error']:>8.3f}  {row['claim_precision']:>9.3f}"
                )
        return "\n".join(lines)


def _answer_covered(expected: str | None, generated: str, threshold: float) -> bool:
    """True when the generated answer contains enough of the expected answer's content words."""
    if not expected:
        return False
    need = content_tokens(expected)
    if not need:
        return False
    return len(need & content_tokens(generated)) / len(need) >= threshold


def diagnose_claims(
    retriever: Retriever,
    cases: list[EvalCase],
    *,
    k: int = 5,
    mode: RetrievalMode = "hybrid",
    tau: float = 0.6,
    answer_tau: float = 0.6,
    answer_fn: AnswerFn | None = None,
    aligner: Aligner | None = None,
) -> ClaimReport:
    """Retrieve, answer and attribute for every case.

    ``tau`` is the alignment score a chunk needs to ground a claim; ``answer_tau``
    is the share of the expected answer's content words the answer must contain to
    count as correct. Cases without ``relevant_doc_ids`` are treated as unanswerable
    only if ``answerable`` is False; an answerable case needs ``expected_answer``.
    """
    answer_fn = answer_fn or answer
    aligner = aligner or lexical_support

    claim_counts = dict.fromkeys(CLAIM_LABELS, 0)
    outcomes = dict.fromkeys(OUTCOMES, 0)
    cat: dict[str, dict] = {}
    per_case: list[dict] = []
    n_answerable = n_unanswerable = abstained = n_claims = 0

    for case in cases:
        chunks = retriever.retrieve(case.question, k=k, mode=mode)
        generated = answer_fn(case.question, chunks)
        gold = set(case.relevant_doc_ids)
        retrieved_docs = []
        for sc in chunks:
            if sc.chunk.doc_id not in retrieved_docs:
                retrieved_docs.append(sc.chunk.doc_id)

        abstain = is_abstention(generated.text)
        claims = [] if abstain else split_claims(generated.text)

        claim_rows = []
        counts = dict.fromkeys(CLAIM_LABELS, 0)
        for claim in claims:
            best_gold, gold_doc = 0.0, None
            best_other, other_doc = 0.0, None
            for sc in chunks:
                score = aligner(claim, sc.chunk.text)
                if sc.chunk.doc_id in gold:
                    if score > best_gold:
                        best_gold, gold_doc = score, sc.chunk.doc_id
                elif score > best_other:
                    best_other, other_doc = score, sc.chunk.doc_id
            if best_gold >= tau:
                label, support, source = "supported", best_gold, gold_doc
            elif best_other >= tau:
                label, support, source = "distractor", best_other, other_doc
            else:
                label, support, source = "unsupported", max(best_gold, best_other), None
            counts[label] += 1
            claim_rows.append({"claim": claim, "label": label, "support": round(support, 3), "source_doc": source})

        row: dict = {
            "question": case.question,
            "category": case.category,
            "answer": generated.text,
            "abstained": abstain,
            "retrieved_docs": retrieved_docs,
            "gold_docs": sorted(gold),
            "claims": claim_rows,
        }

        if not case.answerable:
            n_unanswerable += 1
            abstained += abstain
            row["outcome"] = "abstained" if abstain else "hallucinated"
        else:
            n_answerable += 1
            gold_found = gold <= set(retrieved_docs)
            correct = (not abstain) and _answer_covered(case.expected_answer, generated.text, answer_tau)
            if correct:
                outcome = "ok"
            elif not gold_found:
                outcome = "retrieval-miss"
            else:
                outcome = "generation-error"
            outcomes[outcome] += 1
            row["outcome"] = outcome
            if case.category:
                b = cat.setdefault(
                    case.category,
                    {"n": 0, "ok": 0, "retrieval-miss": 0, "generation-error": 0, "sup": 0, "claims": 0},
                )
                b["n"] += 1
                b[outcome] += 1
                b["sup"] += counts["supported"]
                b["claims"] += len(claims)

        for label in CLAIM_LABELS:
            claim_counts[label] += counts[label]
        n_claims += len(claims)
        per_case.append(row)

    by_category = {
        name: {
            "n": b["n"],
            "ok": round(b["ok"] / b["n"], 4),
            "retrieval_miss": round(b["retrieval-miss"] / b["n"], 4),
            "generation_error": round(b["generation-error"] / b["n"], 4),
            "claim_precision": round(b["sup"] / b["claims"], 4) if b["claims"] else 0.0,
        }
        for name, b in cat.items()
    }
    return ClaimReport(
        k=k,
        mode=mode,
        tau=tau,
        n_cases=n_answerable,
        n_unanswerable=n_unanswerable,
        n_claims=n_claims,
        claim_counts=claim_counts,
        outcomes=outcomes,
        abstained=abstained,
        by_category=by_category,
        per_case=per_case,
    )
