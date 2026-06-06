#!/usr/bin/env python
"""Eval regression guard: fail with a non-zero exit code if any metric drops below threshold.

Usage:
  python scripts/check_eval.py data/eval.jsonl --k 3

The thresholds below represent a floor derived from the baseline offline run.
If scores *drop* below them, this script exits 1 so CI fails. Raise the
thresholds when you improve the pipeline; never lower them to pass CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root so the script works without install when run from CI.
sys.path.insert(0, str(Path(__file__).parent.parent))

from lumen_rag.engine import RagEngine
from lumen_rag.eval import evaluate
from lumen_rag.eval.harness import load_cases
from lumen_rag.retrieval import Retriever

# Minimum acceptable scores. Adjust upward as the pipeline improves.
THRESHOLDS: dict[str, float] = {
    "recall@k": 0.80,
    "hit_rate": 0.80,
    "mrr": 0.70,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Lumen RAG eval regression guard")
    parser.add_argument("dataset", help="Path to JSONL eval set")
    parser.add_argument("--k", type=int, default=3, help="Top-k to evaluate")
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["vector", "bm25", "hybrid"],
        help="Retrieval mode",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Override index directory (defaults to LUMEN_INDEX_DIR or .lumen_index)",
    )
    args = parser.parse_args()

    engine = RagEngine.load(args.index_dir) if args.index_dir else RagEngine.load()
    cases = load_cases(args.dataset)
    report = evaluate(Retriever(engine.store, engine.embedder), cases, k=args.k)

    scores = report.as_dict()
    print(f"\n  Retrieval eval — {scores['n_cases']} cases @ k={scores['k']}")
    print("  " + "-" * 34)
    for key in ("recall@k", "precision@k", "mrr", "ndcg@k", "hit_rate"):
        threshold = THRESHOLDS.get(key)
        status = ""
        if threshold is not None:
            status = " ✓" if scores[key] >= threshold else f" ✗  (threshold {threshold})"
        print(f"  {key:<14} {scores[key]:.4f}{status}")

    failures = [
        f"{key}={scores[key]:.4f} < threshold {thr}"
        for key, thr in THRESHOLDS.items()
        if scores[key] < thr
    ]

    if failures:
        print("\n  REGRESSION DETECTED:", ", ".join(failures), file=sys.stderr)
        return 1

    print("\n  All thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
