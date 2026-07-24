#!/usr/bin/env python
"""Benchmark retrieval quality across pipeline configurations.

Runs the same labelled eval set (data/eval.jsonl, 20 docs / 19 questions)
through three configurations to quantify what chunking and hybrid retrieval
actually buy you:

  1. naive       — whole document as one chunk, vector search only
  2. + chunking   — sentence-aware chunking (size=120, overlap=20), vector only
  3. + hybrid     — same chunking, + BM25 and Reciprocal Rank Fusion

Prints a markdown table (paste straight into the README) and writes
benchmark_results.json alongside it.

Usage:
  python scripts/benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lumen_rag.embeddings import get_embedder
from lumen_rag.eval import evaluate
from lumen_rag.eval.harness import load_cases
from lumen_rag.ingestion.loaders import _LOADERS, load_file
from lumen_rag.ingestion.pipeline import ingest_documents
from lumen_rag.retrieval import Retriever
from lumen_rag.store import VectorStore

DOCS_DIR = ROOT / "data" / "docs"
EVAL_PATH = ROOT / "data" / "eval.jsonl"
K = 5

CONFIGS = [
    {"name": "naive (1 chunk/doc, vector-only)", "chunk_size": 100_000, "overlap": 0, "mode": "vector"},
    {"name": "+ sentence chunking (vector-only)", "chunk_size": 120, "overlap": 20, "mode": "vector"},
    {"name": "+ hybrid (BM25 + RRF)", "chunk_size": 120, "overlap": 20, "mode": "hybrid"},
]


def build_retriever(chunk_size: int, overlap: int) -> Retriever:
    embedder = get_embedder()
    store = VectorStore(dim=embedder.dim)
    docs = [load_file(p) for p in sorted(DOCS_DIR.iterdir()) if p.suffix.lower() in _LOADERS]
    ingest_documents(docs, store=store, embedder=embedder, chunk_size=chunk_size, overlap=overlap)
    return Retriever(store, embedder)


def main() -> None:
    cases = load_cases(EVAL_PATH)
    rows = []
    for cfg in CONFIGS:
        retriever = build_retriever(cfg["chunk_size"], cfg["overlap"])
        report = evaluate(retriever, cases, k=K)
        d = report.as_dict()
        rows.append({"config": cfg["name"], **d})

    header = ["Configuration", "recall@5", "precision@5", "MRR", "nDCG@5", "hit rate"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for r in rows:
        lines.append(
            "| {config} | {recall@k:.2f} | {precision@k:.2f} | {mrr:.2f} | {ndcg@k:.2f} | {hit_rate:.2f} |".format(
                **r
            )
        )
    table = "\n".join(lines)
    print(table)

    out = ROOT / "benchmark_results.json"
    out.write_text(json.dumps({"k": K, "n_cases": len(cases), "results": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
