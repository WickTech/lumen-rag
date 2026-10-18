#!/usr/bin/env python
"""Benchmark retrieval quality across pipeline configurations.

Runs a labelled eval set through several configurations and prints a markdown
table (paste straight into the README) plus a per-category breakdown. Results
are also written to a JSON file next to the repo root.

Eval sets (``--set``):

  v1  data/docs (7 files) + data/eval.jsonl (19 questions). Tiny and saturated:
      k=5 returns 5 of the 7 documents, so recall is near 1.0 for any config.
      Kept as a fast smoke test.
  v2  data/eval_v2/docs (120 short docs) + data/eval_v2/eval.jsonl (176
      questions: direct, paraphrase, disambiguation, multihop, unanswerable).
      Discriminates between configurations; see data/eval_v2/README.md.

Configurations compare chunking and retrieval mode. The retrieval mode is
applied per config (vector | bm25 | hybrid), with the cross-encoder reranker
disabled so the numbers are reproducible offline.

Usage:
  python scripts/benchmark.py            # v1
  python scripts/benchmark.py --set v2
"""
from __future__ import annotations

import argparse
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

K = 5

SETS = {
    "v1": {
        "docs": ROOT / "data" / "docs",
        "eval": ROOT / "data" / "eval.jsonl",
        "out": ROOT / "benchmark_results.json",
        "configs": [
            {"name": "naive (1 chunk/doc, vector-only)", "chunk_size": 100_000, "overlap": 0, "mode": "vector"},
            {"name": "+ sentence chunking (vector-only)", "chunk_size": 120, "overlap": 20, "mode": "vector"},
            {"name": "BM25 only (sentence chunking)", "chunk_size": 120, "overlap": 20, "mode": "bm25"},
            {"name": "hybrid (BM25 + RRF, sentence chunking)", "chunk_size": 120, "overlap": 20, "mode": "hybrid"},
        ],
    },
    "v2": {
        "docs": ROOT / "data" / "eval_v2" / "docs",
        "eval": ROOT / "data" / "eval_v2" / "eval.jsonl",
        "out": ROOT / "benchmark_results_v2.json",
        "configs": [
            {"name": "vector-only", "chunk_size": 120, "overlap": 20, "mode": "vector"},
            {"name": "BM25 only", "chunk_size": 120, "overlap": 20, "mode": "bm25"},
            {"name": "hybrid (BM25 + RRF)", "chunk_size": 120, "overlap": 20, "mode": "hybrid"},
        ],
    },
}


def build_retriever(docs_dir: Path, chunk_size: int, overlap: int) -> Retriever:
    embedder = get_embedder()
    store = VectorStore(dim=embedder.dim)
    docs = [load_file(p) for p in sorted(docs_dir.iterdir()) if p.suffix.lower() in _LOADERS]
    ingest_documents(docs, store=store, embedder=embedder, chunk_size=chunk_size, overlap=overlap)
    # reranker=None: keep the benchmark offline and reproducible.
    return Retriever(store, embedder, reranker=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lumen RAG retrieval benchmark")
    parser.add_argument("--set", dest="which", choices=sorted(SETS), default="v1")
    args = parser.parse_args()
    spec = SETS[args.which]

    cases = load_cases(spec["eval"])
    rows = []
    reports = []
    for cfg in spec["configs"]:
        retriever = build_retriever(spec["docs"], cfg["chunk_size"], cfg["overlap"])
        report = evaluate(retriever, cases, k=K, mode=cfg["mode"])
        rows.append({"config": cfg["name"], "mode": cfg["mode"], **report.as_dict(), "by_category": report.by_category})
        reports.append((cfg["name"], report))

    header = ["Configuration", f"recall@{K}", f"precision@{K}", "MRR", f"nDCG@{K}", "hit rate"]
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
    print("\n".join(lines))

    categories = sorted({c for r in rows for c in r["by_category"]})
    if categories:
        print("\nMRR by question category\n")
        head = ["Configuration", *categories]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for r in rows:
            cells = [f"{r['by_category'][c]['mrr']:.2f}" if c in r["by_category"] else "-" for c in categories]
            print("| " + " | ".join([r["config"], *cells]) + " |")

    out = spec["out"]
    out.write_text(
        json.dumps(
            {
                "set": args.which,
                "k": K,
                "n_cases": reports[0][1].n_cases,
                "n_unanswerable": reports[0][1].n_unanswerable,
                "results": rows,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
