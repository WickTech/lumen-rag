#!/usr/bin/env python
"""Score the retrieval stack on BEIR SciFact, a public benchmark.

SciFact: 5,183 scientific abstracts, 300 test claims, graded relevance judgements
(``test.tsv``; any score > 0 counts as relevant). One chunk per abstract (title and
abstract text), so chunk-level retrieval equals document retrieval. The data is
downloaded from the Hugging Face Hub at run time and cached; nothing is committed.

Metrics follow BEIR: nDCG@10 and recall@100. The reranked row reorders a pool of
``k * LUMEN_RERANK_POOL_FACTOR`` candidates, so recall@100 is not reported for it.

Usage:
  python scripts/beir_eval.py                                  # hashing rows only
  python scripts/beir_eval.py --embedder fastembed --rerank    # adds bge-small + rerank
  python scripts/beir_eval.py --limit 50                       # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lumen_rag.eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from lumen_rag.ingestion.pipeline import ingest_documents  # noqa: E402
from lumen_rag.retrieval import Retriever  # noqa: E402
from lumen_rag.store import VectorStore  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import RERANKER, make_embedder  # noqa: E402

NDCG_K, RECALL_K = 10, 100
ONE_CHUNK = 1_000_000  # words: never splits an abstract
OUT = ROOT / "benchmark_results_beir_scifact.json"


def load_scifact() -> tuple[list[dict], dict[str, str], dict[str, set[str]]]:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("BEIR evaluation needs the 'beir' extra: pip install 'lumen-rag[beir]'") from exc

    def fetch(repo: str, name: str) -> str:
        return hf_hub_download(repo, name, repo_type="dataset")

    corpus = pq.read_table(fetch("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet")).to_pylist()
    queries = pq.read_table(fetch("BeIR/scifact", "queries/queries-00000-of-00001.parquet")).to_pylist()
    docs = [
        {"id": str(r["_id"]), "text": f"{r['title']}. {r['text']}" if r["title"] else r["text"]}
        for r in corpus
    ]
    query_text = {str(r["_id"]): r["text"] for r in queries}
    qrels: dict[str, set[str]] = {}
    for line in Path(fetch("BeIR/scifact-qrels", "test.tsv")).read_text().splitlines()[1:]:
        qid, did, score = line.split("\t")
        if int(score) > 0:
            qrels.setdefault(qid, set()).add(did)
    return docs, query_text, qrels


def score(retriever: Retriever, mode: str, queries: list[tuple[str, set[str]]], rerank: bool) -> dict:
    ndcg, recall = [], []
    start = time.perf_counter()
    for text, relevant in queries:
        k = NDCG_K if rerank else RECALL_K
        ids: list[str] = []
        for sc in retriever.retrieve(text, k=k, mode=mode):
            if sc.chunk.doc_id not in ids:
                ids.append(sc.chunk.doc_id)
        ndcg.append(ndcg_at_k(ids, relevant, NDCG_K))
        recall.append(recall_at_k(ids, relevant, RECALL_K))
    n = len(queries)
    return {
        "ndcg@10": round(sum(ndcg) / n, 4),
        "recall@100": None if rerank else round(sum(recall) / n, 4),
        "ms_per_query": round((time.perf_counter() - start) * 1000 / n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lumen RAG on BEIR SciFact")
    parser.add_argument("--embedder", choices=["hashing", "fastembed", "openai"], default="hashing")
    parser.add_argument("--rerank", action="store_true", help=f"add a hybrid + {RERANKER} row")
    parser.add_argument("--limit", type=int, default=0, help="only the first N test queries")
    args = parser.parse_args()

    docs, query_text, qrels = load_scifact()
    queries = [(query_text[q], rel) for q, rel in sorted(qrels.items())]
    if args.limit:
        queries = queries[: args.limit]
    print(f"SciFact: {len(docs)} abstracts, {len(queries)} test queries, embedder {args.embedder}")

    embedder = make_embedder(args.embedder)
    start = time.perf_counter()
    store = ingest_documents(
        docs, store=VectorStore(dim=embedder.dim), embedder=embedder,
        chunk_size=ONE_CHUNK, overlap=0, contextual=False,
    )
    index_s = round(time.perf_counter() - start, 1)

    rows = []
    plain = Retriever(store, embedder, reranker=None)
    for name, mode in (("vector-only", "vector"), ("BM25 only", "bm25"), ("hybrid (BM25 + RRF)", "hybrid")):
        rows.append({"config": name, **score(plain, mode, queries, rerank=False)})
    if args.rerank:
        from lumen_rag.retrieval.rerank import CrossEncoderReranker

        reranked = Retriever(store, embedder, reranker=CrossEncoderReranker(RERANKER))
        rows.append({"config": "hybrid + rerank", **score(reranked, "hybrid", queries, rerank=True)})

    print(f"\nindexing: {index_s}s\n")
    print("| Configuration | nDCG@10 | recall@100 | ms/query |\n|---|---|---|---|")
    for r in rows:
        rec = "-" if r["recall@100"] is None else f"{r['recall@100']:.3f}"
        print(f"| {r['config']} | {r['ndcg@10']:.3f} | {rec} | {r['ms_per_query']:.1f} |")

    out = OUT if args.embedder == "hashing" else OUT.with_name(f"{OUT.stem}_{args.embedder}.json")
    if not args.limit:
        out.write_text(json.dumps({
            "dataset": "BeIR/scifact", "n_docs": len(docs), "n_queries": len(queries),
            "embedder": args.embedder, "index_seconds": index_s, "results": rows,
        }, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
