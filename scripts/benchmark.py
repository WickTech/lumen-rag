#!/usr/bin/env python
"""Benchmark retrieval quality across pipeline configurations.

Runs a labelled eval set through several configurations and prints a markdown
table (paste straight into the README) plus a per-category breakdown. Results
are also written to a JSON file next to the repo root.

Eval sets (``--set``):

  v1  data/docs (7 files) + data/eval.jsonl (19 questions). Tiny and saturated:
      k=5 returns 5 of the 7 documents, so recall is near 1.0 for any config.
      Kept as a fast smoke test.
  v2  data/eval_v2/docs (120 short docs) + data/eval_v2/eval.jsonl (216
      questions: direct, paraphrase, disambiguation, multihop, unanswerable).
      Discriminates between configurations; see data/eval_v2/README.md.

Configurations compare chunking and retrieval mode. The retrieval mode is
applied per config (vector | bm25 | hybrid), with the cross-encoder reranker
disabled unless ``--rerank`` adds a "hybrid + rerank" row.

``--embedder`` picks the dense model (hashing is the offline default; fastembed
runs BAAI/bge-small-en-v1.5 locally; openai needs OPENAI_API_KEY). Results for a
non-default embedder go to a file suffixed with its name.

``--abstain`` also calibrates abstention on each configuration: threshold chosen
on half the questions, refusal and false-refusal rates reported on the other half.

``--reader MODEL`` (with ``--abstain``) compares answerability signals on the
first configuration: retrieval similarity, an extractive SQuAD 2.0 reader, and
the two chained (similarity gate first, then the reader).

Usage:
  python scripts/benchmark.py            # v1
  python scripts/benchmark.py --set v2
  python scripts/benchmark.py --set v2 --embedder fastembed --rerank --abstain
  python scripts/benchmark.py --set v2 --embedder fastembed --abstain \
      --reader onnx-community/tinyroberta-squad2-ONNX
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lumen_rag.embeddings import FastEmbedEmbedder, HashingEmbedder, OpenAIEmbedder
from lumen_rag.eval import evaluate
from lumen_rag.eval.abstention import calibrate, gate_together, outcome, score_cases, split
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


# Small, fast cross-encoder so the benchmark runs on a laptop CPU in minutes;
# override with LUMEN_BENCH_RERANKER (e.g. BAAI/bge-reranker-base).
SMALL_CHUNK, SMALL_OVERLAP = 40, 10
RERANKER = os.getenv("LUMEN_BENCH_RERANKER", "Xenova/ms-marco-MiniLM-L-6-v2")


def make_embedder(name: str):
    if name == "hashing":
        return HashingEmbedder()
    if name == "fastembed":
        return FastEmbedEmbedder(os.getenv("LUMEN_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    if name == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("--embedder openai needs OPENAI_API_KEY")
        return OpenAIEmbedder()
    raise ValueError(name)


def build_store(docs_dir: Path, chunk_size: int, overlap: int, embedder, contextual: bool = False) -> VectorStore:
    store = VectorStore(dim=embedder.dim)
    docs = [load_file(p) for p in sorted(docs_dir.iterdir()) if p.suffix.lower() in _LOADERS]
    ingest_documents(
        docs, store=store, embedder=embedder, chunk_size=chunk_size, overlap=overlap, contextual=contextual
    )
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Lumen RAG retrieval benchmark")
    parser.add_argument("--set", dest="which", choices=sorted(SETS), default="v1")
    parser.add_argument("--embedder", choices=["hashing", "fastembed", "openai"], default="hashing")
    parser.add_argument("--rerank", action="store_true", help=f"add a hybrid + {RERANKER} row")
    parser.add_argument("--abstain", action="store_true", help="calibrate and report abstention")
    parser.add_argument("--decompose", action="store_true", help="add rows with query decomposition")
    parser.add_argument(
        "--contextual",
        action="store_true",
        help="add small-chunk rows with and without contextual chunk headers",
    )
    parser.add_argument("--reader", default="", help="with --abstain: SQuAD 2.0 ONNX reader to compare")
    args = parser.parse_args()
    spec = SETS[args.which]
    embedder = make_embedder(args.embedder)

    configs = list(spec["configs"])
    if args.rerank:
        base = configs[-1]
        configs.append({**base, "name": f"{base['name']} + rerank", "rerank": True})
        # Dense + rerank too: with a strong embedder, dense alone can beat the fusion.
        dense = configs[0]
        configs.append({**dense, "name": f"{dense['name']} + rerank", "rerank": True})

    if args.contextual:
        # v2 documents fit in one default chunk, so their title is always in it.
        # Small chunks are where a chunk loses its document; compare them with and
        # without a "Document: <title>. Section: ..." header.
        small = {"chunk_size": SMALL_CHUNK, "overlap": SMALL_OVERLAP}
        configs += [
            {**c, **small, "name": f"{c['name']} ({SMALL_CHUNK}-word chunks){label}", "contextual": ctx}
            for c in list(spec["configs"])
            for ctx, label in ((False, ""), (True, " + headers"))
        ]

    if args.decompose:
        # Same retrieval, compound questions split into sub-queries fused by RRF.
        configs += [
            {**c, "name": f"{c['name']} + decomposition", "decompose": True}
            for c in list(configs)
            if c["mode"] != "bm25"
        ]

    cases = load_cases(spec["eval"])
    n_queries = sum(c.answerable for c in cases)
    stores: dict[tuple, VectorStore] = {}  # embed the corpus once per chunking
    reranker = None
    rows = []
    reports = []
    for cfg in configs:
        key = (cfg["chunk_size"], cfg["overlap"], cfg.get("contextual", False))
        if key not in stores:
            stores[key] = build_store(spec["docs"], *key[:2], embedder, contextual=key[2])
        if cfg.get("rerank") and reranker is None:
            from lumen_rag.retrieval.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(RERANKER)
        retriever = Retriever(
            stores[key],
            embedder,
            reranker=reranker if cfg.get("rerank") else None,
            decompose=cfg.get("decompose", False),
        )
        start = time.perf_counter()
        report = evaluate(retriever, cases, k=K, mode=cfg["mode"])
        latency_ms = (time.perf_counter() - start) * 1000 / max(1, n_queries)
        row = {
            "config": cfg["name"],
            "mode": cfg["mode"],
            **report.as_dict(),
            "latency_ms": round(latency_ms, 1),
            "by_category": report.by_category,
        }
        if args.abstain:
            row["abstention"] = calibrate(score_cases(retriever, cases, k=K, mode=cfg["mode"]))
        rows.append(row)
        reports.append((cfg["name"], report))

    reader_section = None
    if args.reader and args.abstain:
        from lumen_rag.reader import ExtractiveReader

        reader = ExtractiveReader(args.reader)
        cfg0 = configs[0]
        retriever = Retriever(
            stores[(cfg0["chunk_size"], cfg0["overlap"], False)], embedder, reranker=None, decompose=False
        )
        sim = score_cases(retriever, cases, k=K, mode=cfg0["mode"])
        start = time.perf_counter()
        read = score_cases(retriever, cases, k=K, mode=cfg0["mode"], reader=reader)
        read_ms = (time.perf_counter() - start) * 1000 / len(cases)
        sim_cal = calibrate(sim)
        both = gate_together(sim, sim_cal["conservative"]["threshold"], read)
        reader_section = {
            "model": args.reader,
            "config": cfg0["name"],
            "ms_per_query_with_retrieval": round(read_ms, 1),
            "signals": {
                "retrieval similarity": sim_cal,
                "reader": calibrate(read),
                "similarity gate, then reader": calibrate(both),
            },
            # SQuAD 2.0's own decision rule: answer when the best span beats "no answer".
            "squad_threshold_0_test": {
                "reader": outcome(split(read)[1], 0.0),
                "similarity gate, then reader": outcome(split(both)[1], 0.0),
            },
        }

    print(f"Embedder: {args.embedder}\n")
    header = ["Configuration", f"recall@{K}", f"precision@{K}", "MRR", f"nDCG@{K}", "hit rate", "ms/query"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for r in rows:
        lines.append(
            "| {config} | {recall@k:.2f} | {precision@k:.2f} | {mrr:.2f} | {ndcg@k:.2f} | {hit_rate:.2f} "
            "| {latency_ms:.1f} |".format(**r)
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

    if args.abstain:
        print("\nAbstention (thresholds chosen on the dev half, rates on the held-out half)\n")
        head = [
            "Configuration", "AUROC", "operating point", "threshold", "unanswerable refused",
            "answerable wrongly refused", "correct outcome", "correct outcome, no abstention",
        ]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for r in rows:
            a = r["abstention"]
            base = a["no_abstention_test"]["correct_outcome_rate"]
            points = [
                ("max Youden J", a["threshold"], a["test"]),
                (
                    f"≤{a['conservative']['max_false_refusal']:.0%} false refusals",
                    a["conservative"]["threshold"],
                    a["conservative"]["test"],
                ),
            ]
            for label, thr, t in points:
                print(
                    f"| {r['config']} | {a['auroc_test']:.2f} | {label} | {thr:.3f} "
                    f"| {t['refusal_rate_unanswerable']:.2f} | {t['false_refusal_rate']:.2f} "
                    f"| {t['correct_outcome_rate']:.2f} | {base:.2f} |"
                )
        first = rows[0]["abstention"]
        print(f"\n(held-out half: {first['n_test']} questions, {first['n_unanswerable_test']} unanswerable)")

    if reader_section:
        print(
            f"\nAnswerability signals on {reader_section['config']} (reader: {reader_section['model']}, "
            f"{reader_section['ms_per_query_with_retrieval']:.0f} ms/query incl. retrieval)\n"
        )
        head = [
            "Signal", "AUROC (held-out)", "AUROC near-miss", "AUROC off-topic", "operating point",
            "threshold", "unanswerable refused", "answerable wrongly refused", "correct outcome",
        ]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for name, a in reader_section["signals"].items():
            points = [
                ("max Youden J", a["threshold"], a["test"]),
                ("≤5% false refusals", a["conservative"]["threshold"], a["conservative"]["test"]),
            ]
            if name in reader_section["squad_threshold_0_test"]:
                points.append(("SQuAD rule (0)", 0.0, reader_section["squad_threshold_0_test"][name]))
            for label, thr, t in points:
                print(
                    f"| {name} | {a['auroc_test']:.2f} | {a['auroc_near_miss']:.2f} | {a['auroc_off_topic']:.2f} "
                    f"| {label} | {thr:.3f} | {t['refusal_rate_unanswerable']:.2f} "
                    f"| {t['false_refusal_rate']:.2f} | {t['correct_outcome_rate']:.2f} |"
                )

    out = spec["out"]
    if args.embedder != "hashing":
        out = out.with_name(f"{out.stem}_{args.embedder}{out.suffix}")
    out.write_text(
        json.dumps(
            {
                "set": args.which,
                "k": K,
                "embedder": args.embedder,
                "n_cases": reports[0][1].n_cases,
                "n_unanswerable": reports[0][1].n_unanswerable,
                "results": rows,
                "answerability": reader_section,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
