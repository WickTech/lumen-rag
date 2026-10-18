#!/usr/bin/env python
"""Retrieval eval report for pull requests.

Two subcommands:

  run      Measure the v2 eval set (vector, bm25, hybrid; reranker off, offline
           hashing embedder) for the code and data under ``--root`` and write JSON.
  compare  Diff a base run against a head run, print a markdown report, and exit 1
           if any tracked metric dropped by more than ``--max-drop``.

``run`` only uses the package under ``--root``, so CI can measure the pull-request
base commit (checked out in a separate worktree) with this script from the head
commit. Retrieval is deterministic offline, so any drop beyond the tolerance is
a real change, not noise.

Usage:
  python scripts/eval_report.py run --root . --out head.json
  python scripts/eval_report.py run --root ../base --out base.json
  python scripts/eval_report.py compare --base base.json --head head.json --out report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

MARKER = "<!-- lumen-eval-report -->"
MODES = ("vector", "bm25", "hybrid")
TRACKED = ("recall@k", "mrr", "ndcg@k")
LABELS = {"recall@k": "recall@5", "mrr": "MRR", "ndcg@k": "nDCG@5"}
K = 5


def data_fingerprint(root: Path) -> str:
    """Hash of the eval set and its documents, to tell whether the data changed."""
    data = root / "data" / "eval_v2"
    h = hashlib.sha256()
    for path in sorted((data / "docs").glob("*.md")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update((data / "eval.jsonl").read_bytes())
    return h.hexdigest()[:16]


def run_benchmark(root: Path, k: int = K) -> dict:
    """Measure the v2 set using the lumen_rag package found under ``root``."""
    root = root.resolve()
    # Deterministic and offline: no API embedder, no cross-encoder.
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("LUMEN_RERANKER", None)
    sys.path.insert(0, str(root))

    from lumen_rag.embeddings import get_embedder
    from lumen_rag.eval import evaluate
    from lumen_rag.eval.harness import load_cases
    from lumen_rag.ingestion.loaders import load_file
    from lumen_rag.ingestion.pipeline import ingest_documents
    from lumen_rag.retrieval import Retriever
    from lumen_rag.store import VectorStore

    import lumen_rag

    package_dir = Path(lumen_rag.__file__).resolve().parent
    if root not in package_dir.parents:
        raise RuntimeError(f"measured {package_dir}, which is not under --root {root}")

    data = root / "data" / "eval_v2"
    cases = load_cases(data / "eval.jsonl")
    embedder = get_embedder()
    store = VectorStore(dim=embedder.dim)
    docs = [load_file(p) for p in sorted((data / "docs").glob("*.md"))]
    ingest_documents(docs, store=store, embedder=embedder, chunk_size=120, overlap=20)
    retriever = Retriever(store, embedder, reranker=None)

    modes: dict[str, dict] = {}
    n_cases = n_unanswerable = 0
    for mode in MODES:
        report = evaluate(retriever, cases, k=k, mode=mode)
        n_cases, n_unanswerable = report.n_cases, report.n_unanswerable
        d = report.as_dict()
        modes[mode] = {
            "overall": {key: d[key] for key in ("recall@k", "precision@k", "mrr", "ndcg@k", "hit_rate")},
            "by_category": report.by_category,
        }
    return {
        "k": k,
        "n_cases": n_cases,
        "n_unanswerable": n_unanswerable,
        "data_fingerprint": data_fingerprint(root),
        "modes": modes,
    }


def _fmt_delta(delta: float, drop_limit: float) -> str:
    if abs(delta) < 0.0005:
        return "0.000"
    mark = "🔻" if delta < -drop_limit else ("▲" if delta > 0 else "▼")
    return f"{delta:+.3f} {mark}"


def compare(base: dict | None, head: dict, max_drop: float = 0.01) -> tuple[str, list[str], bool]:
    """Return ``(markdown, regressions, comparable)``.

    ``comparable`` is False when there is no baseline or the eval data changed
    between base and head; in that case deltas are shown but never gate a merge.
    """
    lines = [MARKER, "## Retrieval eval report", ""]
    regressions: list[str] = []

    if base is None:
        lines += [
            "⚠️ No baseline available (the base commit could not be measured), so there is nothing to compare against.",
            "",
        ]
        comparable = False
    elif base.get("data_fingerprint") != head.get("data_fingerprint"):
        lines += [
            "⚠️ The eval set or its documents changed in this pull request, so the numbers below are **not "
            "comparable** with the base branch and do not block the merge.",
            "",
        ]
        comparable = False
    else:
        comparable = True

    if comparable:
        for mode in MODES:
            for key in TRACKED:
                b = base["modes"][mode]["overall"][key]
                h = head["modes"][mode]["overall"][key]
                if b - h > max_drop:
                    regressions.append(f"{mode} {LABELS[key]}: {b:.3f} -> {h:.3f} ({h - b:+.3f})")
        if regressions:
            lines += [f"❌ **{len(regressions)} metric(s) dropped by more than {max_drop:g}:**", ""]
            lines += [f"- {r}" for r in regressions]
            lines += [""]
        else:
            lines += [f"✅ No metric dropped by more than {max_drop:g}.", ""]

    n = head["n_cases"]
    lines += [f"{n} answerable questions (+ {head['n_unanswerable']} unanswerable, not scored), k={head['k']}, offline hashing embedder, no reranker.", ""]

    show_base = base is not None
    header = ["Mode", "Metric"] + (["Base", "Head", "Δ"] if show_base else ["Head"])
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for mode in MODES:
        for key in TRACKED:
            h = head["modes"][mode]["overall"][key]
            if show_base:
                b = base["modes"][mode]["overall"][key]
                lines.append(f"| {mode} | {LABELS[key]} | {b:.3f} | {h:.3f} | {_fmt_delta(h - b, max_drop)} |")
            else:
                lines.append(f"| {mode} | {LABELS[key]} | {h:.3f} |")
    lines.append("")

    categories = sorted({c for m in MODES for c in head["modes"][m]["by_category"]})
    lines += ["<details><summary>MRR by question category</summary>", ""]
    cat_header = ["Mode", "Category"] + (["Base", "Head", "Δ"] if show_base else ["Head"])
    lines += ["| " + " | ".join(cat_header) + " |", "|" + "---|" * len(cat_header)]
    for mode in MODES:
        for cat in categories:
            h = head["modes"][mode]["by_category"].get(cat, {}).get("mrr")
            if h is None:
                continue
            if show_base:
                b = base["modes"][mode]["by_category"].get(cat, {}).get("mrr")
                if b is None:
                    lines.append(f"| {mode} | {cat} | - | {h:.3f} | - |")
                else:
                    lines.append(f"| {mode} | {cat} | {b:.3f} | {h:.3f} | {_fmt_delta(h - b, max_drop)} |")
            else:
                lines.append(f"| {mode} | {cat} | {h:.3f} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines), regressions, comparable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="measure the v2 eval set for the code under --root")
    run.add_argument("--root", default=".", help="repository root to measure (default: .)")
    run.add_argument("--out", required=True, help="JSON file to write")
    run.add_argument("--k", type=int, default=K)

    cmp_ = sub.add_parser("compare", help="diff base and head runs; exit 1 on regression")
    cmp_.add_argument("--base", help="base JSON (optional; omit if no baseline)")
    cmp_.add_argument("--head", required=True, help="head JSON")
    cmp_.add_argument("--max-drop", type=float, default=0.01, help="largest allowed absolute drop per metric")
    cmp_.add_argument("--out", help="write the markdown report here as well as stdout")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        result = run_benchmark(Path(args.root), k=args.k)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out} ({result['n_cases']} answerable cases)")
        return 0

    head = json.loads(Path(args.head).read_text(encoding="utf-8"))
    base_path = Path(args.base) if args.base else None
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path and base_path.exists() else None
    markdown, regressions, _ = compare(base, head, args.max_drop)
    print(markdown)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
