"""Sanity checks for a labelled eval set against its document folder.

Catches the mistakes that silently make a benchmark meaningless: labels that
point at documents that do not exist, evidence spans that are not actually in
the cited document, duplicate questions, and unanswerable cases that carry
labels. Used by ``scripts/validate_eval_set.py`` and the test suite.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

CATEGORIES = {"direct", "paraphrase", "disambiguation", "multihop", "unanswerable"}


def validate_eval_set(eval_path: str | Path, docs_dir: str | Path) -> tuple[list[str], dict]:
    """Return ``(errors, stats)``. An empty ``errors`` list means the set is consistent."""
    docs_dir = Path(docs_dir)
    docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(docs_dir.glob("*.md"))}
    errors: list[str] = []
    seen: set[str] = set()
    categories: Counter[str] = Counter()
    used_docs: set[str] = set()

    lines = Path(eval_path).read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        where = f"line {lineno}"
        question = row.get("question", "")
        if not question:
            errors.append(f"{where}: missing question")
        if question in seen:
            errors.append(f"{where}: duplicate question {question[:50]!r}")
        seen.add(question)

        category = row.get("category")
        categories[category or "(none)"] += 1
        if category not in CATEGORIES:
            errors.append(f"{where}: unknown category {category!r}")

        ids = row.get("relevant_doc_ids", [])
        if not row.get("answerable", True):
            if ids or row.get("evidence") or row.get("expected_answer"):
                errors.append(f"{where}: unanswerable case must have no ids, evidence or answer")
            continue

        if not ids:
            errors.append(f"{where}: answerable case has no relevant_doc_ids")
        if category == "multihop" and len(ids) < 2:
            errors.append(f"{where}: multihop case needs at least 2 relevant docs")
        if category in {"direct", "paraphrase", "disambiguation"} and len(ids) != 1:
            errors.append(f"{where}: {category} case needs exactly 1 relevant doc")
        for doc_id in ids:
            if doc_id not in docs:
                errors.append(f"{where}: relevant doc {doc_id!r} not found in {docs_dir}")
            used_docs.add(doc_id)

        evidence = row.get("evidence", [])
        if {e["doc_id"] for e in evidence} != set(ids):
            errors.append(f"{where}: evidence docs do not match relevant_doc_ids")
        for e in evidence:
            text = docs.get(e["doc_id"])
            if text is not None and e["span"] not in text:
                errors.append(f"{where}: evidence span not found verbatim in {e['doc_id']!r}")
        if not row.get("expected_answer"):
            errors.append(f"{where}: answerable case has no expected_answer")

    stats = {
        "n_docs": len(docs),
        "n_cases": sum(categories.values()),
        "categories": dict(categories),
        "docs_never_relevant": len(set(docs) - used_docs),
    }
    return errors, stats
