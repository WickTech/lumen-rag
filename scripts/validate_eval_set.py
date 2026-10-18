#!/usr/bin/env python
"""Validate an eval set against its docs folder; exit 1 if anything is inconsistent.

Usage:
  python scripts/validate_eval_set.py                       # data/eval_v2
  python scripts/validate_eval_set.py EVAL.jsonl DOCS_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lumen_rag.eval.validate import validate_eval_set


def main() -> int:
    if len(sys.argv) == 3:
        eval_path, docs_dir = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        eval_path = ROOT / "data" / "eval_v2" / "eval.jsonl"
        docs_dir = ROOT / "data" / "eval_v2" / "docs"
    errors, stats = validate_eval_set(eval_path, docs_dir)
    print(f"docs={stats['n_docs']} cases={stats['n_cases']} categories={stats['categories']}")
    for err in errors:
        print("  ERROR", err, file=sys.stderr)
    if errors:
        print(f"{len(errors)} problem(s) found", file=sys.stderr)
        return 1
    print("eval set is consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
