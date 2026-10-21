#!/usr/bin/env python
"""End-to-end check on a real document: the NIST AI Risk Management Framework 1.0.

Downloads NIST AI 100-1 (January 2023, 48 pages, a US-government work in the public
domain), indexes it, and asks ten questions with known outcomes: six the document
answers, three near misses (on-topic, but the fact is not in it) and one off-topic.
Each question runs under three setups so the effect of each abstention gate shows:
no gates, the retrieval-similarity gate, and the similarity gate plus the reader.

This is a spot check on one document, not a benchmark; see scripts/benchmark.py
for calibrated numbers.

Usage (needs the embed, reader and pdf extras; downloads ~200 MB of models once):
  python scripts/real_pdf_check.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lumen_rag import config
from lumen_rag.engine import RagEngine
from lumen_rag.ingestion.loaders import load_file

PDF_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
READER = "onnx-community/roberta-base-squad2-ONNX"

# (kind, question): "answerable", "near-miss" or "off-topic".
QUESTIONS = [
    ("answerable", "Who was the Secretary of Commerce when the AI RMF was published?"),
    ("answerable", "Which function is designed to be cross-cutting across the other three?"),
    ("answerable", "What does TEVV stand for?"),
    ("answerable", "Where can comments on the AI RMF Playbook be sent?"),
    ("answerable", "When is a formal review of the framework expected to take place?"),
    ("answerable", "What are the characteristics of trustworthy AI systems?"),
    ("near-miss", "When will version 2.0 of the AI RMF be released?"),
    ("near-miss", "How much does AI RMF certification cost?"),
    ("near-miss", "What fine does an organization pay for not following the AI RMF?"),
    ("off-topic", "What is the boiling point of water at sea level?"),
]

SETUPS = [
    ("no gates", None, ""),
    ("similarity >= 0.66", 0.66, ""),
    ("similarity >= 0.66, then reader >= -3.5", 0.66, READER),
]


def main() -> None:
    config.settings.embedder = "fastembed"
    config.settings.reader_threshold = -3.5
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "NIST.AI.100-1.pdf"
        print(f"Downloading {PDF_URL}")
        # nvlpubs.nist.gov refuses Python's default user agent.
        resp = httpx.get(PDF_URL, follow_redirects=True, timeout=60, headers={"User-Agent": "lumen-rag/0.1"})
        resp.raise_for_status()
        pdf.write_bytes(resp.content)
        engine = RagEngine()
        doc = load_file(pdf)
        total = engine.add_documents([doc])
        print(f"Indexed {pdf.name}: {total} chunks\n")

        summary = []
        for name, threshold, reader in SETUPS:
            config.settings.abstain_threshold = threshold
            config.settings.reader_model = reader
            right = 0
            print(f"=== {name}")
            for kind, question in QUESTIONS:
                start = time.perf_counter()
                result = engine.query(question, k=5, mode="vector")
                ms = (time.perf_counter() - start) * 1000
                ok = result.abstained == (kind != "answerable")
                right += ok
                sim = "-" if result.confidence is None else f"{result.confidence:.3f}"
                read = "-" if result.answerability is None else f"{result.answerability:+.1f}"
                verdict = "refused " if result.abstained else "answered"
                print(f"  [{'ok' if ok else 'XX'}] {kind:10} {verdict} sim={sim} reader={read} {ms:5.0f} ms  {question}")
            summary.append((name, right))
            print()

        print("| Setup | Right outcome |")
        print("|---|---|")
        for name, right in summary:
            print(f"| {name} | {right}/{len(QUESTIONS)} |")


if __name__ == "__main__":
    main()
