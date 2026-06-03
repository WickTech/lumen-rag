"""Sentence-aware text chunking with overlap.

Splits on sentence boundaries and packs sentences into ~`chunk_size`-word
windows with `overlap` words carried into the next chunk. Overlap preserves
context across boundaries so an answer that straddles two chunks is still
retrievable.
"""
from __future__ import annotations

import re

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in _SENT_RE.split(text) if s]


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Return a list of overlapping chunks, each up to ~chunk_size words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    count = 0

    for sentence in _sentences(text):
        words = sentence.split()
        if count + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # carry the last `overlap` words into the next window
            carry = current[-overlap:] if overlap else []
            current = list(carry)
            count = len(carry)
        current.extend(words)
        count += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks
