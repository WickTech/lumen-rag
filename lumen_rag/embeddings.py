"""Embedding providers.

Two implementations behind one interface:

* ``OpenAIEmbedder``    — real embeddings when an API key is present.
* ``HashingEmbedder``   — a deterministic, dependency-free fallback so the
                          whole engine (and its test suite) runs offline.

The hashing embedder is a bag-of-words feature hasher with L2 normalisation.
It is *not* semantically smart, but it is stable and good enough to exercise
ranking logic and the eval harness without network access.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, L2-normalised
        ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic feature-hashing embedder. No network, no model download."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                out[i, idx] += sign
        return _normalize(out)


class OpenAIEmbedder:
    """Wraps the OpenAI embeddings endpoint (or any compatible base URL)."""

    def __init__(self) -> None:
        from openai import OpenAI  # imported lazily; optional dependency

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = settings.embed_model
        self.dim = 1536  # text-embedding-3-small

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return _normalize(vecs)


def get_embedder() -> Embedder:
    """Pick the real embedder if configured, else the offline fallback."""
    if settings.offline:
        return HashingEmbedder()
    return OpenAIEmbedder()
