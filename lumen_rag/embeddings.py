"""Embedding providers.

Three implementations behind one interface, picked by ``LUMEN_EMBEDDER``:

* ``OpenAIEmbedder``    — hosted embeddings (``openai``; the ``auto`` default when an API key is present).
* ``FastEmbedEmbedder`` — local ONNX embeddings, ``BAAI/bge-small-en-v1.5`` by default
                          (``fastembed``; needs the ``[embed]`` extra, no torch, no API key).
* ``HashingEmbedder``   — a deterministic, dependency-free fallback so the
                          whole engine (and its test suite) runs offline.

The hashing embedder is a bag-of-words feature hasher with L2 normalisation.
It is *not* semantically smart, but it is stable and good enough to exercise
ranking logic and the eval harness without network access.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from .config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, L2-normalised
        ...


def embed_query(embedder: Embedder, text: str) -> np.ndarray:
    """Embed one search query, using the embedder's query-side encoding when it has one.

    Asymmetric models (bge, e5) encode queries and passages differently; the others
    embed both the same way.
    """
    fn = getattr(embedder, "embed_query", None)
    if fn is not None:
        return fn(text)
    return embedder.embed([text])[0]


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
        self.dim = _OPENAI_DIMS.get(self._model, 1536)

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return _normalize(vecs)


_OPENAI_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}


class FastEmbedEmbedder:
    """Local ONNX embeddings through fastembed. Downloads the model once, then runs offline."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "fastembed is required for LUMEN_EMBEDDER=fastembed. "
                "Install it with: pip install 'lumen-rag[embed]'"
            ) from e

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dim = int(self._model.embedding_size)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.array(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self._model.query_embed([text])), dtype=np.float32))[0]


@lru_cache(maxsize=4)
def _fastembed(model_name: str) -> FastEmbedEmbedder:
    # Loading the ONNX session is the slow part; share one per model across engines.
    return FastEmbedEmbedder(model_name)


def get_embedder() -> Embedder:
    """Build the embedder selected by ``LUMEN_EMBEDDER``.

    ``auto`` (the default) keeps the original behaviour: OpenAI when an API key is
    set, otherwise the offline hashing embedder.
    """
    kind = settings.embedder.lower()
    if kind == "auto":
        kind = "hashing" if settings.offline else "openai"
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return _fastembed(settings.fastembed_model)
    if kind == "openai":
        if settings.offline:
            raise RuntimeError("LUMEN_EMBEDDER=openai needs OPENAI_API_KEY")
        return OpenAIEmbedder()
    raise ValueError(f"unknown LUMEN_EMBEDDER {settings.embedder!r}; use auto, hashing, fastembed or openai")
