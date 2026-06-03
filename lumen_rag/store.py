"""A tiny, persistable in-memory vector store with exact cosine search.

Deliberately not a full vector DB — the point is to make the retrieval maths
visible and dependency-free. Vectors are L2-normalised on insert, so cosine
similarity is a single matrix-vector dot product. Swap this module for
pgvector/Qdrant/Chroma without touching the retriever.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    metadata: dict


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("chunks and vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[ScoredChunk]:
        if len(self._chunks) == 0:
            return []
        scores = self._vectors @ query_vec.reshape(-1)  # cosine (already normalised)
        k = min(k, len(self._chunks))
        # argpartition for top-k, then sort just those.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [ScoredChunk(self._chunks[i], float(scores[i])) for i in top]

    # --- persistence -------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self._vectors)
        (d / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False)
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        d = Path(directory)
        vectors = np.load(d / "vectors.npy")
        store = cls(dim=int(vectors.shape[1]) if vectors.size else 0)
        store._vectors = vectors.astype(np.float32)
        store._chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
        return store
