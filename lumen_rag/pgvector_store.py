"""Postgres + pgvector vector store.

Implements the same ``Store`` interface as the in-memory ``VectorStore`` so the
engine, retriever and ingestion pipeline do not change. Search is an exact scan
by default (no ANN index), which makes results match the in-memory store; pass
``index="hnsw"`` to trade exactness for speed on large collections.

Every row carries a ``namespace``. Several stores with different namespaces can
share one table and never see each other's chunks, which is how a single
deployment serves several users or teams.

Needs the optional ``psycopg`` driver: ``pip install 'lumen-rag[pgvector]'``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .store import Chunk, ScoredChunk

if TYPE_CHECKING:  # annotation only; pydantic already depends on typing_extensions
    from typing_extensions import Self

# Up to 40 characters so derived index names (e.g. <table>_embedding_hnsw) fit in 63.
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,39}$")


def _vector_literal(vec: np.ndarray) -> str:
    """pgvector text form, e.g. ``[0.1,0.2]``; used with an explicit ``::vector`` cast."""
    return "[" + ",".join(repr(float(x)) for x in np.asarray(vec, dtype=np.float32).reshape(-1)) + "]"


class PgVectorStore:
    def __init__(
        self,
        dsn: str,
        dim: int,
        namespace: str = "default",
        table: str = "lumen_chunks",
        index: str = "none",
    ) -> None:
        if not _IDENTIFIER_RE.match(table):
            # The table name is interpolated into SQL, so it must be a plain identifier.
            raise ValueError(f"invalid table name {table!r}: use lowercase letters, digits and underscores")
        if not namespace:
            raise ValueError("namespace must be a non-empty string")
        if index not in ("none", "hnsw"):
            raise ValueError("index must be 'none' (exact scan) or 'hnsw' (approximate)")
        if dim <= 0:
            raise ValueError("dim must be positive")
        try:
            import psycopg
            from psycopg import sql
        except ImportError as e:
            raise ImportError(
                "psycopg is required for the pgvector store. Install it with: pip install 'lumen-rag[pgvector]'"
            ) from e

        self.dim = dim
        self.namespace = namespace
        self.table = table
        self._sql = sql
        # Identifiers go through psycopg.sql so they are quoted, never string-formatted;
        # the regex above additionally restricts them to plain lowercase names.
        self._t = sql.Identifier(table)
        self._build_queries()
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._ensure_schema(index)

    def _build_queries(self) -> None:
        """Compose every statement once; execute() only ever receives a finished query object."""
        sql, t = self._sql, self._t
        self._q_create_table = sql.SQL(
            """CREATE TABLE IF NOT EXISTS {t} (
                seq bigserial PRIMARY KEY,
                namespace text NOT NULL,
                id text NOT NULL,
                doc_id text NOT NULL,
                text text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dim}) NOT NULL,
                UNIQUE (namespace, id)
            )"""
        ).format(t=t, dim=sql.Literal(self.dim))
        self._q_create_ns_index = sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {t} (namespace)").format(
            idx=sql.Identifier(f"{self.table}_namespace_idx"), t=t
        )
        self._q_create_hnsw = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {idx} ON {t} USING hnsw (embedding vector_cosine_ops)"
        ).format(idx=sql.Identifier(f"{self.table}_embedding_hnsw"), t=t)
        self._q_count = sql.SQL("SELECT count(*) FROM {t} WHERE namespace = %s").format(t=t)
        self._q_insert = sql.SQL(
            """INSERT INTO {t} (namespace, id, doc_id, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (namespace, id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id, text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding"""
        ).format(t=t)
        self._q_search = sql.SQL(
            """SELECT id, doc_id, text, metadata, 1 - (embedding <=> %s::vector) AS score
                FROM {t}
                WHERE namespace = %s
                ORDER BY embedding <=> %s::vector, seq
                LIMIT %s"""
        ).format(t=t)
        self._q_chunks = sql.SQL(
            "SELECT id, doc_id, text, metadata FROM {t} WHERE namespace = %s ORDER BY seq"
        ).format(t=t)
        self._q_clear = sql.SQL("DELETE FROM {t} WHERE namespace = %s").format(t=t)

    # --- schema -------------------------------------------------------------
    def _ensure_schema(self, index: str) -> None:
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(self._q_create_table)
        self._conn.execute(self._q_create_ns_index)
        existing = self._conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
            (self.table,),
        ).fetchone()
        if existing is not None and existing[0] != self.dim:
            raise ValueError(
                f"table {self.table!r} stores {existing[0]}-dimensional vectors "
                f"but this store expects {self.dim}"
            )
        if index == "hnsw":
            self._conn.execute(self._q_create_hnsw)

    # --- Store interface ----------------------------------------------------
    def __len__(self) -> int:
        row = self._conn.execute(self._q_count, (self.namespace,)).fetchone()
        return int(row[0])

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("chunks and vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        rows = [
            (self.namespace, c.id, c.doc_id, c.text, json.dumps(c.metadata), _vector_literal(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        # Re-ingesting a chunk id replaces it (the in-memory store would keep both copies).
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(self._q_insert, rows)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[ScoredChunk]:
        if k < 1:
            return []
        literal = _vector_literal(query_vec)
        rows = self._conn.execute(self._q_search, (literal, self.namespace, literal, k)).fetchall()
        return [
            ScoredChunk(Chunk(id=r[0], text=r[2], doc_id=r[1], metadata=r[3] or {}), float(r[4])) for r in rows
        ]

    def chunks(self) -> list[Chunk]:
        rows = self._conn.execute(self._q_chunks, (self.namespace,)).fetchall()
        return [Chunk(id=r[0], text=r[2], doc_id=r[1], metadata=r[3] or {}) for r in rows]

    def save(self, directory: str | Path) -> None:
        """No-op: rows are committed as they are added."""

    # --- extras -------------------------------------------------------------
    def clear(self) -> int:
        """Delete every chunk in this namespace and return how many were removed."""
        cur = self._conn.execute(self._q_clear, (self.namespace,))
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
