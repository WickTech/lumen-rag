"""Central configuration, loaded from environment with sensible offline defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    # Embedding backend: "auto" (OpenAI when a key is set, else hashing), "hashing",
    # "fastembed" (local ONNX, needs the [embed] extra) or "openai".
    embedder: str = os.getenv("LUMEN_EMBEDDER", "auto")
    fastembed_model: str = os.getenv("LUMEN_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    # Refuse to answer when retrieval confidence is below this value (see
    # Retriever.retrieve_with_confidence). Empty disables abstention.
    abstain_threshold: float | None = (
        float(os.environ["LUMEN_ABSTAIN_THRESHOLD"]) if os.getenv("LUMEN_ABSTAIN_THRESHOLD") else None
    )
    # Extractive reader (SQuAD 2.0 model, ONNX) that checks whether the retrieved
    # passages contain an answer; e.g. "onnx-community/tinyroberta-squad2-ONNX".
    # Empty disables it (needs the [reader] extra when set).
    reader_model: str = os.getenv("LUMEN_READER", "")
    reader_file: str = os.getenv("LUMEN_READER_FILE", "onnx/model_quantized.onnx")
    # Refuse when the reader's answerability (best span minus no-answer score) is below this.
    reader_threshold: float = float(os.getenv("LUMEN_READER_THRESHOLD", "0"))
    # "1" answers offline with the sentence holding the reader's span instead of the
    # top passages: more precise, but wrong on multi-hop questions (see README).
    reader_answers: bool = os.getenv("LUMEN_READER_ANSWERS", "") == "1"
    # "1" splits compound questions into sub-queries and fuses their results (RRF).
    decompose: bool = os.getenv("LUMEN_DECOMPOSE", "") == "1"
    # "1" prepends "Document: <title>. Section: <headings>." to each chunk for embedding
    # and BM25 (contextual retrieval, built offline). Re-ingest after changing it.
    contextual: bool = os.getenv("LUMEN_CONTEXTUAL", "") == "1"
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    index_dir: str = os.getenv("LUMEN_INDEX_DIR", ".lumen_index")
    admin_key: str = os.getenv("ADMIN_KEY", "")
    # Cross-encoder model for second-stage reranking, e.g. "BAAI/bge-reranker-base".
    # Empty disables reranking (requires the [rerank] extra when set).
    reranker_model: str = os.getenv("LUMEN_RERANKER", "")
    # Candidate pool size multiplier fed to the reranker (pool = k * this).
    rerank_pool_factor: int = int(os.getenv("LUMEN_RERANK_POOL", "4"))
    # "1" routes PDF/DOCX/PPTX/HTML through Docling (requires the [docling] extra).
    use_docling: bool = os.getenv("LUMEN_DOCLING", "") == "1"
    # OpenTelemetry tracing: "" (off), "console" or "otlp" (needs the [tracing] extra).
    tracing: str = os.getenv("LUMEN_TRACING", "")
    # "1" also records question and answer text on spans; off by default (lengths only).
    trace_content: bool = os.getenv("LUMEN_TRACE_CONTENT", "") == "1"
    # Vector store backend: "memory" (default, persisted to index_dir) or "pgvector"
    # (requires the [pgvector] extra and LUMEN_PG_DSN).
    store: str = os.getenv("LUMEN_STORE", "memory")
    pg_dsn: str = os.getenv("LUMEN_PG_DSN", "")
    pg_table: str = os.getenv("LUMEN_PG_TABLE", "lumen_chunks")
    # Isolates document collections per user/team on a shared backend (pgvector).
    namespace: str = os.getenv("LUMEN_NAMESPACE", "default")

    @property
    def offline(self) -> bool:
        """True when no API key is configured; engine falls back to local embeddings."""
        return not self.openai_api_key


settings = Settings()
