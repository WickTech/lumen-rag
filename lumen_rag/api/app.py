"""FastAPI surface for the RAG engine: ingest, query, stats, health."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import settings
from ..engine import RagEngine

engine: RagEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    index = Path(settings.index_dir)
    if (index / "chunks.json").exists():
        engine = RagEngine.load(index)
    else:
        engine = RagEngine()
    yield


app = FastAPI(
    title="Lumen RAG",
    version="0.1.0",
    description="Ingest documents, retrieve with vector search, answer with citations.",
    lifespan=lifespan,
)


class Document(BaseModel):
    id: str | None = None
    text: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Document]
    chunk_size: int = 120
    overlap: int = 20


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "offline_mode": settings.offline}


@app.get("/stats")
def stats() -> dict:
    return {"chunks_indexed": len(engine.store), "embedding_dim": engine.store.dim}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    docs = [d.model_dump() for d in req.documents]
    total = engine.add_documents(docs, chunk_size=req.chunk_size, overlap=req.overlap)
    engine.save()
    return {"chunks_indexed": total}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if len(engine.store) == 0:
        raise HTTPException(status_code=409, detail="Index is empty. Ingest documents first.")
    result = engine.query(req.question, k=req.k)
    return {"answer": result.text, "citations": result.citations}
