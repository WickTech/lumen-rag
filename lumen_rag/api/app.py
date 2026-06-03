"""FastAPI surface for the RAG engine: ingest, query, stream, stats, health."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..engine import RagEngine
from ..llm import answer_stream

# Replaced by lifespan; initialised here so the name always exists (e.g. in tests).
engine: RagEngine = RagEngine()


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


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Each event is ``data: <json>\\n\\n``. Token events carry ``{"token": "..."}``
    and the final event carries ``{"done": true, "citations": [...]}``.
    """
    if len(engine.store) == 0:
        raise HTTPException(status_code=409, detail="Index is empty. Ingest documents first.")

    chunks = engine.retriever.retrieve(req.question, k=req.k)

    async def _sse() -> AsyncGenerator[str, None]:
        for token, citations in answer_stream(req.question, chunks):
            if citations is not None:
                payload = json.dumps({"done": True, "citations": citations})
            else:
                payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")
