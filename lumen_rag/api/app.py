"""FastAPI surface for the RAG engine: ingest, query, stream, stats, health."""
from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import settings
from ..engine import RagEngine
from ..eval import evaluate
from ..eval.harness import load_cases
from ..ingestion.loaders import _LOADERS, load_file
from ..llm import answer_stream
from ..retrieval import Retriever, RetrievalMode

_STATIC_DIR = Path(__file__).resolve().parent / "static"

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

if _STATIC_DIR.is_dir():
    app.mount("/demo", StaticFiles(directory=_STATIC_DIR, html=True), name="demo")


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
    mode: RetrievalMode = "hybrid"


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/demo/")


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
    result = engine.query(req.question, k=req.k, mode=req.mode)
    return {"answer": result.text, "citations": result.citations}


@app.post("/ingest/upload")
async def ingest_upload(files: list[UploadFile], chunk_size: int = 120, overlap: int = 20) -> dict:
    docs = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in _LOADERS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")
        data = await f.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            doc = load_file(tmp.name)
            doc["id"] = Path(f.filename).stem
            doc["metadata"]["source"] = f.filename
            docs.append(doc)
    total = engine.add_documents(docs, chunk_size=chunk_size, overlap=overlap)
    engine.save()
    return {"files_indexed": len(docs), "chunks_indexed": total}


@app.post("/ingest/sample")
def ingest_sample() -> dict:
    sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "docs"
    if not sample_dir.is_dir():
        raise HTTPException(status_code=404, detail="Bundled sample corpus not found.")
    files = sorted(p for p in sample_dir.iterdir() if p.suffix.lower() in _LOADERS)
    docs = [load_file(p) for p in files]
    total = engine.add_documents(docs)
    engine.save()
    return {"files_indexed": len(docs), "chunks_indexed": total}


@app.post("/reset")
def reset() -> dict:
    global engine
    engine = RagEngine()
    engine.save()
    return {"status": "reset"}


@app.get("/eval")
def run_eval(dataset: str = "data/eval.jsonl", k: int = 5) -> dict:
    path = Path(dataset)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Eval set not found: {dataset}")
    if len(engine.store) == 0:
        raise HTTPException(status_code=409, detail="Index is empty. Ingest documents first.")
    cases = load_cases(path)
    report = evaluate(Retriever(engine.store, engine.embedder), cases, k=k)
    return report.as_dict() | {"per_case": report.per_case}


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Each event is ``data: <json>\\n\\n``. Token events carry ``{"token": "..."}``
    and the final event carries ``{"done": true, "citations": [...]}``.
    """
    if len(engine.store) == 0:
        raise HTTPException(status_code=409, detail="Index is empty. Ingest documents first.")

    chunks = engine.retriever.retrieve(req.question, k=req.k, mode=req.mode)

    async def _sse() -> AsyncGenerator[str, None]:
        for token, citations in answer_stream(req.question, chunks):
            if citations is not None:
                payload = json.dumps({"done": True, "citations": citations})
            else:
                payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")
