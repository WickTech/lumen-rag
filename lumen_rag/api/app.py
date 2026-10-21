"""FastAPI surface for the RAG engine: ingest, query, stream, stats, health."""
from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from secrets import compare_digest

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import tracing
from ..config import settings
from ..engine import RagEngine
from ..eval import evaluate
from ..eval.harness import load_cases
from ..ingestion.loaders import _LOADERS, load_file
from ..llm import ABSTAIN_TEXT, answer_stream_async
from ..retrieval import RetrievalMode, Retriever
from .ratelimit import enforce

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB per file

# Replaced by lifespan; initialised here so the name always exists (e.g. in tests).
engine: RagEngine = RagEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    tracing.setup_from_env()  # no-op unless LUMEN_TRACING is set
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


_MAX_DOC_CHARS = 2_000_000  # ~2M chars per document
_MAX_DOCS_PER_REQUEST = 100


class Document(BaseModel):
    id: str | None = None
    text: str = Field(min_length=1, max_length=_MAX_DOC_CHARS)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    # Unlike /ingest/upload (capped per-file at _MAX_UPLOAD_BYTES), this JSON
    # body had no bound at all — an unauthenticated request could carry
    # arbitrarily many arbitrarily large documents in one call.
    documents: list[Document] = Field(max_length=_MAX_DOCS_PER_REQUEST)
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
    emb = engine.embedder
    return {
        "status": "ok",
        "offline_mode": settings.offline,
        # Which embedder is live, so the demo UI can say what it is running on.
        "embedder": type(emb).__name__,
        "embedding_model": getattr(emb, "model_name", None),
        "abstain_threshold": settings.abstain_threshold,
    }


@app.get("/stats")
def stats() -> dict:
    return {"chunks_indexed": len(engine.store), "embedding_dim": engine.store.dim}


@app.post("/ingest")
def ingest(req: IngestRequest, request: Request) -> dict:
    enforce(request, "ingest", limit=10, window_s=60)
    docs = [d.model_dump() for d in req.documents]
    total = engine.add_documents(docs, chunk_size=req.chunk_size, overlap=req.overlap)
    engine.save()
    return {"chunks_indexed": total}


@app.post("/query")
async def query(req: QueryRequest, request: Request) -> dict:
    enforce(request, "query", limit=30, window_s=60)
    if len(engine.store) == 0:
        raise HTTPException(status_code=409, detail="Index is empty. Ingest documents first.")
    result = await engine.aquery(req.question, k=req.k, mode=req.mode)
    return {
        "answer": result.text,
        "citations": result.citations,
        "abstained": result.abstained,
        "confidence": result.confidence,
    }


def _load_upload(data: bytes, filename: str, suffix: str) -> dict:
    """Parse one uploaded file into a document (blocking)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            doc = load_file(tmp.name)
        except ImportError as e:  # optional loader extra not installed
            raise HTTPException(status_code=400, detail=str(e)) from e
    doc["id"] = Path(filename).stem
    doc["metadata"]["source"] = filename
    return doc


def _index_documents(eng: RagEngine, docs: list[dict], chunk_size: int, overlap: int) -> int:
    """Embed, store and persist documents (blocking)."""
    total = eng.add_documents(docs, chunk_size=chunk_size, overlap=overlap)
    eng.save()
    return total


@app.post("/ingest/upload")
async def ingest_upload(
    request: Request, files: list[UploadFile], chunk_size: int = 120, overlap: int = 20
) -> dict:
    enforce(request, "ingest", limit=10, window_s=60)
    docs = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in _LOADERS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")
        data = await f.read()
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{f.filename} exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit",
            )
        # Parsing (PDF/DOCX can be heavy) and indexing (embedding calls) block: run them off the loop.
        docs.append(await asyncio.to_thread(_load_upload, data, f.filename or "", suffix))
    total = await asyncio.to_thread(_index_documents, engine, docs, chunk_size, overlap)
    return {"files_indexed": len(docs), "chunks_indexed": total}


@app.post("/ingest/sample")
def ingest_sample(request: Request) -> dict:
    enforce(request, "ingest", limit=10, window_s=60)
    sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "docs"
    if not sample_dir.is_dir():
        raise HTTPException(status_code=404, detail="Bundled sample corpus not found.")
    files = sorted(p for p in sample_dir.iterdir() if p.suffix.lower() in _LOADERS)
    docs = [load_file(p) for p in files]
    total = engine.add_documents(docs)
    engine.save()
    return {"files_indexed": len(docs), "chunks_indexed": total}


@app.post("/reset")
def reset(request: Request) -> dict:
    # Wipes the (single, shared) index — on the public HF Space demo this
    # would let any visitor nuke the corpus for everyone else. Gated behind
    # ADMIN_KEY when set; stays open for local/offline dev when it isn't.
    # Rate-limited first (before the key check) so guessing ADMIN_KEY is
    # throttled too, not just successful resets.
    enforce(request, "reset", limit=5, window_s=60)

    admin_key = settings.admin_key
    if admin_key:
        provided = request.headers.get("x-admin-key", "")
        if not compare_digest(provided, admin_key):
            raise HTTPException(status_code=404, detail="Not found")

    global engine
    engine = RagEngine()
    engine.save()
    return {"status": "reset"}


_EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@app.get("/eval")
def run_eval(request: Request, dataset: str = "eval.jsonl", k: int = 5) -> dict:
    enforce(request, "query", limit=30, window_s=60)
    # dataset was previously a raw, unsanitized path — `?dataset=../../.env`
    # (or any other file) let an unauthenticated caller probe or read
    # arbitrary files that happen to parse as JSONL. Resolve it against a
    # fixed base directory and reject anything that escapes it.
    path = (_EVAL_DIR / dataset).resolve()
    if path != _EVAL_DIR and _EVAL_DIR not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid dataset path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Eval set not found: {dataset}")
    if len(engine.store) == 0:
        raise HTTPException(status_code=409, detail="Index is empty. Ingest documents first.")
    cases = load_cases(path)
    report = evaluate(Retriever(engine.store, engine.embedder), cases, k=k)
    return report.as_dict() | {"per_case": report.per_case}


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Each event is ``data: <json>\\n\\n``. Token events carry ``{"token": "..."}``
    and the final event carries ``{"done": true, "citations": [...]}``.
    """
    enforce(request, "query", limit=30, window_s=60)
    if len(engine.store) == 0:
        raise HTTPException(status_code=409, detail="Index is empty. Ingest documents first.")

    # Retrieval can be CPU-bound or a network round trip: keep it off the event loop.
    chunks, confidence, abstained = await asyncio.to_thread(engine.retrieve, req.question, req.k, req.mode)

    async def _sse() -> AsyncGenerator[str, None]:
        if abstained:
            yield f"data: {json.dumps({'token': ABSTAIN_TEXT})}\n\n"
            done = {"done": True, "citations": [], "abstained": True, "confidence": confidence}
            yield f"data: {json.dumps(done)}\n\n"
            return
        async for token, citations in answer_stream_async(req.question, chunks):
            if citations is not None:
                payload = json.dumps(
                    {"done": True, "citations": citations, "abstained": False, "confidence": confidence}
                )
            else:
                payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")
