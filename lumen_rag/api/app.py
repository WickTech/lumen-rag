"""FastAPI surface for the RAG engine: ingest, query, stream, stats, health."""
from __future__ import annotations

import asyncio
import hmac
import json
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

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
_SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "docs"
_READ_CHUNK = 64 * 1024

# Replaced by lifespan; initialised here so the name always exists (e.g. in tests).
engine: RagEngine = RagEngine()

# Serialises every index mutation (add_documents, save, engine swaps): the store is not
# safe for concurrent writers. Always read `engine` after acquiring it, so a write never
# lands on an engine that a concurrent /reset has just replaced.
index_lock = asyncio.Lock()


def _load_sample_docs() -> list[dict]:
    """Parse the bundled sample corpus (blocking)."""
    files = sorted(p for p in _SAMPLE_DIR.iterdir() if p.suffix.lower() in _LOADERS)
    return [load_file(p) for p in files]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    tracing.setup_from_env()  # no-op unless LUMEN_TRACING is set
    index = Path(settings.index_dir)
    async with index_lock:
        if (index / "chunks.json").exists():
            engine = RagEngine.load(index)
        else:
            engine = RagEngine()
        # The hosted demo locks visitor ingestion, so it starts with the sample corpus
        # instead of an empty index (LUMEN_SEED_SAMPLE; on by default when hosted).
        if settings.seed_sample and len(engine.store) == 0 and _SAMPLE_DIR.is_dir():
            docs = await asyncio.to_thread(_load_sample_docs)
            await asyncio.to_thread(_index_documents, engine, docs, 120, 20)
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


def ingest_locked() -> bool:
    """True when the demo UI cannot ingest without a key: ADMIN_KEY set, or hosted."""
    return bool(settings.admin_key) or settings.hosted


def admin_denial(provided: str) -> int | None:
    """Status code that refuses an index mutation, or None to allow it.

    ADMIN_KEY set: the caller must send it (401 otherwise). Not set: refused when
    hosted (403, so a public Space is never left open), allowed for local use.
    """
    admin_key = settings.admin_key
    if admin_key:
        return None if hmac.compare_digest(provided.encode(), admin_key.encode()) else 401
    return 403 if settings.hosted else None


def require_admin(request: Request) -> None:
    """Dependency guarding every route that mutates the index."""
    status = admin_denial(request.headers.get("x-admin-key", ""))
    if status == 401:
        # Throttle key guessing: failed attempts share one bucket per client.
        enforce(request, "admin-auth", limit=5, window_s=60)
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Key header.")
    if status == 403:
        raise HTTPException(
            status_code=403,
            detail="Ingestion is locked on the public demo (no ADMIN_KEY configured).",
        )


def check_chunking(chunk_size: int, overlap: int) -> None:
    """Raise ValueError unless chunk_size and overlap are within the configured bounds."""
    lo, hi = settings.min_chunk_size, settings.max_chunk_size
    if not lo <= chunk_size <= hi:
        raise ValueError(f"chunk_size must be between {lo} and {hi}")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")


class Document(BaseModel):
    id: str | None = None
    text: str = Field(min_length=1, max_length=_MAX_DOC_CHARS)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    # Unlike /ingest/upload (capped per-file at settings.max_upload_bytes), this JSON
    # body had no bound at all — an unauthenticated request could carry
    # arbitrarily many arbitrarily large documents in one call.
    documents: list[Document] = Field(max_length=_MAX_DOCS_PER_REQUEST)
    chunk_size: int = 120
    overlap: int = 20

    @model_validator(mode="after")
    def _bounded_chunking(self) -> IngestRequest:
        check_chunking(self.chunk_size, self.overlap)  # ValueError -> 422
        return self


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
        "reader": settings.reader_model or None,
        "reader_threshold": settings.reader_threshold if settings.reader_model else None,
        # True when the demo UI cannot ingest or reset (ADMIN_KEY set, or hosted without one).
        "ingest_locked": ingest_locked(),
    }


@app.get("/stats")
def stats() -> dict:
    return {"chunks_indexed": len(engine.store), "embedding_dim": engine.store.dim}


@app.post("/ingest", dependencies=[Depends(require_admin)])
async def ingest(req: IngestRequest, request: Request) -> dict:
    enforce(request, "ingest", limit=10, window_s=60)
    docs = [d.model_dump() for d in req.documents]
    async with index_lock:
        total = await asyncio.to_thread(_index_documents, engine, docs, req.chunk_size, req.overlap)
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
        "answerability": result.answerability,
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


async def _read_bounded(f: UploadFile, limit: int) -> bytes:
    """Read an upload in chunks, refusing with 413 as soon as it passes `limit` bytes."""
    parts: list[bytes] = []
    size = 0
    while chunk := await f.read(_READ_CHUNK):
        size += len(chunk)
        if size > limit:
            raise HTTPException(
                status_code=413,
                detail=f"{f.filename} exceeds the {limit // (1024 * 1024)}MB upload limit",
            )
        parts.append(chunk)
    return b"".join(parts)


@app.post("/ingest/upload", dependencies=[Depends(require_admin)])
async def ingest_upload(
    request: Request, files: list[UploadFile], chunk_size: int = 120, overlap: int = 20
) -> dict:
    enforce(request, "ingest", limit=10, window_s=60)
    try:
        check_chunking(chunk_size, overlap)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=413, detail=f"At most {settings.max_upload_files} files per upload"
        )
    docs = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in _LOADERS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")
        data = await _read_bounded(f, settings.max_upload_bytes)
        # Parsing (PDF/DOCX can be heavy) and indexing (embedding calls) block: run them off the loop.
        docs.append(await asyncio.to_thread(_load_upload, data, f.filename or "", suffix))
    async with index_lock:
        total = await asyncio.to_thread(_index_documents, engine, docs, chunk_size, overlap)
    return {"files_indexed": len(docs), "chunks_indexed": total}


@app.post("/ingest/sample", dependencies=[Depends(require_admin)])
async def ingest_sample(request: Request) -> dict:
    enforce(request, "ingest", limit=10, window_s=60)
    if not _SAMPLE_DIR.is_dir():
        raise HTTPException(status_code=404, detail="Bundled sample corpus not found.")
    docs = await asyncio.to_thread(_load_sample_docs)
    async with index_lock:
        total = await asyncio.to_thread(_index_documents, engine, docs, 120, 20)
    return {"files_indexed": len(docs), "chunks_indexed": total}


@app.post("/reset", dependencies=[Depends(require_admin)])
async def reset(request: Request) -> dict:
    # Wipes the (single, shared) index. require_admin gates it (ADMIN_KEY, or
    # refused when hosted without one) and throttles failed key attempts.
    enforce(request, "reset", limit=5, window_s=60)

    global engine
    async with index_lock:
        fresh = RagEngine()
        await asyncio.to_thread(fresh.save)
        engine = fresh
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
    r = await asyncio.to_thread(engine.retrieve, req.question, req.k, req.mode)
    answerability = r.reading.answerability if r.reading else None

    async def _sse() -> AsyncGenerator[str, None]:
        if r.abstained:
            yield f"data: {json.dumps({'token': ABSTAIN_TEXT})}\n\n"
            done = {
                "done": True,
                "citations": [],
                "abstained": True,
                "confidence": r.confidence,
                "answerability": answerability,
            }
            yield f"data: {json.dumps(done)}\n\n"
            return
        reading = r.reading if settings.reader_answers else None
        async for token, citations in answer_stream_async(req.question, r.chunks, reading):
            if citations is not None:
                payload = json.dumps(
                    {
                        "done": True,
                        "citations": citations,
                        "abstained": False,
                        "confidence": r.confidence,
                        "answerability": answerability,
                    }
                )
            else:
                payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")
