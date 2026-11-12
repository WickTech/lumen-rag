"""Entry point for the Lumen RAG Space.

The Space serves the same thing as `lumen serve`: the demo UI at / (the one in
the README GIF) and the HTTP API with its docs at /docs.

Free accounts can only host Gradio Spaces on ZeroGPU, and ZeroGPU registers its
@spaces.GPU functions inside Gradio's own launch(), not under a hand-run server
(that fails with "No @spaces.GPU function detected during startup"). So the
Gradio app below is still what launches, but the FastAPI routes are handed to
it first and take precedence, including /. The Gradio blocks stay as the
backend ZeroGPU needs, and use the same engine and index as the API.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

# The Space runs a real local embedder (bge-small, ONNX via fastembed, CPU) and
# refuses low-confidence questions. Set before importing lumen_rag: settings are
# read at import. The threshold is the calibrated value from scripts/benchmark.py.
os.environ.setdefault("LUMEN_EMBEDDER", "fastembed")
os.environ.setdefault("LUMEN_ABSTAIN_THRESHOLD", "0.66")
# Second gate: a SQuAD 2.0 reader refuses near misses that similarity lets through.
os.environ.setdefault("LUMEN_READER", "onnx-community/roberta-base-squad2-ONNX")
os.environ.setdefault("LUMEN_READER_THRESHOLD", "-3.5")

import importlib

import gradio as gr
import spaces

from lumen_rag.api.ratelimit import check as rate_check
from lumen_rag.engine import RagEngine
from lumen_rag.eval import evaluate
from lumen_rag.eval.harness import load_cases
from lumen_rag.ingestion.loaders import _LOADERS, load_file
from lumen_rag.retrieval import Retriever

# The FastAPI routes in lumen_rag/api/app.py enforce their own rate limits, upload cap
# and admin gate; the Gradio callbacks below bypass those routes, so they repeat the
# same protections, adapted to Gradio's callback model. The admin gate and the index
# write lock are the API module's own (admin_denial, index_lock), not copies.
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB per file
_ADMIN_KEY = os.getenv("ADMIN_KEY", "")


def _client_ip(request: gr.Request | None) -> str:
    return request.client.host if request and request.client else "unknown"


def _rate_limited(request: gr.Request | None, bucket: str, limit: int, window_s: float) -> bool:
    return rate_check(f"{bucket}:{_client_ip(request)}", limit, window_s)


def _find_repo_root() -> Path:
    # Locally this file lives at deploy/hf-space/gradio_app.py (repo root 2 dirs up
    # from its directory); on the deployed Space it's copied to app.py at the root.
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent.parent):
        if (candidate / "data" / "docs").is_dir():
            return candidate
    return here


_REPO_ROOT = _find_repo_root()
_SAMPLE_DIR = _REPO_ROOT / "data" / "docs"
_EVAL_PATH = _REPO_ROOT / "data" / "eval.jsonl"

# The FastAPI module, not the `app` object lumen_rag.api re-exports: its `engine`
# global is what the HTML UI and the API use, and the lifespan may replace it,
# so it is looked up on every call rather than bound once.
_api = importlib.import_module("lumen_rag.api.app")


def _engine() -> RagEngine:
    return _api.engine


def _require_admin(admin_key: str) -> None:
    # Same policy as the API routes: ADMIN_KEY set -> it must match; unset -> refused
    # on the Space (SPACE_ID is set there), open when run locally.
    status = _api.admin_denial(admin_key or "")
    if status == 401:
        raise gr.Error("Invalid admin key.")
    if status == 403:
        raise gr.Error("Ingestion is locked on the public demo.")


async def _add_documents(docs: list[dict]) -> int:
    # One writer at a time, shared with the API routes; read the engine after locking
    # so a concurrent reset cannot leave this write on a discarded engine.
    async with _api.index_lock:
        eng = _engine()
        return await asyncio.to_thread(eng.add_documents, docs)


def _load_sample_docs() -> list[dict]:
    files = sorted(p for p in _SAMPLE_DIR.iterdir() if p.suffix.lower() in _LOADERS)
    return [load_file(p) for p in files]


async def load_sample(admin_key: str, request: gr.Request) -> str:
    if _rate_limited(request, "ingest", limit=10, window_s=60):
        raise gr.Error("Too many requests, try again shortly.")
    _require_admin(admin_key)
    docs = await asyncio.to_thread(_load_sample_docs)
    total = await _add_documents(docs)
    return f"Indexed {len(docs)} sample files, {total} chunks."


def _load_uploads(files: list[str]) -> list[dict] | str:
    """Parse uploads into documents, or return a message explaining the refusal (blocking)."""
    if len(files) > _api.settings.max_upload_files:
        return f"At most {_api.settings.max_upload_files} files per upload."
    docs = []
    for f in files:
        suffix = Path(f).suffix.lower()
        if suffix not in _LOADERS:
            continue
        if Path(f).stat().st_size > _MAX_UPLOAD_BYTES:
            return f"{Path(f).name} exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
        try:
            doc = load_file(f)
        except ImportError as e:  # optional loader extra not installed
            raise gr.Error(str(e)) from e
        doc["id"] = Path(f).stem
        docs.append(doc)
    if not docs:
        return f"No supported files. Supported: {sorted(_LOADERS)}"
    return docs


async def upload_files(files: list[str], admin_key: str, request: gr.Request) -> str:
    if _rate_limited(request, "ingest", limit=10, window_s=60):
        raise gr.Error("Too many requests, try again shortly.")
    _require_admin(admin_key)
    docs = await asyncio.to_thread(_load_uploads, files)
    if isinstance(docs, str):
        return docs
    total = await _add_documents(docs)
    return f"Indexed {len(docs)} files, {total} chunks."


async def reset_index(admin_key: str, request: gr.Request) -> str:
    # Wipes the (single, shared) index for every visitor. Rate-limited first so
    # guessing ADMIN_KEY is throttled too, then the same admin gate as the API
    # (refused on the Space when no ADMIN_KEY is configured).
    if _rate_limited(request, "reset", limit=5, window_s=60):
        raise gr.Error("Too many requests, try again shortly.")
    _require_admin(admin_key)
    async with _api.index_lock:
        _api.engine = RagEngine()
    return "Index reset."


@spaces.GPU(duration=30)
def ask(question: str, k: int, mode: str, request: gr.Request):
    if _rate_limited(request, "query", limit=30, window_s=60):
        raise gr.Error("Too many requests, try again shortly.")
    if len(_engine().store) == 0:
        return "Index is empty — load the sample corpus or upload files first.", ""
    result = _engine().query(question, k=int(k), mode=mode)
    lines = []
    if result.confidence is not None:
        verdict = "below threshold, abstained" if result.abstained else "answered"
        reader = "" if result.answerability is None else f", reader answerability {result.answerability:.1f}"
        lines.append(f"Retrieval confidence {result.confidence:.3f}{reader} ({verdict})")
    lines += [
        f"[{c['n']}] {c['doc_id']} — {(c.get('context') or c['source'])} (score={c['score']:.3f})"
        for c in result.citations
    ]
    return result.text, "\n".join(lines)


def run_eval(k: int, request: gr.Request) -> str:
    if _rate_limited(request, "query", limit=30, window_s=60):
        raise gr.Error("Too many requests, try again shortly.")
    if not _EVAL_PATH.exists():
        return "Bundled eval set not found."
    if len(_engine().store) == 0:
        return "Index is empty — load the sample corpus first."
    cases = load_cases(_EVAL_PATH)
    eng = _engine()
    report = evaluate(Retriever(eng.store, eng.embedder), cases, k=int(k))
    d = report.as_dict()
    return "\n".join(f"{key}: {value}" for key, value in d.items())


with gr.Blocks(title="Lumen RAG") as demo:
    gr.Markdown(
        "# Lumen RAG\n"
        "Transparent, evaluated RAG: ingest documents, retrieve with dense "
        "(bge-small, running locally on CPU), BM25 or hybrid search, answer with "
        "citations, and refuse questions the documents can't answer. No API key: "
        "answers are extractive, not generated."
    )

    with gr.Row():
        sample_btn = gr.Button("Load sample corpus")
        reset_btn = gr.Button("Reset index")
    if _ADMIN_KEY:
        admin_key_box = gr.Textbox(label="Admin key (required to load, upload or reset)", type="password")
    else:
        admin_key_box = gr.State("")
    upload = gr.File(label="Or upload documents", file_count="multiple")
    status = gr.Textbox(label="Index status", interactive=False)

    sample_btn.click(load_sample, inputs=admin_key_box, outputs=status)
    reset_btn.click(reset_index, inputs=admin_key_box, outputs=status)
    upload.upload(upload_files, inputs=[upload, admin_key_box], outputs=status)

    gr.Markdown("---")

    question = gr.Textbox(label="Question")
    with gr.Row():
        k = gr.Slider(1, 20, value=5, step=1, label="k")
        mode = gr.Dropdown(["vector", "hybrid", "bm25"], value="vector", label="Retrieval mode")
    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=4)
    citations_box = gr.Textbox(label="Confidence and citations", lines=4)

    ask_btn.click(ask, inputs=[question, k, mode], outputs=[answer_box, citations_box])

    gr.Markdown("---")

    with gr.Row():
        eval_k = gr.Slider(1, 20, value=5, step=1, label="eval k")
        eval_btn = gr.Button("Run retrieval eval (recall@k, MRR, nDCG@k)")
    eval_box = gr.Textbox(label="Eval report", lines=6)
    eval_btn.click(run_eval, inputs=eval_k, outputs=eval_box)


if __name__ == "__main__":
    demo.launch(
        # Routes passed to the FastAPI constructor are matched before the ones
        # Gradio adds, so / redirects to the demo UI instead of Gradio's page.
        app_kwargs={"routes": list(_api.app.routes), "lifespan": _api.lifespan},
        # SSR puts a Node server in front that owns /; keep Python serving it.
        ssr_mode=False,
    )
