"""Gradio front-end for the Lumen RAG Space (free-tier CPU, no Docker needed).

Wraps `lumen_rag.RagEngine` directly in-process — no HTTP layer.
"""
from __future__ import annotations

from pathlib import Path

import spaces
import gradio as gr

from lumen_rag.engine import RagEngine
from lumen_rag.eval import evaluate
from lumen_rag.eval.harness import load_cases
from lumen_rag.ingestion.loaders import _LOADERS, load_file
from lumen_rag.retrieval import Retriever

def _find_repo_root() -> Path:
    # Locally this file lives at deploy/hf-space/gradio_app.py (repo root 3 up);
    # on the deployed Space it's copied to app.py at the repo root (0 up).
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent.parent.parent):
        if (candidate / "data" / "docs").is_dir():
            return candidate
    return here


_REPO_ROOT = _find_repo_root()
_SAMPLE_DIR = _REPO_ROOT / "data" / "docs"
_EVAL_PATH = _REPO_ROOT / "data" / "eval.jsonl"

engine = RagEngine()


def load_sample() -> str:
    files = sorted(p for p in _SAMPLE_DIR.iterdir() if p.suffix.lower() in _LOADERS)
    docs = [load_file(p) for p in files]
    total = engine.add_documents(docs)
    return f"Indexed {len(docs)} sample files, {total} chunks."


def upload_files(files: list[str]) -> str:
    docs = []
    for f in files:
        suffix = Path(f).suffix.lower()
        if suffix not in _LOADERS:
            continue
        doc = load_file(f)
        doc["id"] = Path(f).stem
        docs.append(doc)
    if not docs:
        return f"No supported files. Supported: {sorted(_LOADERS)}"
    total = engine.add_documents(docs)
    return f"Indexed {len(docs)} files, {total} chunks."


def reset_index() -> str:
    global engine
    engine = RagEngine()
    return "Index reset."


@spaces.GPU(duration=30)
def ask(question: str, k: int, mode: str):
    if len(engine.store) == 0:
        return "Index is empty — load the sample corpus or upload files first.", ""
    result = engine.query(question, k=int(k), mode=mode)
    citations = "\n".join(
        f"[{c['n']}] {c['doc_id']} — {c['source']} (score={c['score']:.3f})"
        for c in result.citations
    )
    return result.text, citations


def run_eval(k: int) -> str:
    if not _EVAL_PATH.exists():
        return "Bundled eval set not found."
    if len(engine.store) == 0:
        return "Index is empty — load the sample corpus first."
    cases = load_cases(_EVAL_PATH)
    report = evaluate(Retriever(engine.store, engine.embedder), cases, k=int(k))
    d = report.as_dict()
    return "\n".join(f"{key}: {value}" for key, value in d.items())


with gr.Blocks(title="Lumen RAG") as demo:
    gr.Markdown(
        "# Lumen RAG\n"
        "Transparent, evaluated RAG: ingest documents, retrieve with hybrid "
        "vector+BM25 search, answer with citations. Runs 100% offline on a "
        "deterministic hashing embedder — no API key required."
    )

    with gr.Row():
        sample_btn = gr.Button("Load sample corpus")
        reset_btn = gr.Button("Reset index")
    upload = gr.File(label="Or upload documents", file_count="multiple")
    status = gr.Textbox(label="Index status", interactive=False)

    sample_btn.click(load_sample, outputs=status)
    reset_btn.click(reset_index, outputs=status)
    upload.upload(upload_files, inputs=upload, outputs=status)

    gr.Markdown("---")

    question = gr.Textbox(label="Question")
    with gr.Row():
        k = gr.Slider(1, 20, value=5, step=1, label="k")
        mode = gr.Dropdown(["hybrid", "vector", "bm25"], value="hybrid", label="Retrieval mode")
    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=4)
    citations_box = gr.Textbox(label="Citations", lines=4)

    ask_btn.click(ask, inputs=[question, k, mode], outputs=[answer_box, citations_box])

    gr.Markdown("---")

    with gr.Row():
        eval_k = gr.Slider(1, 20, value=5, step=1, label="eval k")
        eval_btn = gr.Button("Run retrieval eval (recall@k, MRR, nDCG@k)")
    eval_box = gr.Textbox(label="Eval report", lines=6)
    eval_btn.click(run_eval, inputs=eval_k, outputs=eval_box)


if __name__ == "__main__":
    demo.launch()
