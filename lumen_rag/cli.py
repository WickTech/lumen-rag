"""Command-line interface: ingest, ask, eval, serve."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

# Ensure Unicode output (arrows, box-drawing) doesn't crash on Windows' cp1252 console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from .config import settings
from .engine import RagEngine
from .eval import evaluate
from .eval.harness import load_cases
from .ingestion.loaders import _LOADERS, load_file
from .retrieval import Retriever, RetrievalMode

app = typer.Typer(help="Lumen RAG — ingest, ask, and evaluate a RAG pipeline.")


@app.command()
def ingest(
    paths: list[str] = typer.Argument(..., help="Files or directories (.txt/.md/.html/.pdf/.docx/.pptx)."),
    chunk_size: int = 120,
    overlap: int = 20,
) -> None:
    """Index documents into the persistent vector store."""
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in _LOADERS:
                files.extend(path.rglob(f"*{ext}"))
        else:
            files.append(path)

    engine = (
        RagEngine.load() if Path(settings.index_dir, "chunks.json").exists() else RagEngine()
    )
    # Go through the loaders so PDF/DOCX/HTML (and Docling, when enabled) are
    # parsed rather than read as raw bytes.
    docs = [load_file(f) for f in files]
    total = engine.add_documents(docs, chunk_size=chunk_size, overlap=overlap)
    engine.save()
    typer.echo(f"Indexed {len(files)} file(s) → {total} chunks in {settings.index_dir}/")


@app.command()
def ask(
    question: str,
    k: int = 5,
    mode: str = typer.Option("hybrid", help="Retrieval mode: vector | bm25 | hybrid"),
) -> None:
    """Query the index and print an answer with citations."""
    engine = RagEngine.load()
    result = engine.query(question, k=k, mode=mode)  # type: ignore[arg-type]
    typer.echo("\n" + result.text + "\n")
    typer.echo("Sources:")
    for c in result.citations:
        typer.echo(f"  [{c['n']}] {c['doc_id']} (score={c['score']})")


@app.command(name="eval")
def run_eval(dataset: str, k: int = 5) -> None:
    """Score the retriever against a JSONL of labelled questions."""
    engine = RagEngine.load()
    cases = load_cases(dataset)
    report = evaluate(Retriever(engine.store, engine.embedder), cases, k=k)
    typer.echo("\n" + report.pretty() + "\n")
    typer.echo(json.dumps(report.as_dict()))


@app.command(name="eval-answers")
def run_answer_eval(
    dataset: str,
    k: int = 5,
    mode: str = typer.Option("hybrid", help="Retrieval mode: vector | bm25 | hybrid"),
    judge: str = typer.Option("gpt-4o-mini", help="Judge model for DeepEval metrics."),
    threshold: float = typer.Option(0.5, help="Per-metric pass threshold (0-1)."),
    out: str = typer.Option("", help="Optional path to write per-case JSON results."),
) -> None:
    """Judge generated answers (faithfulness, relevancy, ...) with DeepEval."""
    from .eval.answer_quality import evaluate_answers

    engine = RagEngine.load()
    cases = load_cases(dataset)
    try:
        report = evaluate_answers(
            engine, cases, k=k, mode=mode, threshold=threshold, judge_model=judge  # type: ignore[arg-type]
        )
    except (ImportError, RuntimeError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo("\n" + report.pretty() + "\n")
    typer.echo(json.dumps(report.as_dict()))
    if out:
        Path(out).write_text(json.dumps(report.per_case, indent=2), encoding="utf-8")
        typer.echo(f"Per-case results → {out}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("lumen_rag.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
