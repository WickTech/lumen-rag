"""Answer synthesis from retrieved context, with inline citations.

Offline (no API key), ``answer`` returns an extractive answer: the top chunks
stitched together with citation markers. This keeps the full RAG loop runnable
and testable end-to-end without a model. With a key, it calls the chat model
using a grounded, citation-enforcing prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .store import ScoredChunk

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the numbered "
    "context passages. Cite sources inline as [1], [2]. If the context does not "
    "contain the answer, say you don't know."
)


@dataclass
class Answer:
    text: str
    citations: list[dict]  # [{"n": 1, "doc_id": ..., "source": ..., "score": ...}]


def _format_context(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(f"[{i + 1}] {c.chunk.text}" for i, c in enumerate(chunks))


def _citations(chunks: list[ScoredChunk]) -> list[dict]:
    return [
        {
            "n": i + 1,
            "doc_id": c.chunk.doc_id,
            "source": c.chunk.metadata.get("source"),
            "score": round(c.score, 4),
        }
        for i, c in enumerate(chunks)
    ]


def answer(question: str, chunks: list[ScoredChunk]) -> Answer:
    if not chunks:
        return Answer("I don't have any indexed context to answer that.", [])

    citations = _citations(chunks)

    if settings.offline:
        # Extractive fallback: surface the best passages with markers.
        body = " ".join(f"{c.chunk.text} [{i + 1}]" for i, c in enumerate(chunks[:2]))
        return Answer(f"(offline extractive answer) {body}", citations)

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{_format_context(chunks)}\n\nQuestion: {question}",
            },
        ],
        temperature=0.1,
    )
    return Answer(resp.choices[0].message.content or "", citations)
