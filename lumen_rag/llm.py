"""Answer synthesis from retrieved context, with inline citations.

Offline (no API key), ``answer`` returns an extractive answer: the top chunks
stitched together with citation markers. This keeps the full RAG loop runnable
and testable end-to-end without a model. With a key, it calls the chat model
using a grounded, citation-enforcing prompt.

Every entry point has a blocking and an ``async`` form. The API uses the async
ones so a slow model call never occupies a worker thread or stalls the event loop;
both share one cached, connection-pooled client per process instead of building a
new client (and a new TLS handshake) for every request.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from . import tracing
from .config import settings
from .store import ScoredChunk

if TYPE_CHECKING:
    from .reader import Reading

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the numbered "
    "context passages. Cite sources inline as [1], [2]. If the context does not "
    "contain the answer, say you don't know."
)

_NO_CONTEXT = "I don't have any indexed context to answer that."

ABSTAIN_TEXT = "I can't find this in the indexed documents, so I won't guess."


@dataclass
class Answer:
    text: str
    citations: list[dict]  # [{"n": 1, "doc_id": ..., "source": ..., "score": ...}]
    # True when the engine refused because retrieval confidence was below
    # LUMEN_ABSTAIN_THRESHOLD; no model was called.
    abstained: bool = False
    # Retrieval confidence, set only when abstention is enabled.
    confidence: float | None = None
    # Reader answerability (best span minus no-answer score), set only with LUMEN_READER.
    answerability: float | None = None


def abstain(confidence: float) -> Answer:
    return Answer(ABSTAIN_TEXT, [], abstained=True, confidence=confidence)


def _format_context(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(f"[{i + 1}] {c.chunk.text}" for i, c in enumerate(chunks))


def _citations(chunks: list[ScoredChunk]) -> list[dict]:
    return [
        {
            "n": i + 1,
            "doc_id": c.chunk.doc_id,
            "source": c.chunk.metadata.get("source"),
            # Contextual header ("Document: ... Section: ..."), when ingestion added one.
            "context": c.chunk.metadata.get("context"),
            "score": round(c.score, 4),
        }
        for i, c in enumerate(chunks)
    ]


def _messages(question: str, chunks: list[ScoredChunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{_format_context(chunks)}\n\nQuestion: {question}"},
    ]


def _offline_text(chunks: list[ScoredChunk], reading: Reading | None = None) -> str:
    # With a reader, answer with the sentence holding the extracted span.
    if reading is not None and reading.sentence and reading.chunk_index is not None:
        return f"(offline extractive answer) {reading.sentence} [{reading.chunk_index + 1}]"
    # Otherwise surface the best passages with markers.
    body = " ".join(f"{c.chunk.text} [{i + 1}]" for i, c in enumerate(chunks[:2]))
    return f"(offline extractive answer) {body}"


@lru_cache(maxsize=1)
def _client():
    """One blocking client per process: it owns an HTTP connection pool."""
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


@lru_cache(maxsize=1)
def _async_client():
    """One async client per process (its pool is bound to the loop that first uses it)."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def reset_clients() -> None:
    """Drop the cached clients, e.g. after changing the API key or base URL at runtime."""
    _client.cache_clear()
    _async_client.cache_clear()


def _record_usage(span, usage) -> None:
    """Attach OpenTelemetry GenAI token counts when the provider reported them."""
    if usage is None:
        return
    tracing.set_attr(span, "gen_ai.usage.input_tokens", getattr(usage, "prompt_tokens", None))
    tracing.set_attr(span, "gen_ai.usage.output_tokens", getattr(usage, "completion_tokens", None))


def answer(question: str, chunks: list[ScoredChunk], reading: Reading | None = None) -> Answer:
    if not chunks:
        return Answer(_NO_CONTEXT, [])

    citations = _citations(chunks)
    with tracing.span(
        "lumen.answer", **{"lumen.offline": settings.offline, "lumen.context_chunks": len(chunks)}
    ) as span:
        if settings.offline:
            text = _offline_text(chunks, reading)
        else:
            span.set_attribute("gen_ai.request.model", settings.chat_model)
            resp = _client().chat.completions.create(
                model=settings.chat_model, messages=_messages(question, chunks), temperature=0.1
            )
            _record_usage(span, getattr(resp, "usage", None))
            text = resp.choices[0].message.content or ""
        span.set_attribute("lumen.answer.chars", len(text))
        tracing.set_attr(span, "lumen.answer.text", tracing.content_attr(text))
    return Answer(text, citations)


async def answer_async(question: str, chunks: list[ScoredChunk], reading: Reading | None = None) -> Answer:
    """``answer`` without blocking the event loop."""
    if not chunks:
        return Answer(_NO_CONTEXT, [])

    citations = _citations(chunks)
    with tracing.span(
        "lumen.answer", **{"lumen.offline": settings.offline, "lumen.context_chunks": len(chunks)}
    ) as span:
        if settings.offline:
            text = _offline_text(chunks, reading)
        else:
            span.set_attribute("gen_ai.request.model", settings.chat_model)
            resp = await _async_client().chat.completions.create(
                model=settings.chat_model, messages=_messages(question, chunks), temperature=0.1
            )
            _record_usage(span, getattr(resp, "usage", None))
            text = resp.choices[0].message.content or ""
        span.set_attribute("lumen.answer.chars", len(text))
        tracing.set_attr(span, "lumen.answer.text", tracing.content_attr(text))
    return Answer(text, citations)


def answer_stream(
    question: str, chunks: list[ScoredChunk], reading: Reading | None = None
) -> Iterator[tuple[str, list | None]]:
    """Yield (token: str, citations: list | None) tuples.

    The final tuple carries citations; all prior tuples have citations=None.
    Callers can use this to build an SSE stream without buffering the full answer.
    Blocking: use ``answer_stream_async`` inside an event loop.
    """
    if not chunks:
        yield (_NO_CONTEXT, _citations(chunks))
        return

    citations = _citations(chunks)

    if settings.offline:
        for word in _offline_text(chunks, reading).split(" "):
            yield (word + " ", None)
        yield ("", citations)
        return

    stream = _client().chat.completions.create(
        model=settings.chat_model, messages=_messages(question, chunks), temperature=0.1, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield (delta, None)
    yield ("", citations)


async def answer_stream_async(
    question: str, chunks: list[ScoredChunk], reading: Reading | None = None
) -> AsyncIterator[tuple[str, list | None]]:
    """Async twin of ``answer_stream``: the same (token, citations) tuples, never blocking the loop."""
    if not chunks:
        yield (_NO_CONTEXT, _citations(chunks))
        return

    citations = _citations(chunks)
    # Not made "current": this generator yields, and a context attached here would be
    # detached from a different context on cancellation.
    span = tracing.start_span(
        "lumen.answer",
        **{"lumen.offline": settings.offline, "lumen.context_chunks": len(chunks), "lumen.streaming": True},
    )
    n_chars = 0
    try:
        if settings.offline:
            for word in _offline_text(chunks, reading).split(" "):
                n_chars += len(word) + 1
                yield (word + " ", None)
                await asyncio.sleep(0)  # let other requests run between tokens
        else:
            span.set_attribute("gen_ai.request.model", settings.chat_model)
            stream = await _async_client().chat.completions.create(
                model=settings.chat_model, messages=_messages(question, chunks), temperature=0.1, stream=True
            )
            async for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if delta:
                    n_chars += len(delta)
                    yield (delta, None)
        span.set_attribute("lumen.answer.chars", n_chars)
        yield ("", citations)
    finally:
        span.end()
