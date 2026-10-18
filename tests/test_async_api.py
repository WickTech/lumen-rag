"""The API must not block its event loop, and the model client must be pooled."""
import asyncio
import importlib
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

from lumen_rag import config, llm
from lumen_rag.engine import RagEngine
from lumen_rag.llm import Answer
from lumen_rag.store import Chunk, ScoredChunk

# Not `import lumen_rag.api.app as ...`: lumen_rag.api re-exports `app`, which shadows the module.
api = importlib.import_module("lumen_rag.api.app")

_DOCS = [
    {"id": "leave", "text": "Employees receive thirty working days of annual paid leave."},
    {"id": "robots", "text": "The H200 robot lifts thirty five kilograms to the top shelf."},
]


@pytest.fixture
def seeded(monkeypatch):
    eng = RagEngine()
    eng.add_documents(_DOCS, chunk_size=50, overlap=0)
    monkeypatch.setattr(api, "engine", eng)
    monkeypatch.setattr(api, "enforce", lambda *a, **k: None)  # many requests from one test client
    return eng


class LoopLag:
    """Heartbeat task that records the longest time the event loop went without running it.

    A coroutine that blocks the loop (time.sleep, a sync SDK call) shows up as a gap
    as long as the block, however the request itself is timed.
    """

    async def __aenter__(self):
        self.max_gap = 0.0
        self._stop = False
        self._task = asyncio.create_task(self._beat())
        await asyncio.sleep(0.03)  # let the heartbeat establish its rhythm
        self.max_gap = 0.0
        return self

    async def _beat(self):
        loop = asyncio.get_running_loop()
        last = loop.time()
        while not self._stop:
            await asyncio.sleep(0.01)
            now = loop.time()
            self.max_gap = max(self.max_gap, now - last - 0.01)
            last = now

    async def __aexit__(self, *exc):
        self._stop = True
        await self._task


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")


async def test_query_returns_an_answer_with_citations(seeded):
    async with _client() as c:
        r = await c.post("/query", json={"question": "how many days of annual leave", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["doc_id"] == "leave"
    assert "offline extractive answer" in body["answer"]


async def test_empty_index_is_a_409(monkeypatch):
    monkeypatch.setattr(api, "engine", RagEngine())
    monkeypatch.setattr(api, "enforce", lambda *a, **k: None)
    async with _client() as c:
        r = await c.post("/query", json={"question": "anything"})
    assert r.status_code == 409


async def test_slow_model_calls_overlap(seeded, monkeypatch):
    """20 requests that each wait 0.25 s on the model finish together, not one after another."""

    async def slow_answer(question, chunks):
        await asyncio.sleep(0.25)
        return Answer("ok", [])

    monkeypatch.setattr("lumen_rag.engine.answer_async", slow_answer)
    async with _client() as c:
        started = time.perf_counter()
        responses = await asyncio.gather(
            *[c.post("/query", json={"question": "annual leave days"}) for _ in range(20)]
        )
        elapsed = time.perf_counter() - started
    assert all(r.status_code == 200 for r in responses)
    assert elapsed < 1.5, f"{elapsed:.2f}s: calls were serialised (sequential would take 5s)"


async def test_event_loop_stays_free_during_slow_retrieval(seeded, monkeypatch):
    """Blocking retrieval runs in a worker thread, so the loop keeps running other work."""
    real_retrieve = seeded.retriever.retrieve

    def slow_retrieve(*args, **kwargs):
        time.sleep(0.4)  # blocking on purpose: stands in for a DB or embedding-API round trip
        return real_retrieve(*args, **kwargs)

    monkeypatch.setattr(seeded.retriever, "retrieve", slow_retrieve)
    async with _client() as c, LoopLag() as lag:
        responses = await asyncio.gather(
            c.post("/query/stream", json={"question": "annual leave days"}),
            c.post("/query", json={"question": "annual leave days"}),
            c.get("/health"),
        )
    assert all(r.status_code == 200 for r in responses)
    assert lag.max_gap < 0.15, f"event loop stalled for {lag.max_gap:.2f}s during retrieval"


async def test_upload_parsing_and_indexing_run_off_the_loop(seeded, monkeypatch):
    def slow_load(path):
        time.sleep(0.4)  # blocking parser
        return {"id": "up", "text": "Uploaded text about warehouse robots.", "metadata": {}}

    monkeypatch.setattr(api, "load_file", slow_load)
    monkeypatch.setattr(RagEngine, "save", lambda self, directory=None: None)
    async with _client() as c, LoopLag() as lag:
        response = await c.post("/ingest/upload", files=[("files", ("note.txt", b"hello", "text/plain"))])
    assert response.status_code == 200 and response.json()["files_indexed"] == 1
    assert lag.max_gap < 0.15, f"event loop stalled for {lag.max_gap:.2f}s during an upload"


async def test_stream_sends_tokens_then_done(seeded):
    async with _client() as c:
        r = await c.post("/query/stream", json={"question": "annual leave days", "k": 2})
    events = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert len(events) >= 3
    assert '"done": true' in events[-1] and "leave" in events[-1]


# --- online (API key) paths, with a fake OpenAI SDK ---------------------------------


class _FakeResponse:
    def __init__(self, text="The answer is thirty [1].", usage=(11, 7)):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        self.usage = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1])


def _chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))])


@pytest.fixture
def online(monkeypatch):
    """Pretend an API key is set and count how many SDK clients get constructed."""
    monkeypatch.setattr(config.settings, "openai_api_key", "sk-test", raising=False)
    llm.reset_clients()
    built = {"sync": 0, "async": 0}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built["sync"] += 1
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            if kwargs.get("stream"):
                return iter([_chunk("Thirty "), _chunk("days"), _chunk(None)])
            return _FakeResponse()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            built["async"] += 1
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        async def _create(self, **kwargs):
            if kwargs.get("stream"):
                async def gen():
                    yield SimpleNamespace(choices=[])  # providers can send empty-choice keep-alives
                    for part in ("Thirty ", "days"):
                        yield _chunk(part)
                return gen()
            return _FakeResponse()

    # A stand-in module: the real SDK is an optional extra and may not be installed (it is not in CI).
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeAsyncOpenAI))
    yield built
    llm.reset_clients()


def _chunks():
    # Built by hand: with a (fake) API key set, RagEngine would pick the OpenAI embedder.
    return [ScoredChunk(Chunk(f"c{i}", d["text"], d["id"], {}), 1.0 - i / 10) for i, d in enumerate(_DOCS)]


def test_blocking_client_is_built_once_and_reused(online):
    chunks = _chunks()
    for _ in range(3):
        assert llm.answer("how many days?", chunks).text == "The answer is thirty [1]."
    list(llm.answer_stream("how many days?", chunks))
    assert online["sync"] == 1


async def test_async_client_is_built_once_and_shared_by_concurrent_calls(online):
    chunks = _chunks()
    results = await asyncio.gather(*[llm.answer_async("how many days?", chunks) for _ in range(5)])
    assert {r.text for r in results} == {"The answer is thirty [1]."}
    assert online["async"] == 1


async def test_async_stream_yields_tokens_then_citations_and_skips_empty_choices(online):
    parts = [(t, c) async for t, c in llm.answer_stream_async("how many days?", _chunks())]
    assert [t for t, c in parts[:-1]] == ["Thirty ", "days"]
    assert parts[-1][0] == "" and parts[-1][1][0]["doc_id"] == "leave"


async def test_no_context_short_circuits_without_a_client(online):
    assert (await llm.answer_async("anything", [])).citations == []
    first = await anext(llm.answer_stream_async("anything", []))
    assert first[0].startswith("I don't have")
    assert online == {"sync": 0, "async": 0}
