"""Index-mutating routes: admin gate, hosted lock, upload/chunking bounds, write lock, seeding."""
import asyncio
import importlib
import time

import httpx
import numpy as np
import pytest

from lumen_rag import config
from lumen_rag.api import ratelimit
from lumen_rag.engine import RagEngine

api = importlib.import_module("lumen_rag.api.app")

_KEY = "test-admin-key"  # test-only value, not a real secret
_DOC = {"documents": [{"id": "d1", "text": "We deploy at four in the afternoon on weekdays."}]}

# One request per guarded route: (path, request kwargs).
_ROUTES = [
    ("/ingest", {"json": _DOC}),
    ("/ingest/upload", {"files": [("files", ("note.txt", b"Robots lift boxes.", "text/plain"))]}),
    ("/ingest/sample", {}),
    ("/reset", {}),
]
_ROUTE_IDS = [path for path, _ in _ROUTES]


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Fresh engine and lock, index persisted under tmp_path, rate limits off."""
    eng = RagEngine()
    monkeypatch.setattr(api, "engine", eng)
    monkeypatch.setattr(api, "index_lock", asyncio.Lock())  # bound to this test's event loop
    monkeypatch.setattr(api, "enforce", lambda *a, **k: None)
    monkeypatch.setattr(config.settings, "index_dir", str(tmp_path / "index"))
    return eng


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")


# --- admin gate ------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "kwargs"), _ROUTES, ids=_ROUTE_IDS)
async def test_key_set_requires_the_key(env, monkeypatch, path, kwargs):
    monkeypatch.setattr(config.settings, "admin_key", _KEY)
    async with _client() as c:
        missing = await c.post(path, **kwargs)
        wrong = await c.post(path, headers={"X-Admin-Key": "nope"}, **kwargs)
        right = await c.post(path, headers={"X-Admin-Key": _KEY}, **kwargs)
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert right.status_code == 200, right.text


@pytest.mark.parametrize(("path", "kwargs"), _ROUTES, ids=_ROUTE_IDS)
async def test_rejected_requests_do_not_touch_the_index(env, monkeypatch, path, kwargs):
    env.add_documents([{"id": "keep", "text": "Existing content stays put."}])
    before = len(api.engine.store)
    monkeypatch.setattr(config.settings, "admin_key", _KEY)
    async with _client() as c:
        r = await c.post(path, headers={"X-Admin-Key": "nope"}, **kwargs)
    assert r.status_code == 401
    assert api.engine is env and len(env.store) == before


@pytest.mark.parametrize(("path", "kwargs"), _ROUTES, ids=_ROUTE_IDS)
async def test_hosted_without_key_is_403(env, monkeypatch, path, kwargs):
    monkeypatch.setattr(config.settings, "hosted", True)
    async with _client() as c:
        r = await c.post(path, **kwargs)
        with_header = await c.post(path, headers={"X-Admin-Key": "anything"}, **kwargs)
    assert r.status_code == 403
    assert "locked" in r.json()["detail"]
    assert with_header.status_code == 403
    assert len(env.store) == 0


@pytest.mark.parametrize(("path", "kwargs"), _ROUTES, ids=_ROUTE_IDS)
async def test_local_without_key_is_open(env, path, kwargs):
    async with _client() as c:
        r = await c.post(path, **kwargs)
    assert r.status_code == 200, r.text


async def test_hosted_with_key_accepts_the_key(env, monkeypatch):
    monkeypatch.setattr(config.settings, "hosted", True)
    monkeypatch.setattr(config.settings, "admin_key", _KEY)
    async with _client() as c:
        r = await c.post("/ingest", json=_DOC, headers={"X-Admin-Key": _KEY})
    assert r.status_code == 200 and r.json()["chunks_indexed"] == 1


async def test_failed_key_attempts_are_throttled(env, monkeypatch):
    monkeypatch.setattr(api, "enforce", ratelimit.enforce)
    monkeypatch.setattr(ratelimit, "_buckets", ratelimit.defaultdict(list))
    monkeypatch.setattr(config.settings, "admin_key", _KEY)
    async with _client() as c:
        codes = [(await c.post("/reset", headers={"X-Admin-Key": "guess"})).status_code for _ in range(6)]
    assert codes == [401] * 5 + [429]


@pytest.mark.parametrize(
    ("admin_key", "hosted", "locked"),
    [("", False, False), ("", True, True), (_KEY, False, True), (_KEY, True, True)],
)
async def test_health_reports_ingest_locked(env, monkeypatch, admin_key, hosted, locked):
    monkeypatch.setattr(config.settings, "admin_key", admin_key)
    monkeypatch.setattr(config.settings, "hosted", hosted)
    async with _client() as c:
        r = await c.get("/health")
    assert r.json()["ingest_locked"] is locked


# --- bounds ----------------------------------------------------------------------------


async def test_oversize_upload_is_413(env, monkeypatch):
    monkeypatch.setattr(config.settings, "max_upload_bytes", 1000)
    async with _client() as c:
        big = await c.post("/ingest/upload", files=[("files", ("big.txt", b"a " * 600, "text/plain"))])
        ok = await c.post("/ingest/upload", files=[("files", ("ok.txt", b"a " * 400, "text/plain"))])
    assert big.status_code == 413 and "upload limit" in big.json()["detail"]
    assert ok.status_code == 200
    assert {c.doc_id for c in env.store.chunks()} == {"ok"}


async def test_upload_larger_than_one_read_chunk_is_counted_across_chunks(env, monkeypatch):
    monkeypatch.setattr(config.settings, "max_upload_bytes", 100_000)
    payload = b"word " * 30_000  # 150 kB: several 64 kB reads, each under the limit on its own
    async with _client() as c:
        r = await c.post("/ingest/upload", files=[("files", ("big.txt", payload, "text/plain"))])
    assert r.status_code == 413


async def test_too_many_files_is_rejected(env, monkeypatch):
    monkeypatch.setattr(config.settings, "max_upload_files", 2)
    files = [("files", (f"f{i}.txt", b"Some text here.", "text/plain")) for i in range(3)]
    async with _client() as c:
        r = await c.post("/ingest/upload", files=files)
        ok = await c.post("/ingest/upload", files=files[:2])
    assert r.status_code == 413 and "At most 2 files" in r.json()["detail"]
    assert ok.status_code == 200 and ok.json()["files_indexed"] == 2


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (5, 0), (10_000, 20), (120, 120), (120, 500), (120, -1)],
)
async def test_bad_chunking_is_422_on_ingest(env, chunk_size, overlap):
    body = _DOC | {"chunk_size": chunk_size, "overlap": overlap}
    async with _client() as c:
        r = await c.post("/ingest", json=body)
    assert r.status_code == 422
    assert len(env.store) == 0


@pytest.mark.parametrize(("chunk_size", "overlap"), [(5, 0), (10_000, 20), (120, 120), (120, -1)])
async def test_bad_chunking_is_422_on_upload(env, chunk_size, overlap):
    async with _client() as c:
        r = await c.post(
            "/ingest/upload",
            params={"chunk_size": chunk_size, "overlap": overlap},
            files=[("files", ("note.txt", b"Robots lift boxes.", "text/plain"))],
        )
    assert r.status_code == 422
    assert len(env.store) == 0


async def test_chunking_bounds_come_from_settings(env, monkeypatch):
    monkeypatch.setattr(config.settings, "max_chunk_size", 50)
    async with _client() as c:
        over = await c.post("/ingest", json=_DOC | {"chunk_size": 60, "overlap": 0})
        at = await c.post("/ingest", json=_DOC | {"chunk_size": 50, "overlap": 0})
    assert over.status_code == 422 and at.status_code == 200


# --- write serialisation ---------------------------------------------------------------


async def test_concurrent_ingests_keep_every_document(env, monkeypatch):
    """The store's add is a read-modify-write; the sleep widens the window a race needs."""
    store = env.store

    def racy_add(chunks, vectors):
        old = store._vectors
        time.sleep(0.01)
        store._vectors = np.vstack([old, vectors.astype(np.float32)])
        store._chunks.extend(chunks)

    monkeypatch.setattr(store, "add", racy_add)
    n = 12
    async with _client() as c:
        responses = await asyncio.gather(
            *[
                c.post("/ingest", json={"documents": [{"id": f"doc{i}", "text": f"Fact number {i} is here."}]})
                for i in range(n)
            ]
        )
    assert all(r.status_code == 200 for r in responses)
    assert {ch.doc_id for ch in store.chunks()} == {f"doc{i}" for i in range(n)}
    assert store._vectors.shape[0] == len(store.chunks()) == n


async def test_reset_waits_for_an_in_flight_ingest(env, monkeypatch):
    """An ingest holding the lock finishes on the old engine before reset swaps it out."""
    order = []
    real_index = api._index_documents

    def slow_index(eng, docs, chunk_size, overlap):
        time.sleep(0.2)
        order.append("ingest")
        return real_index(eng, docs, chunk_size, overlap)

    monkeypatch.setattr(api, "_index_documents", slow_index)
    async with _client() as c:
        ingest = asyncio.create_task(c.post("/ingest", json=_DOC))
        await asyncio.sleep(0.05)
        reset = await c.post("/reset")
        order.append("reset")
        assert (await ingest).status_code == 200
    assert reset.status_code == 200
    assert order == ["ingest", "reset"]
    assert len(api.engine.store) == 0  # reset ran last


# --- startup seeding -------------------------------------------------------------------


async def test_startup_seeds_the_sample_corpus_when_the_index_is_empty(env, monkeypatch):
    monkeypatch.setattr(config.settings, "seed_sample", True)
    async with api.lifespan(api.app):
        seeded = api.engine
    assert len(seeded.store) > 0
    assert {"people-ops", "deploys-ops"} <= {c.doc_id for c in seeded.store.chunks()}


async def test_startup_does_not_seed_a_non_empty_index(env, monkeypatch):
    existing = RagEngine()
    existing.add_documents([{"id": "mine", "text": "Only my own document is indexed."}])
    existing.save()  # to settings.index_dir (tmp_path)
    monkeypatch.setattr(config.settings, "seed_sample", True)
    async with api.lifespan(api.app):
        loaded = api.engine
    assert {c.doc_id for c in loaded.store.chunks()} == {"mine"}


async def test_startup_does_not_seed_when_disabled(env):
    async with api.lifespan(api.app):
        loaded = api.engine
    assert len(loaded.store) == 0
