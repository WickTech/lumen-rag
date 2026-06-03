"""Tests for the streaming /query/stream endpoint."""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import lumen_rag.api.app as api_module
from lumen_rag.api.app import app
from lumen_rag.engine import RagEngine

_DOC = {"id": "d1", "text": "Engineers get 25 vacation days per year."}


@pytest.fixture
def client():
    """TestClient with a seeded engine. Seeds AFTER lifespan so we override it."""
    with TestClient(app) as c:
        eng = RagEngine()
        eng.add_documents([_DOC])
        api_module.engine = eng
        yield c


def test_stream_returns_sse_events(client):
    resp = client.post("/query/stream", json={"question": "How many vacation days?", "k": 3})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
    assert len(lines) >= 2  # at least one token + final done event

    events = [json.loads(ln[len("data: "):]) for ln in lines]
    done_events = [e for e in events if e.get("done")]
    token_events = [e for e in events if "token" in e]

    assert len(done_events) == 1
    assert "citations" in done_events[0]
    assert len(token_events) >= 1


def test_stream_409_when_empty(client):
    # Simulate empty index by patching store length on the live engine.
    with patch.object(type(api_module.engine.store), "__len__", return_value=0):
        resp = client.post("/query/stream", json={"question": "anything", "k": 3})
    assert resp.status_code == 409
