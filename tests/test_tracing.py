"""Optional OpenTelemetry tracing: span tree, attributes, privacy, and the no-op path."""
import sys
from types import SimpleNamespace

import pytest

from lumen_rag import config, llm, tracing
from lumen_rag.engine import RagEngine

pytest.importorskip("opentelemetry.sdk.trace")
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

_DOCS = [
    {"id": "leave", "text": "Employees receive thirty working days of annual paid leave."},
    {"id": "robots", "text": "The H200 robot lifts thirty five kilograms to the top shelf."},
]


@pytest.fixture
def spans(monkeypatch):
    """Route lumen spans to an in-memory exporter without touching the global provider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    real_api = tracing._api()
    fake = SimpleNamespace(get_tracer=lambda name: provider.get_tracer(name))
    # Keep every other attribute of the real API module reachable.
    monkeypatch.setattr(tracing, "_api", lambda: SimpleNamespace(**{**vars(real_api), "get_tracer": fake.get_tracer}))
    return exporter


@pytest.fixture
def engine():
    eng = RagEngine()
    eng.add_documents(_DOCS, chunk_size=50, overlap=0)
    return eng


def _by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_query_produces_a_query_retrieve_answer_tree(spans, engine):
    engine.query("how many days of annual leave", k=2, mode="bm25")
    s = _by_name(spans)
    assert set(s) == {"lumen.query", "lumen.retrieve", "lumen.answer"}
    root = s["lumen.query"]
    assert root.parent is None
    assert s["lumen.retrieve"].parent.span_id == root.context.span_id
    assert s["lumen.answer"].parent.span_id == root.context.span_id
    assert root.attributes["lumen.mode"] == "bm25" and root.attributes["lumen.k"] == 2
    assert root.attributes["lumen.chunks"] == 2
    assert s["lumen.retrieve"].attributes["lumen.results"] == 2
    assert s["lumen.retrieve"].attributes["lumen.store"] == "VectorStore"
    assert s["lumen.answer"].attributes["lumen.offline"] is True
    assert s["lumen.answer"].attributes["lumen.answer.chars"] > 0


async def test_async_query_keeps_retrieval_a_child_across_the_worker_thread(spans, engine):
    await engine.aquery("how many days of annual leave", k=2)
    s = _by_name(spans)
    assert s["lumen.retrieve"].parent.span_id == s["lumen.query"].context.span_id
    assert s["lumen.answer"].parent.span_id == s["lumen.query"].context.span_id


def test_question_and_answer_text_are_not_recorded_by_default(spans, engine, monkeypatch):
    monkeypatch.setattr(config.settings, "trace_content", False, raising=False)
    engine.query("how many days of annual leave", k=2)
    attrs = {k: v for sp in spans.get_finished_spans() for k, v in sp.attributes.items()}
    assert attrs["lumen.question.chars"] == len("how many days of annual leave")
    assert "lumen.question.text" not in attrs and "lumen.answer.text" not in attrs
    assert not any("annual leave" in str(v) for v in attrs.values())


def test_text_is_recorded_only_when_opted_in(spans, engine, monkeypatch):
    monkeypatch.setattr(config.settings, "trace_content", True, raising=False)
    engine.query("how many days of annual leave", k=2)
    attrs = {k: v for sp in spans.get_finished_spans() for k, v in sp.attributes.items()}
    assert attrs["lumen.question.text"] == "how many days of annual leave"
    assert "annual paid leave" in attrs["lumen.answer.text"]


def test_errors_are_recorded_and_re_raised(spans, engine, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(engine.retriever, "_retrieve", boom)
    with pytest.raises(RuntimeError, match="index unavailable"):
        engine.query("anything")
    s = _by_name(spans)
    assert s["lumen.retrieve"].status.status_code == StatusCode.ERROR
    assert s["lumen.query"].status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in s["lumen.query"].events)


async def test_streaming_answer_span_ends_even_when_the_client_disconnects(spans, engine):
    chunks = engine.retriever.retrieve("annual leave days", k=2)
    stream = llm.answer_stream_async("annual leave days", chunks)
    await anext(stream)  # take one token, then drop the stream like a disconnecting client
    await stream.aclose()
    answer = _by_name(spans)["lumen.answer"]
    assert answer.attributes["lumen.streaming"] is True
    assert answer.end_time is not None


async def test_streaming_answer_span_covers_the_whole_stream(spans, engine):
    chunks = engine.retriever.retrieve("annual leave days", k=2)
    pieces = [t async for t, _ in llm.answer_stream_async("annual leave days", chunks)]
    assert _by_name(spans)["lumen.answer"].attributes["lumen.answer.chars"] == sum(len(p) for p in pieces)


def test_token_usage_is_recorded_for_online_answers(spans, monkeypatch):
    monkeypatch.setattr(config.settings, "openai_api_key", "sk-test", raising=False)
    llm.reset_clients()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            message = SimpleNamespace(content="Thirty [1].")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=9),
            )

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    from lumen_rag.store import Chunk, ScoredChunk

    llm.answer("how many days?", [ScoredChunk(Chunk("c", "Thirty days.", "d", {}), 1.0)])
    attrs = _by_name(spans)["lumen.answer"].attributes
    llm.reset_clients()
    assert attrs["gen_ai.usage.input_tokens"] == 120 and attrs["gen_ai.usage.output_tokens"] == 9
    assert attrs["gen_ai.request.model"] == config.settings.chat_model
    assert attrs["lumen.offline"] is False


def test_everything_still_works_when_opentelemetry_is_missing(engine, monkeypatch):
    monkeypatch.setattr(tracing, "_api", lambda: None)
    assert engine.query("annual leave", k=1).citations
    with tracing.span("x", a=1) as sp:
        sp.set_attribute("k", "v")  # accepted and ignored
    assert tracing.start_span("y") is not None


def test_none_attribute_values_are_dropped(spans):
    with tracing.span("probe", keep=1, drop=None) as sp:
        tracing.set_attr(sp, "also_drop", None)
        tracing.set_attr(sp, "also_keep", 2)
    attrs = _by_name(spans)["probe"].attributes
    assert dict(attrs) == {"keep": 1, "also_keep": 2}


def test_setup_from_env_validates_the_setting(monkeypatch):
    monkeypatch.setattr(tracing, "_setup_done", False)
    monkeypatch.setattr(config.settings, "tracing", "", raising=False)
    assert tracing.setup_from_env() is False
    monkeypatch.setattr(config.settings, "tracing", "bogus", raising=False)
    with pytest.raises(ValueError, match="unknown LUMEN_TRACING"):
        tracing.setup_from_env()


def test_setup_from_env_installs_a_console_provider_once(monkeypatch):
    installed = []
    monkeypatch.setattr("opentelemetry.trace.set_tracer_provider", installed.append)
    monkeypatch.setattr(tracing, "_setup_done", False)
    monkeypatch.setattr(config.settings, "tracing", "console", raising=False)
    assert tracing.setup_from_env() is True
    assert tracing.setup_from_env() is False  # idempotent
    assert len(installed) == 1 and isinstance(installed[0], TracerProvider)


def test_otlp_without_the_exporter_explains_the_extra(monkeypatch):
    monkeypatch.setattr(tracing, "_setup_done", False)
    monkeypatch.setattr(config.settings, "tracing", "otlp", raising=False)
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    with pytest.raises(RuntimeError, match=r"lumen-rag\[tracing\]"):
        tracing.setup_from_env()


def test_otlp_export_reaches_a_collector_over_http(monkeypatch, engine):
    """End to end: LUMEN_TRACING=otlp posts protobuf spans to /v1/traces on the configured endpoint."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append((self.path, self.headers.get("Content-Type"), body))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.end_headers()

        def log_message(self, *args):  # keep test output clean
            pass

    server = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    installed = []
    monkeypatch.setattr("opentelemetry.trace.set_tracer_provider", installed.append)
    monkeypatch.setattr(tracing, "_setup_done", False)
    monkeypatch.setattr(config.settings, "tracing", "otlp", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "lumen-test")
    try:
        assert tracing.setup_from_env() is True
        provider = installed[0]
        real_api = tracing._api()
        monkeypatch.setattr(
            tracing,
            "_api",
            lambda: SimpleNamespace(**{**vars(real_api), "get_tracer": provider.get_tracer}),
        )
        engine.query("how many days of annual leave", k=1)
        assert provider.force_flush(timeout_millis=5000)
    finally:
        server.shutdown()
        provider.shutdown()
    assert received, "the collector never got a request"
    path, content_type, body = received[0]
    assert path == "/v1/traces"
    assert content_type == "application/x-protobuf"
    assert b"lumen.query" in body and b"lumen.retrieve" in body and b"lumen-test" in body
    assert b"how many days of annual leave" not in body  # content capture is off by default
