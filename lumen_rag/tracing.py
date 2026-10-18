"""Optional OpenTelemetry tracing.

Nothing here is required: without the OpenTelemetry packages, or without a tracer
provider, ``span()`` costs a couple of function calls and records nothing.

Turn it on with ``LUMEN_TRACING``:

  ``console``  print finished spans to stdout (good for a first look)
  ``otlp``     export over OTLP/HTTP using the standard ``OTEL_EXPORTER_OTLP_*``
               variables; Langfuse, Arize Phoenix, Jaeger and Grafana Tempo all
               accept this. Needs ``pip install 'lumen-rag[tracing]'``.

Spans (all under the tracer ``lumen_rag``):

  lumen.query      one per ``RagEngine.query`` / ``aquery``
  lumen.retrieve   one per retrieval: mode, k, result count, store backend
  lumen.answer     one per generated answer: model, offline flag, token usage

Question and answer *text* is not recorded unless ``LUMEN_TRACE_CONTENT=1``; only
lengths are. Someone else's documents flow through this engine, so the default is to
keep them out of your trace backend.

An application that already configures a global OpenTelemetry provider (for example
with ``opentelemetry-instrument``) gets these spans without setting ``LUMEN_TRACING``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from .config import settings

TRACER_NAME = "lumen_rag"


class _NoopSpan:
    """Stands in for a span when OpenTelemetry is absent."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def end(self) -> None:
        pass


_NOOP = _NoopSpan()
_setup_done = False


def _api():
    """The OpenTelemetry trace API module, or None when it is not installed."""
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    return trace


def content_attr(text: str) -> str | None:
    """The text itself only when the operator opted in; otherwise None (not recorded)."""
    return text if settings.trace_content else None


def set_attr(span: Any, key: str, value: Any) -> None:
    """``span.set_attribute`` that skips None (OpenTelemetry rejects None values)."""
    if value is not None:
        span.set_attribute(key, value)


def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attrs.items() if v is not None}


@contextmanager
def span(name: str, **attributes: Any):
    """Context manager around a span; yields an object with ``set_attribute``.

    Exceptions inside the block are recorded on the span and re-raised.
    """
    api = _api()
    if api is None:
        yield _NOOP
        return
    tracer = api.get_tracer(TRACER_NAME)
    with tracer.start_as_current_span(name, attributes=_clean(attributes)) as current:
        yield current


def start_span(name: str, **attributes: Any):
    """A span that is *not* made current, ended by the caller with ``.end()``.

    For code that spans ``yield`` points (async generators), where attaching and
    detaching a context across iterations is fragile. Parented to the current span.
    """
    api = _api()
    if api is None:
        return _NOOP
    return api.get_tracer(TRACER_NAME).start_span(name, attributes=_clean(attributes))


def setup_from_env() -> bool:
    """Install a tracer provider according to ``LUMEN_TRACING``. Idempotent.

    Returns True when a provider was installed by this call.
    """
    global _setup_done
    kind = settings.tracing.strip().lower()
    if not kind or _setup_done:
        return False
    if kind not in ("console", "otlp"):
        raise ValueError(f"unknown LUMEN_TRACING {settings.tracing!r}; use 'console' or 'otlp'")
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError as e:
        raise RuntimeError(
            "LUMEN_TRACING needs OpenTelemetry. Install it with: pip install 'lumen-rag[tracing]'"
        ) from e

    if kind == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as e:
            raise RuntimeError(
                "LUMEN_TRACING=otlp needs the OTLP exporter. Install it with: pip install 'lumen-rag[tracing]'"
            ) from e
        exporter = OTLPSpanExporter()  # endpoint and headers come from OTEL_EXPORTER_OTLP_* variables
    else:
        exporter = ConsoleSpanExporter()

    provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "lumen-rag")}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _setup_done = True
    return True
