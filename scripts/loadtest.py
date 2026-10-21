#!/usr/bin/env python
"""Load-test the API's /query endpoint against a *simulated* model latency.

What it measures: how the API behaves when each answer waits on a slow language
model. The model is simulated (a fixed sleep) and retrieval runs offline on the
bundled corpus, so the numbers say nothing about real model speed. They show how
the server copes while waiting, which is what the async work changes.

Two variants run the same requests against the real FastAPI app:

  non-blocking   the model call is awaited (what the async client does)
  blocking       the model call blocks the event loop (what a synchronous SDK call
                 inside async code does); shown to make the cost visible

For each variant it reports throughput, request latency, and the event-loop lag:
the longest stretch in which a 10 ms heartbeat task could not run. Lag is what turns
one slow model call into a stall for every other client, including /health.

Usage:
  python scripts/loadtest.py
  python scripts/loadtest.py --latency 0.2 --requests 100 --concurrency 25
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import httpx

from lumen_rag.engine import RagEngine
from lumen_rag.ingestion.loaders import load_file
from lumen_rag.llm import Answer

api = importlib.import_module("lumen_rag.api.app")


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


async def run_variant(name: str, blocking: bool, latency: float, requests: int, concurrency: int) -> dict:
    async def simulated_model(question, chunks):
        if blocking:
            time.sleep(latency)  # noqa: ASYNC251 - deliberately blocks the loop to show the contrast
        else:
            await asyncio.sleep(latency)
        return Answer("simulated answer", [])

    import lumen_rag.engine as engine_module

    engine_module.answer_async = simulated_model

    lags: list[float] = []
    stop = False

    async def heartbeat():
        loop = asyncio.get_running_loop()
        last = loop.time()
        while not stop:
            await asyncio.sleep(0.01)
            now = loop.time()
            lags.append(max(0.0, now - last - 0.01))
            last = now

    latencies: list[float] = []
    gate = asyncio.Semaphore(concurrency)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://load") as client:

        async def one():
            async with gate:
                t0 = time.perf_counter()
                r = await client.post("/query", json={"question": "how many days of annual leave", "k": 3})
                r.raise_for_status()
                latencies.append(time.perf_counter() - t0)

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.05)
        started = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(requests)])
        elapsed = time.perf_counter() - started
        stop = True
        await beat

    return {
        "variant": name,
        "throughput": requests / elapsed,
        "p50": statistics.median(latencies),
        "p95": percentile(latencies, 95),
        "lag_max": max(lags),
        "lag_p95": percentile(lags, 95),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latency", type=float, default=0.1, help="simulated model latency in seconds")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    engine = RagEngine()
    engine.add_documents([load_file(p) for p in sorted((ROOT / "data" / "docs").glob("*.md"))])
    api.engine = engine
    api.enforce = lambda *a, **k: None  # the in-process load would trip the per-IP rate limit

    print(
        f"{args.requests} requests, {args.concurrency} in flight, simulated model latency "
        f"{args.latency * 1000:.0f} ms, offline retrieval on {len(engine.store)} chunks\n"
    )
    rows = [
        await run_variant("non-blocking model call", False, args.latency, args.requests, args.concurrency),
        await run_variant("blocking model call in the loop", True, args.latency, args.requests, args.concurrency),
    ]
    print("| Variant | requests/s | p50 latency | p95 latency | loop lag max | loop lag p95 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['variant']} | {r['throughput']:.1f} | {r['p50'] * 1000:.0f} ms | {r['p95'] * 1000:.0f} ms | "
            f"{r['lag_max'] * 1000:.0f} ms | {r['lag_p95'] * 1000:.0f} ms |"
        )


if __name__ == "__main__":
    asyncio.run(main())
