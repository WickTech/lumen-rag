"""In-memory sliding-window limiter for the cost-incurring endpoints.

Single-process only — fine for the local FastAPI/uvicorn deployment and the
single-container HF Space this app actually runs as (see Dockerfile). Would
need a shared store (Redis) behind a multi-instance deployment.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

_buckets: dict[str, list[float]] = defaultdict(list)


def check(bucket_key: str, limit: int, window_s: float) -> bool:
    """Returns True if `bucket_key` is currently over `limit` hits in the
    trailing `window_s` window; otherwise records this hit and returns False.
    No FastAPI dependency, so both the API app and the Gradio Space (which
    doesn't pull in FastAPI directly) can use this.
    """
    now = time.monotonic()
    hits = _buckets[bucket_key]
    cutoff = now - window_s
    while hits and hits[0] < cutoff:
        hits.pop(0)

    if len(hits) >= limit:
        return True

    hits.append(now)
    return False


def enforce(request: Request, key: str, limit: int, window_s: float) -> None:
    from fastapi import HTTPException

    ip = request.client.host if request.client else "unknown"
    if check(f"{key}:{ip}", limit, window_s):
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly.")
