"""Stub inference server for local integration testing.

Mimics the teammate's inference engine (5 endpoints) with configurable
fake latencies so the decision service can be tested end-to-end without a GPU.

Run:  conda run -n copilot python -m decision_service.stub_inference
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
PORT = int(os.getenv("STUB_PORT", "8000"))
THROUGHPUT_TOK_S = float(os.getenv("STUB_THROUGHPUT", "750"))
PREFILL_LATENCY_S = float(os.getenv("STUB_PREFILL_LATENCY", "0.15"))
WARMUP_S = float(os.getenv("STUB_WARMUP", "3"))

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_start_time: float = 0.0
_inflight: int = 0
_inflight_tokens: int = 0
_cache: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.monotonic()
    yield


app = FastAPI(title="Stub Inference Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class GenerateReq(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = Field(default_factory=list)
    # accept & ignore extra fields from decision service
    model_config = {"extra": "allow"}


class LoglikelihoodReq(BaseModel):
    request_id: str
    prompt: str
    continuation: str
    model_config = {"extra": "allow"}


class RollingReq(BaseModel):
    request_id: str
    prompt: str
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    from fastapi.responses import JSONResponse
    elapsed = time.monotonic() - _start_time
    if elapsed < WARMUP_S:
        return JSONResponse(status_code=503, content={"status": "warming_up"})
    return {"status": "ok"}


@app.get("/status")
async def status():
    return {
        "running": {
            "decode_tokens_remaining": _inflight_tokens,
            "task_count": _inflight,
        },
        "waiting": {
            "compute_tokens_remaining": 0,
            "task_count": 0,
        },
    }


@app.post("/generate")
async def generate(req: GenerateReq):
    global _inflight, _inflight_tokens

    if req.request_id in _cache:
        return _cache[req.request_id]

    _inflight += 1
    _inflight_tokens += req.max_tokens
    try:
        delay = req.max_tokens / THROUGHPUT_TOK_S
        await asyncio.sleep(delay)

        resp = {
            "id": req.request_id,
            "text": f"stub response for {req.prompt[:50]}",
            "finish_reason": "stop",
            "prompt_tokens": len(req.prompt) // 4,
            "completion_tokens": req.max_tokens,
        }
        _cache[req.request_id] = resp
        return resp
    finally:
        _inflight -= 1
        _inflight_tokens -= req.max_tokens


@app.post("/loglikelihood")
async def loglikelihood(req: LoglikelihoodReq):
    if req.request_id in _cache:
        return _cache[req.request_id]

    await asyncio.sleep(PREFILL_LATENCY_S)

    resp = {
        "id": req.request_id,
        "accuracy": -random.uniform(0.5, 5.0),
        "prompt_tokens": len(req.prompt) // 4,
        "continuation_tokens": len(req.continuation) // 4,
    }
    _cache[req.request_id] = resp
    return resp


@app.post("/loglikelihood_rolling")
async def loglikelihood_rolling(req: RollingReq):
    if req.request_id in _cache:
        return _cache[req.request_id]

    await asyncio.sleep(PREFILL_LATENCY_S)

    resp = {
        "id": req.request_id,
        "accuracy": -random.uniform(50, 500),
        "prompt_tokens": len(req.prompt) // 4,
    }
    _cache[req.request_id] = resp
    return resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        f"Stub inference server ready on port {PORT} "
        f"(throughput={THROUGHPUT_TOK_S} tok/s, prefill={PREFILL_LATENCY_S}s)"
    )
    uvicorn.run(
        "decision_service.stub_inference:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
