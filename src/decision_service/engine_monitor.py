from __future__ import annotations
import asyncio
import time
import logging
from dataclasses import dataclass, field
from decision_service.schemas import EngineStats

logger = logging.getLogger(__name__)

DEFAULT_THROUGHPUT = 750.0


class EngineStatsPoller:
    def __init__(self, inference_client, poll_interval_s: float = 0.2):
        self._client = inference_client
        self._interval = poll_interval_s
        self._latest: EngineStats | None = None
        self._healthy = True

    def latest(self) -> EngineStats | None:
        return self._latest

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    async def run(self):
        while True:
            try:
                self._latest = await self._client.status()
                self._healthy = True
            except Exception as e:
                logger.warning("Failed to poll engine stats: %s", e)
                self._healthy = False
            await asyncio.sleep(self._interval)


@dataclass
class _Sample:
    completion_tokens: int
    wall_time_s: float
    timestamp: float


class ThroughputMeter:
    def __init__(self, window_s: float = 5.0, default_tok_per_s: float = DEFAULT_THROUGHPUT):
        self._window_s = window_s
        self._default = default_tok_per_s
        self._samples: list[_Sample] = []

    def record(self, completion_tokens: int, wall_time_s: float, timestamp: float | None = None):
        ts = timestamp if timestamp is not None else time.monotonic()
        self._samples.append(_Sample(completion_tokens, wall_time_s, ts))

    def current_tok_per_s(self, now: float | None = None) -> float:
        now = now if now is not None else time.monotonic()
        cutoff = now - self._window_s
        self._samples = [s for s in self._samples if s.timestamp >= cutoff]
        if not self._samples:
            return self._default
        total_tokens = sum(s.completion_tokens for s in self._samples)
        total_time = sum(s.wall_time_s for s in self._samples)
        if total_time <= 0:
            return self._default
        return total_tokens / total_time
