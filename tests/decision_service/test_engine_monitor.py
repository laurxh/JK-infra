import asyncio
import time
import pytest

from decision_service.engine_monitor import EngineStatsPoller, ThroughputMeter
from decision_service.schemas import EngineStats, RunningStats, WaitingStats


class FakeInferenceClient:
    def __init__(self, stats_sequence: list[EngineStats]):
        self._stats = stats_sequence
        self._idx = 0

    async def status(self) -> EngineStats:
        s = self._stats[min(self._idx, len(self._stats) - 1)]
        self._idx += 1
        return s


@pytest.mark.asyncio
async def test_poller_caches_latest_stats():
    stats1 = EngineStats(
        running=RunningStats(decode_tokens_remaining=100, task_count=1),
        waiting=WaitingStats(compute_tokens_remaining=0, task_count=0),
    )
    stats2 = EngineStats(
        running=RunningStats(decode_tokens_remaining=50, task_count=1),
        waiting=WaitingStats(compute_tokens_remaining=0, task_count=0),
    )
    fake = FakeInferenceClient([stats1, stats2])
    poller = EngineStatsPoller(fake, poll_interval_s=0.05)
    task = asyncio.create_task(poller.run())
    try:
        await asyncio.sleep(0.12)
        latest = poller.latest()
        assert latest is not None
        assert latest.running.decode_tokens_remaining <= 100
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def test_throughput_meter_sliding_window():
    meter = ThroughputMeter(window_s=1.0)
    now = time.monotonic()
    meter.record(completion_tokens=100, wall_time_s=0.5, timestamp=now)
    meter.record(completion_tokens=200, wall_time_s=0.5, timestamp=now + 0.3)
    assert meter.current_tok_per_s(now + 0.5) == pytest.approx(300.0, rel=0.01)


def test_throughput_meter_expires_old():
    meter = ThroughputMeter(window_s=1.0)
    now = time.monotonic()
    meter.record(completion_tokens=100, wall_time_s=0.5, timestamp=now)
    meter.record(completion_tokens=200, wall_time_s=0.5, timestamp=now + 2.0)
    assert meter.current_tok_per_s(now + 2.5) == pytest.approx(400.0, rel=0.01)


def test_throughput_meter_empty_returns_default():
    meter = ThroughputMeter(window_s=1.0)
    assert meter.current_tok_per_s() == 750.0
