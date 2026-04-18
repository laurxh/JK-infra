import asyncio
import pytest
from decision_service.admission import AdmissionController
from decision_service.schemas import (
    TaskOverview, EngineStats, RunningStats, WaitingStats,
    ExecutionProfile, InflightTask,
)
from decision_service.config import DecisionConfig
from decision_service.inflight_registry import InflightRegistry
import time


def _make_config(**overrides) -> DecisionConfig:
    defaults = dict(
        platform_url="http://mock:8003", inference_url="http://mock:8000",
        token="t", team_name="team", model_name="Qwen3-32B", model_path="/m",
        contestant_port=9000, duration_s=3600,
        sla_levels={"Bronze": 10.0, "Silver": 8.0, "Gold": 6.0, "Platinum": 4.0,
                    "Diamond": 2.0, "Stellar": 1.5, "Glorious": 0.8, "Supreme": 0.5},
        sampling_params={},
        safety_margin=1.2, min_ev=0.1, decode_tok_per_s=750.0, prefill_tok_per_s=18000.0,
        logprob_traffic_ratio=0.3,
    )
    defaults.update(overrides)
    return DecisionConfig(**defaults)


class FakePlatformAsk:
    def __init__(self, accept: bool = True):
        self._accept = accept
        self.asked: list[int] = []

    async def ask(self, task_id: int, sla: str) -> dict:
        self.asked.append(task_id)
        if self._accept:
            return {"status": "accepted", "task": {
                "overview": {}, "messages": [{"ID": 0, "prompt": "test",
                "eval_request_type": "generate_until", "eval_gen_kwargs": {"max_gen_toks": 256}}]
            }}
        return {"status": "rejected", "reason": "expired"}


class FakeStatsPoller:
    def __init__(self, total_tasks=0):
        self._total = total_tasks

    def latest(self):
        return EngineStats(
            running=RunningStats(decode_tokens_remaining=0, task_count=self._total),
            waiting=WaitingStats(compute_tokens_remaining=0, task_count=0),
        )


class FakeThroughputMeter:
    def current_tok_per_s(self, now=None):
        return 750.0


def _make_overview(task_id=1001, request_type="generate_until", reward=1.5, sla="Gold"):
    return TaskOverview(
        task_id=task_id, target_sla=sla, target_reward=reward, max_winners=1,
        eval_task_name="unknown", eval_request_type=request_type,
        eval_sampling_param="", eval_timeout_s=600,
    )


@pytest.mark.asyncio
async def test_admission_picks_from_pool_and_asks():
    overview_q: asyncio.Queue = asyncio.Queue()
    exec_q: asyncio.Queue = asyncio.Queue()
    await overview_q.put(_make_overview(task_id=1, reward=5.0))
    await overview_q.put(_make_overview(task_id=2, reward=10.0))

    cfg = _make_config()
    platform = FakePlatformAsk(accept=True)
    ctrl = AdmissionController(
        overview_queue=overview_q, exec_queue=exec_q,
        platform=platform, stats_poller=FakeStatsPoller(),
        throughput_meter=FakeThroughputMeter(),
        inflight=InflightRegistry(), config=cfg,
    )
    task = asyncio.create_task(ctrl.run())
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Both should be asked (pool has room, engine has slots)
    assert exec_q.qsize() == 2
    assert set(platform.asked) == {1, 2}


@pytest.mark.asyncio
async def test_admission_drops_when_platform_rejects():
    overview_q: asyncio.Queue = asyncio.Queue()
    exec_q: asyncio.Queue = asyncio.Queue()
    await overview_q.put(_make_overview(task_id=1))

    cfg = _make_config()
    platform = FakePlatformAsk(accept=False)
    ctrl = AdmissionController(
        overview_queue=overview_q, exec_queue=exec_q,
        platform=platform, stats_poller=FakeStatsPoller(),
        throughput_meter=FakeThroughputMeter(),
        inflight=InflightRegistry(), config=cfg,
    )
    task = asyncio.create_task(ctrl.run())
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert exec_q.qsize() == 0
    assert ctrl.pool.size == 0  # removed from pool after rejection


@pytest.mark.asyncio
async def test_admission_waits_when_engine_full():
    overview_q: asyncio.Queue = asyncio.Queue()
    exec_q: asyncio.Queue = asyncio.Queue()
    await overview_q.put(_make_overview(task_id=1))

    cfg = _make_config(max_engine_tasks=8)
    platform = FakePlatformAsk(accept=True)
    # Engine reports 8 tasks running = at limit
    ctrl = AdmissionController(
        overview_queue=overview_q, exec_queue=exec_q,
        platform=platform, stats_poller=FakeStatsPoller(total_tasks=8),
        throughput_meter=FakeThroughputMeter(),
        inflight=InflightRegistry(), config=cfg,
    )
    task = asyncio.create_task(ctrl.run())
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Should NOT have asked — engine was full
    assert exec_q.qsize() == 0
    assert len(platform.asked) == 0
    # But overview should still be in pool (waiting for slots)
    assert ctrl.pool.size == 1


@pytest.mark.asyncio
async def test_admission_loglikelihood_gets_correct_profile():
    overview_q: asyncio.Queue = asyncio.Queue()
    exec_q: asyncio.Queue = asyncio.Queue()
    await overview_q.put(_make_overview(task_id=1, request_type="loglikelihood"))

    cfg = _make_config()
    platform = FakePlatformAsk(accept=True)
    ctrl = AdmissionController(
        overview_queue=overview_q, exec_queue=exec_q,
        platform=platform, stats_poller=FakeStatsPoller(),
        throughput_meter=FakeThroughputMeter(),
        inflight=InflightRegistry(), config=cfg,
    )
    task = asyncio.create_task(ctrl.run())
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert exec_q.qsize() == 1
    inflight_task = await exec_q.get()
    assert inflight_task.profile == ExecutionProfile.RAW


@pytest.mark.asyncio
async def test_admission_corrects_inflight_after_ask():
    """After /ask, real max_gen_toks should correct the inflight estimate."""
    overview_q: asyncio.Queue = asyncio.Queue()
    exec_q: asyncio.Queue = asyncio.Queue()
    await overview_q.put(_make_overview(task_id=1, request_type="generate_until", reward=5.0))

    cfg = _make_config()

    class PlatformWithRealTokens:
        asked = []
        async def ask(self, task_id, sla):
            self.asked.append(task_id)
            return {"status": "accepted", "task": {
                "overview": {}, "messages": [{
                    "ID": 0, "prompt": "test",
                    "eval_request_type": "generate_until",
                    "eval_gen_kwargs": {"max_gen_toks": 64, "until": ["\n"], "temperature": 0.0, "top_p": 1.0},
                }]
            }}

    inflight = InflightRegistry()
    platform = PlatformWithRealTokens()
    ctrl = AdmissionController(
        overview_queue=overview_q, exec_queue=exec_q,
        platform=platform, stats_poller=FakeStatsPoller(),
        throughput_meter=FakeThroughputMeter(),
        inflight=inflight, config=cfg,
    )
    task = asyncio.create_task(ctrl.run())
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert exec_q.qsize() == 1
    # The inflight task should have corrected estimate (64, not default 256)
    inflight_task = inflight.get(1) or (await exec_q.get())
    assert inflight_task.estimated_output_tokens == 64
