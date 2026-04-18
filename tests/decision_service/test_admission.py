import asyncio
import pytest
from decision_service.admission import AdmissionController
from decision_service.schemas import (
    TaskOverview, EngineStats, RunningStats, WaitingStats,
    ExecutionProfile, AdmissionAction, InflightTask,
)
from decision_service.decision_engine import DecisionEngine
from decision_service.config import DecisionConfig
from decision_service.history_store import HistoryStore
from decision_service.inflight_registry import InflightRegistry


def _make_config(**overrides) -> DecisionConfig:
    defaults = dict(
        platform_url="http://mock:8003", inference_url="http://mock:8000",
        token="t", team_name="team", model_name="Qwen3-32B", model_path="/m",
        contestant_port=9000, duration_s=3600,
        sla_levels={"Bronze": 10.0, "Silver": 8.0, "Gold": 6.0, "Platinum": 4.0,
                    "Diamond": 2.0, "Stellar": 1.5, "Glorious": 0.8, "Supreme": 0.5},
        sampling_params={"Deterministic": {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
                         "repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}},
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
            return {"status": "accepted", "task": {"overview": {}, "messages": [{"ID": 0, "prompt": "test"}]}}
        return {"status": "rejected", "reason": "expired"}


class FakeStatsPoller:
    def latest(self):
        return EngineStats(
            running=RunningStats(decode_tokens_remaining=0, task_count=0),
            waiting=WaitingStats(compute_tokens_remaining=0, task_count=0),
        )

    @property
    def is_healthy(self):
        return True


class FakeThroughputMeter:
    def current_tok_per_s(self, now=None):
        return 750.0


@pytest.mark.asyncio
async def test_admission_accepts_and_pushes(gen_overview):
    overview_q: asyncio.Queue = asyncio.Queue()
    exec_q: asyncio.Queue = asyncio.Queue()
    await overview_q.put(gen_overview)

    cfg = _make_config()
    platform = FakePlatformAsk(accept=True)
    engine = DecisionEngine(cfg, HistoryStore(), InflightRegistry())
    ctrl = AdmissionController(
        overview_queue=overview_q, exec_queue=exec_q,
        platform=platform, decision_engine=engine,
        stats_poller=FakeStatsPoller(), throughput_meter=FakeThroughputMeter(),
        inflight=InflightRegistry(), config=cfg,
    )
    task = asyncio.create_task(ctrl.run())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert exec_q.qsize() == 1
    assert platform.asked == [gen_overview.task_id]


@pytest.mark.asyncio
async def test_admission_drops_when_rejected(gen_overview):
    overview_q: asyncio.Queue = asyncio.Queue()
    exec_q: asyncio.Queue = asyncio.Queue()
    await overview_q.put(gen_overview)

    cfg = _make_config()
    platform = FakePlatformAsk(accept=False)
    engine = DecisionEngine(cfg, HistoryStore(), InflightRegistry())
    ctrl = AdmissionController(
        overview_queue=overview_q, exec_queue=exec_q,
        platform=platform, decision_engine=engine,
        stats_poller=FakeStatsPoller(), throughput_meter=FakeThroughputMeter(),
        inflight=InflightRegistry(), config=cfg,
    )
    task = asyncio.create_task(ctrl.run())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert exec_q.qsize() == 0
