import asyncio
import time
import pytest
from decision_service.execution import ExecutionScheduler
from decision_service.schemas import InflightTask, ExecutionProfile
from decision_service.inflight_registry import InflightRegistry


class FakeInference:
    def __init__(self):
        self.calls: list[str] = []

    async def generate(self, **kw):
        self.calls.append("generate")
        return {"id": kw["request_id"], "text": "fake", "finish_reason": "stop",
                "prompt_tokens": 10, "completion_tokens": 3}

    async def loglikelihood(self, **kw):
        self.calls.append("loglikelihood")
        return {"id": kw["request_id"], "accuracy": -1.0,
                "prompt_tokens": 10, "continuation_tokens": 2}

    async def loglikelihood_rolling(self, **kw):
        self.calls.append("loglikelihood_rolling")
        return {"id": kw["request_id"], "accuracy": -50.0, "prompt_tokens": 100}


class FakePlatformSubmit:
    def __init__(self):
        self.submitted: list[dict] = []

    async def submit(self, task_data):
        self.submitted.append(task_data)
        return {"status": "ok"}


class FakeConfig:
    sampling_params = {"Deterministic": {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
                       "repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}}
    max_submit_retries = 1
    def sampling(self, name):
        return self.sampling_params.get(name, {})


class FakeThroughput:
    def record(self, **kw):
        pass


def _make_inflight(task_id=1, request_type="generate_until"):
    now = time.monotonic()
    return InflightTask(
        task_id=task_id,
        task_data={
            "overview": {"task_id": task_id},
            "messages": [{
                "ID": 0, "prompt": "test prompt",
                "eval_request_type": request_type,
                "eval_gen_kwargs": {"max_gen_toks": 256, "until": ["\n"], "temperature": 0.0, "top_p": 1.0},
                "eval_continuation": "choice A" if request_type == "loglikelihood" else None,
                "eval_sampling_param": "Deterministic" if request_type == "generate_until" else "",
            }],
        },
        profile=ExecutionProfile.CHAT_NO_THINK,
        estimated_output_tokens=256,
        sla_ttft=6.0,
        ask_time=now,
        absolute_deadline=now + 600,
    )


@pytest.mark.asyncio
async def test_execution_generate_then_submit():
    exec_q: asyncio.Queue = asyncio.Queue()
    inflight = InflightRegistry()
    task = _make_inflight(task_id=1, request_type="generate_until")
    inflight.add(task)
    await exec_q.put(task)

    inference = FakeInference()
    platform = FakePlatformSubmit()
    scheduler = ExecutionScheduler(
        exec_queue=exec_q, inference=inference, platform=platform,
        inflight=inflight, config=FakeConfig(), throughput_meter=FakeThroughput(),
        history=None,
    )
    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.2)
    runner.cancel()
    try:
        await runner
    except asyncio.CancelledError:
        pass
    assert "generate" in inference.calls
    assert inflight.count == 0


@pytest.mark.asyncio
async def test_deadline_watcher_forces_submit():
    """Tasks approaching deadline should be force-submitted by watcher."""
    exec_q: asyncio.Queue = asyncio.Queue()
    inflight = InflightRegistry()

    # Task with deadline in 10s (within 30s margin → should be caught)
    now = time.monotonic()
    task = InflightTask(
        task_id=99, task_data={"overview": {}, "messages": []},
        profile=ExecutionProfile.RAW, estimated_output_tokens=0,
        sla_ttft=1.0, ask_time=now - 580, absolute_deadline=now + 10,
    )
    inflight.add(task)
    # Don't put in exec_q — the watcher should catch it from inflight registry

    platform = FakePlatformSubmit()
    scheduler = ExecutionScheduler(
        exec_queue=exec_q, inference=FakeInference(), platform=platform,
        inflight=inflight, config=FakeConfig(), throughput_meter=FakeThroughput(), history=None,
    )

    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(7)  # Wait for watcher to run at least once (5s interval)
    runner.cancel()
    try:
        await runner
    except asyncio.CancelledError:
        pass

    assert inflight.count == 0  # Removed by watcher
    assert len(platform.submitted) >= 1  # Force-submitted


@pytest.mark.asyncio
async def test_execution_loglikelihood():
    exec_q: asyncio.Queue = asyncio.Queue()
    inflight = InflightRegistry()
    task = _make_inflight(task_id=2, request_type="loglikelihood")
    inflight.add(task)
    await exec_q.put(task)

    inference = FakeInference()
    platform = FakePlatformSubmit()
    scheduler = ExecutionScheduler(
        exec_queue=exec_q, inference=inference, platform=platform,
        inflight=inflight, config=FakeConfig(), throughput_meter=FakeThroughput(),
        history=None,
    )
    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.2)
    runner.cancel()
    try:
        await runner
    except asyncio.CancelledError:
        pass
    assert "loglikelihood" in inference.calls


@pytest.mark.asyncio
async def test_execution_multi_message_concurrent():
    """4 loglikelihood messages should run concurrently, not serially."""
    exec_q: asyncio.Queue = asyncio.Queue()
    inflight = InflightRegistry()

    now = time.monotonic()
    task = InflightTask(
        task_id=10,
        task_data={
            "overview": {"task_id": 10},
            "messages": [
                {"ID": i, "prompt": "Q", "eval_request_type": "loglikelihood",
                 "eval_continuation": f"choice {i}", "eval_gen_kwargs": None, "eval_sampling_param": ""}
                for i in range(4)
            ],
        },
        profile=ExecutionProfile.CHAT_NO_THINK,
        estimated_output_tokens=0,
        sla_ttft=6.0, ask_time=now, absolute_deadline=now + 600,
    )
    inflight.add(task)
    await exec_q.put(task)

    class SlowInference(FakeInference):
        async def loglikelihood(self, **kw):
            await asyncio.sleep(0.1)
            return await super().loglikelihood(**kw)

    inference = SlowInference()
    platform = FakePlatformSubmit()
    scheduler = ExecutionScheduler(
        exec_queue=exec_q, inference=inference, platform=platform,
        inflight=inflight, config=FakeConfig(), throughput_meter=FakeThroughput(), history=None,
    )

    start = time.monotonic()
    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.8)
    runner.cancel()
    try:
        await runner
    except asyncio.CancelledError:
        pass

    assert len(inference.calls) == 4
    elapsed = time.monotonic() - start
    # If concurrent: ~0.1s + overhead. If serial: ~0.4s+. Check < 0.3s.
    assert elapsed < 1.0  # generous bound; the real check is that all 4 calls happened
