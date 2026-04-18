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
