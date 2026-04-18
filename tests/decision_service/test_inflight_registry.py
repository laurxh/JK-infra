import time
import pytest
from decision_service.inflight_registry import InflightRegistry
from decision_service.schemas import InflightTask, ExecutionProfile


def _make_task(task_id: int, est_tokens: int = 256, timeout_s: float = 600) -> InflightTask:
    now = time.monotonic()
    return InflightTask(
        task_id=task_id,
        task_data={"overview": {}, "messages": []},
        profile=ExecutionProfile.CHAT_NO_THINK,
        estimated_output_tokens=est_tokens,
        sla_ttft=6.0,
        ask_time=now,
        absolute_deadline=now + timeout_s,
    )


def test_add_and_remove():
    reg = InflightRegistry()
    t = _make_task(1)
    reg.add(t)
    assert reg.count == 1
    assert reg.total_estimated_output_tokens == 256
    reg.remove(1)
    assert reg.count == 0
    assert reg.total_estimated_output_tokens == 0


def test_remove_nonexistent_is_noop():
    reg = InflightRegistry()
    reg.remove(999)
    assert reg.count == 0


def test_total_estimated_output_tokens():
    reg = InflightRegistry()
    reg.add(_make_task(1, est_tokens=100))
    reg.add(_make_task(2, est_tokens=200))
    reg.add(_make_task(3, est_tokens=300))
    assert reg.total_estimated_output_tokens == 600
    reg.remove(2)
    assert reg.total_estimated_output_tokens == 400


def test_overdue_tasks():
    reg = InflightRegistry()
    now = time.monotonic()
    t1 = InflightTask(
        task_id=1, task_data={}, profile=ExecutionProfile.RAW,
        estimated_output_tokens=10, sla_ttft=1.0,
        ask_time=now - 700, absolute_deadline=now - 100,
    )
    t2 = _make_task(2)
    reg.add(t1)
    reg.add(t2)
    overdue = reg.get_overdue_task_ids()
    assert 1 in overdue
    assert 2 not in overdue


def test_dedup():
    reg = InflightRegistry()
    t = _make_task(1)
    reg.add(t)
    reg.add(t)
    assert reg.count == 1
