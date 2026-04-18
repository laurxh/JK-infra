import pytest
import time
from decision_service.schemas import (
    TaskOverview, EngineStats, RunningStats, WaitingStats, ExecutionProfile,
)


@pytest.fixture
def idle_stats():
    return EngineStats(
        running=RunningStats(decode_tokens_remaining=0, task_count=0),
        waiting=WaitingStats(compute_tokens_remaining=0, task_count=0),
    )


@pytest.fixture
def busy_stats():
    return EngineStats(
        running=RunningStats(decode_tokens_remaining=3000, task_count=6),
        waiting=WaitingStats(compute_tokens_remaining=500, task_count=2),
    )


@pytest.fixture
def gen_overview():
    return TaskOverview(
        task_id=1001, target_sla="Gold", target_reward=1.5, max_winners=1,
        eval_task_name="svamp", eval_request_type="generate_until",
        eval_sampling_param="Deterministic", eval_timeout_s=600,
    )


@pytest.fixture
def ll_overview():
    return TaskOverview(
        task_id=2001, target_sla="Supreme", target_reward=0.8, max_winners=1,
        eval_task_name="commonsenseqa", eval_request_type="loglikelihood",
        eval_sampling_param="", eval_timeout_s=600,
    )
