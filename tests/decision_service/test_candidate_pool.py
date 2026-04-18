import time
import pytest
from decision_service.candidate_pool import CandidatePool
from decision_service.schemas import TaskOverview


def _make_overview(task_id: int, request_type: str = "generate_until",
                   reward: float = 1.0, sla: str = "Gold",
                   age_s: float = 0.0) -> TaskOverview:
    return TaskOverview(
        task_id=task_id,
        target_sla=sla,
        target_reward=reward,
        max_winners=1,
        eval_task_name="unknown",
        eval_request_type=request_type,
        eval_sampling_param="",
        eval_timeout_s=600,
        queried_at=time.monotonic() - age_s,
    )


class TestBasicOperations:
    def test_add_and_size(self):
        pool = CandidatePool()
        pool.add(_make_overview(1))
        pool.add(_make_overview(2))
        assert pool.size == 2

    def test_dedup(self):
        pool = CandidatePool()
        o = _make_overview(1)
        pool.add(o)
        pool.add(o)
        assert pool.size == 1

    def test_remove(self):
        pool = CandidatePool()
        pool.add(_make_overview(1))
        pool.remove(1)
        assert pool.size == 0

    def test_remove_nonexistent(self):
        pool = CandidatePool()
        pool.remove(999)  # should not raise
        assert pool.size == 0


class TestPurgeStale:
    def test_purge_old(self):
        pool = CandidatePool(staleness_s=10.0)
        pool.add(_make_overview(1, age_s=20.0))  # 20s old, stale
        pool.add(_make_overview(2, age_s=1.0))   # 1s old, fresh
        purged = pool.purge_stale()
        assert purged == 1
        assert pool.size == 1

    def test_purge_nothing_fresh(self):
        pool = CandidatePool(staleness_s=60.0)
        pool.add(_make_overview(1, age_s=5.0))
        assert pool.purge_stale() == 0


class TestPickBest:
    def test_empty_pool(self):
        pool = CandidatePool()
        assert pool.pick_best(5, 5) == []

    def test_no_slots(self):
        pool = CandidatePool()
        pool.add(_make_overview(1))
        assert pool.pick_best(5, 0) == []

    def test_loglikelihood_prioritized_over_generate(self):
        pool = CandidatePool()
        pool.add(_make_overview(1, request_type="generate_until", reward=10.0))
        pool.add(_make_overview(2, request_type="loglikelihood", reward=1.0))
        picks = pool.pick_best(1, 1)
        assert len(picks) == 1
        assert picks[0].task_id == 2  # loglikelihood picked first despite lower reward

    def test_same_type_highest_reward_first(self):
        pool = CandidatePool()
        pool.add(_make_overview(1, request_type="loglikelihood", reward=5.0))
        pool.add(_make_overview(2, request_type="loglikelihood", reward=10.0))
        pool.add(_make_overview(3, request_type="loglikelihood", reward=1.0))
        picks = pool.pick_best(2, 2)
        assert picks[0].task_id == 2  # highest reward
        assert picks[1].task_id == 1

    def test_budget_limit(self):
        pool = CandidatePool()
        for i in range(10):
            pool.add(_make_overview(i, request_type="loglikelihood", reward=float(i)))
        picks = pool.pick_best(3, 3)
        assert len(picks) == 3

    def test_mixed_types_priority(self):
        pool = CandidatePool()
        pool.add(_make_overview(1, request_type="generate_until", reward=100.0))
        pool.add(_make_overview(2, request_type="loglikelihood_rolling", reward=5.0))
        pool.add(_make_overview(3, request_type="loglikelihood", reward=1.0))
        picks = pool.pick_best(3, 3)
        assert picks[0].eval_request_type == "loglikelihood"
        assert picks[1].eval_request_type == "loglikelihood_rolling"
        assert picks[2].eval_request_type == "generate_until"
