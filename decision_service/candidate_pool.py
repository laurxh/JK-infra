from __future__ import annotations
import time
import logging
from decision_service.schemas import TaskOverview

logger = logging.getLogger(__name__)


class CandidatePool:
    """Buffers task overviews, buckets by (request_type, SLA, sampling_param), picks by reward."""

    def __init__(self, staleness_s: float = 60.0):
        self._pool: dict[int, TaskOverview] = {}
        self._staleness = staleness_s

    def add(self, overview: TaskOverview):
        if overview.task_id not in self._pool:
            self._pool[overview.task_id] = overview

    def remove(self, task_id: int):
        self._pool.pop(task_id, None)

    @property
    def size(self) -> int:
        return len(self._pool)

    def purge_stale(self) -> int:
        now = time.monotonic()
        stale = [tid for tid, o in self._pool.items()
                 if now - o.queried_at > self._staleness]
        for tid in stale:
            del self._pool[tid]
        return len(stale)

    def pick_best(self, max_picks: int, available_slots: int) -> list[TaskOverview]:
        """Pick up to min(max_picks, available_slots) candidates.

        Priority order:
        1. loglikelihood (cheapest — pure prefill)
        2. loglikelihood_rolling
        3. generate_until (most expensive — occupies decode slots)

        Within each request_type group, pick highest reward first.
        """
        if available_slots <= 0 or max_picks <= 0:
            return []

        budget = min(max_picks, available_slots)
        picks: list[TaskOverview] = []

        # Group by request_type, then sort each group by -reward
        by_type: dict[str, list[TaskOverview]] = {}
        for o in self._pool.values():
            rt = o.eval_request_type or "generate_until"
            by_type.setdefault(rt, []).append(o)

        for rt in ["loglikelihood", "loglikelihood_rolling", "generate_until"]:
            if rt not in by_type:
                continue
            candidates = sorted(by_type[rt], key=lambda o: -o.target_reward)
            for c in candidates:
                if len(picks) >= budget:
                    return picks
                picks.append(c)

        return picks
