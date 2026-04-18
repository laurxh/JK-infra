from __future__ import annotations
import time
import logging
from decision_service.schemas import InflightTask

logger = logging.getLogger(__name__)


class InflightRegistry:
    def __init__(self):
        self._tasks: dict[int, InflightTask] = {}

    def add(self, task: InflightTask):
        if task.task_id in self._tasks:
            logger.debug("Duplicate task_id %d, skipping", task.task_id)
            return
        self._tasks[task.task_id] = task

    def remove(self, task_id: int):
        self._tasks.pop(task_id, None)

    def get(self, task_id: int) -> InflightTask | None:
        return self._tasks.get(task_id)

    @property
    def count(self) -> int:
        return len(self._tasks)

    @property
    def total_estimated_output_tokens(self) -> int:
        return sum(t.estimated_output_tokens for t in self._tasks.values())

    def get_overdue_task_ids(self) -> list[int]:
        now = time.monotonic()
        return [tid for tid, t in self._tasks.items() if now > t.absolute_deadline]

    def all_tasks(self) -> list[InflightTask]:
        return list(self._tasks.values())
