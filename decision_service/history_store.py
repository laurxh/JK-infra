from __future__ import annotations
from collections import defaultdict, deque
from decision_service.schemas import ExecutionProfile

DEFAULT_CORRECTNESS = 0.5


class HistoryStore:
    def __init__(self, window_size: int = 50):
        self._window_size = window_size
        self._data: dict[tuple[str, ExecutionProfile], deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def record(self, task_name: str, profile: ExecutionProfile, correctness: float):
        self._data[(task_name, profile)].append(correctness)

    def expected_correctness(self, task_name: str, profile: ExecutionProfile) -> float:
        key = (task_name, profile)
        samples = self._data.get(key)
        if not samples:
            return DEFAULT_CORRECTNESS
        return sum(samples) / len(samples)
