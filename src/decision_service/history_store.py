from __future__ import annotations
from collections import defaultdict, deque
from decision_service.schemas import ExecutionProfile

# Defaults by request_type (before any data collected)
DEFAULT_CORRECTNESS_BY_TYPE: dict[str, float] = {
    "loglikelihood": 0.7,
    "loglikelihood_rolling": 0.6,
    "generate_until": 0.5,
}
DEFAULT_CORRECTNESS = 0.5


class HistoryStore:
    """Tracks (request_type, profile) → rolling average correctness."""

    def __init__(self, window_size: int = 50):
        self._window_size = window_size
        self._data: dict[tuple[str, ExecutionProfile], deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def record(self, request_type: str, profile: ExecutionProfile, correctness: float):
        """Record a correctness observation after submit."""
        self._data[(request_type, profile)].append(correctness)

    def expected_correctness(self, request_type: str, profile: ExecutionProfile) -> float:
        """Get expected correctness for a (request_type, profile) pair."""
        key = (request_type, profile)
        samples = self._data.get(key)
        if not samples:
            return DEFAULT_CORRECTNESS_BY_TYPE.get(request_type, DEFAULT_CORRECTNESS)
        return sum(samples) / len(samples)
