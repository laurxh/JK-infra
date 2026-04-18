import pytest
from decision_service.history_store import HistoryStore
from decision_service.schemas import ExecutionProfile


def test_default_correctness():
    store = HistoryStore()
    c = store.expected_correctness("svamp", ExecutionProfile.CHAT_NO_THINK)
    assert 0 < c <= 1.0


def test_record_and_query():
    store = HistoryStore()
    store.record("svamp", ExecutionProfile.CHAT_NO_THINK, correctness=1.0)
    store.record("svamp", ExecutionProfile.CHAT_NO_THINK, correctness=0.0)
    c = store.expected_correctness("svamp", ExecutionProfile.CHAT_NO_THINK)
    assert c == 0.5


def test_different_profiles_independent():
    store = HistoryStore()
    store.record("svamp", ExecutionProfile.CHAT_NO_THINK, correctness=1.0)
    store.record("svamp", ExecutionProfile.RAW, correctness=0.0)
    assert store.expected_correctness("svamp", ExecutionProfile.CHAT_NO_THINK) == 1.0
    assert store.expected_correctness("svamp", ExecutionProfile.RAW) == 0.0


def test_sliding_window():
    store = HistoryStore(window_size=3)
    store.record("t", ExecutionProfile.RAW, correctness=0.0)
    store.record("t", ExecutionProfile.RAW, correctness=0.0)
    store.record("t", ExecutionProfile.RAW, correctness=0.0)
    store.record("t", ExecutionProfile.RAW, correctness=1.0)  # pushes out oldest
    assert store.expected_correctness("t", ExecutionProfile.RAW) == pytest.approx(1 / 3)
