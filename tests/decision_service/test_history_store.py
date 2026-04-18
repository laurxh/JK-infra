import pytest
from decision_service.history_store import HistoryStore
from decision_service.schemas import ExecutionProfile


def test_default_correctness_by_type():
    store = HistoryStore()
    assert store.expected_correctness("loglikelihood", ExecutionProfile.CHAT_NO_THINK) == 0.7
    assert store.expected_correctness("generate_until", ExecutionProfile.CHAT_NO_THINK) == 0.5
    assert store.expected_correctness("loglikelihood_rolling", ExecutionProfile.RAW) == 0.6
    assert store.expected_correctness("unknown_type", ExecutionProfile.RAW) == 0.5


def test_record_and_query():
    store = HistoryStore()
    store.record("generate_until", ExecutionProfile.CHAT_NO_THINK, correctness=1.0)
    store.record("generate_until", ExecutionProfile.CHAT_NO_THINK, correctness=0.0)
    c = store.expected_correctness("generate_until", ExecutionProfile.CHAT_NO_THINK)
    assert c == 0.5


def test_different_profiles_independent():
    store = HistoryStore()
    store.record("loglikelihood", ExecutionProfile.CHAT_NO_THINK, correctness=1.0)
    store.record("loglikelihood", ExecutionProfile.RAW, correctness=0.0)
    assert store.expected_correctness("loglikelihood", ExecutionProfile.CHAT_NO_THINK) == 1.0
    assert store.expected_correctness("loglikelihood", ExecutionProfile.RAW) == 0.0


def test_sliding_window():
    store = HistoryStore(window_size=3)
    store.record("generate_until", ExecutionProfile.RAW, correctness=0.0)
    store.record("generate_until", ExecutionProfile.RAW, correctness=0.0)
    store.record("generate_until", ExecutionProfile.RAW, correctness=0.0)
    store.record("generate_until", ExecutionProfile.RAW, correctness=1.0)
    assert store.expected_correctness("generate_until", ExecutionProfile.RAW) == pytest.approx(1 / 3)


def test_different_request_types_independent():
    store = HistoryStore()
    store.record("loglikelihood", ExecutionProfile.CHAT_NO_THINK, correctness=0.9)
    store.record("generate_until", ExecutionProfile.CHAT_NO_THINK, correctness=0.3)
    assert store.expected_correctness("loglikelihood", ExecutionProfile.CHAT_NO_THINK) == 0.9
    assert store.expected_correctness("generate_until", ExecutionProfile.CHAT_NO_THINK) == 0.3
