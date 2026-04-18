import pytest
from decision_service.decision_engine import DecisionEngine
from decision_service.schemas import (
    TaskOverview, EngineStats, RunningStats, WaitingStats,
    ExecutionProfile, AdmissionAction,
)
from decision_service.config import DecisionConfig
from decision_service.history_store import HistoryStore
from decision_service.inflight_registry import InflightRegistry


def _make_config(**overrides) -> DecisionConfig:
    defaults = dict(
        platform_url="http://mock:8003", inference_url="http://mock:8000",
        token="t", team_name="team", model_name="Qwen3-32B", model_path="/m",
        contestant_port=9000, duration_s=3600,
        sla_levels={"Bronze": 10.0, "Silver": 8.0, "Gold": 6.0, "Platinum": 4.0,
                    "Diamond": 2.0, "Stellar": 1.5, "Glorious": 0.8, "Supreme": 0.5},
        sampling_params={"Deterministic": {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
                         "repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}},
        safety_margin=1.2, min_ev=0.1, decode_tok_per_s=750.0, prefill_tok_per_s=18000.0,
        logprob_traffic_ratio=0.3,
    )
    defaults.update(overrides)
    return DecisionConfig(**defaults)


def _make_overview(request_type="generate_until", sla="Gold", reward=1.5, **kw) -> TaskOverview:
    defaults = dict(
        task_id=1, target_sla=sla, target_reward=reward, max_winners=1,
        eval_task_name="svamp", eval_request_type=request_type,
        eval_sampling_param="Deterministic" if request_type == "generate_until" else "",
        eval_timeout_s=600,
    )
    defaults.update(kw)
    return TaskOverview(**defaults)


def _make_stats(decode_remaining=0, running_count=0, prefill_remaining=0, waiting_count=0):
    return EngineStats(
        running=RunningStats(decode_tokens_remaining=decode_remaining, task_count=running_count),
        waiting=WaitingStats(compute_tokens_remaining=prefill_remaining, task_count=waiting_count),
    )


class TestFastPath:
    def test_loglikelihood_idle_engine(self):
        engine = DecisionEngine(_make_config(), HistoryStore(), InflightRegistry())
        overview = _make_overview(request_type="loglikelihood", sla="Supreme", reward=0.8)
        d = engine.decide(overview, _make_stats(), throughput_tok_s=750.0)
        assert d.action == AdmissionAction.ASK_NOW

    def test_loglikelihood_busy_engine_still_accepted(self):
        engine = DecisionEngine(_make_config(), HistoryStore(), InflightRegistry())
        overview = _make_overview(request_type="loglikelihood", sla="Supreme", reward=0.8)
        stats = _make_stats(decode_remaining=5000, running_count=10)
        d = engine.decide(overview, stats, throughput_tok_s=750.0)
        assert d.action == AdmissionAction.ASK_NOW

    def test_loglikelihood_rolling_accepted(self):
        engine = DecisionEngine(_make_config(), HistoryStore(), InflightRegistry())
        overview = _make_overview(request_type="loglikelihood_rolling", sla="Gold", reward=1.0)
        d = engine.decide(overview, _make_stats(), throughput_tok_s=750.0)
        assert d.action == AdmissionAction.ASK_NOW


class TestSlowPath:
    def test_generate_idle_engine_gold_sla_accepted(self):
        engine = DecisionEngine(_make_config(), HistoryStore(), InflightRegistry())
        overview = _make_overview(sla="Gold", reward=1.5)
        d = engine.decide(overview, _make_stats(), throughput_tok_s=750.0)
        assert d.action == AdmissionAction.ASK_NOW
        assert d.profile is not None

    def test_generate_overloaded_engine_dropped(self):
        engine = DecisionEngine(_make_config(), HistoryStore(), InflightRegistry())
        overview = _make_overview(sla="Supreme", reward=1.0)
        stats = _make_stats(decode_remaining=10000, running_count=10)
        d = engine.decide(overview, stats, throughput_tok_s=750.0)
        assert d.action == AdmissionAction.DROP
        assert "infeasible" in d.reason

    def test_generate_low_reward_dropped(self):
        cfg = _make_config(min_ev=2.0)
        engine = DecisionEngine(cfg, HistoryStore(), InflightRegistry())
        overview = _make_overview(sla="Bronze", reward=0.5)
        d = engine.decide(overview, _make_stats(), throughput_tok_s=750.0)
        assert d.action == AdmissionAction.DROP
        assert "low_ev" in d.reason

    def test_think_profile_selected_for_math(self):
        engine = DecisionEngine(_make_config(), HistoryStore(), InflightRegistry())
        overview = _make_overview(sla="Bronze", reward=2.0, eval_task_name="gsm8k")
        d = engine.decide(overview, _make_stats(), throughput_tok_s=750.0)
        assert d.action == AdmissionAction.ASK_NOW
        assert d.profile == ExecutionProfile.CHAT_THINK

    def test_think_profile_fallback_on_tight_sla(self):
        engine = DecisionEngine(_make_config(), HistoryStore(), InflightRegistry())
        overview = _make_overview(sla="Glorious", reward=1.0, eval_task_name="gsm8k")
        d = engine.decide(overview, _make_stats(), throughput_tok_s=750.0)
        if d.action == AdmissionAction.ASK_NOW:
            assert d.profile != ExecutionProfile.CHAT_THINK
