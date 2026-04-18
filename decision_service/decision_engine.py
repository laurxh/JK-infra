from __future__ import annotations
import logging
from decision_service.schemas import (
    TaskOverview, EngineStats, ExecutionProfile, AdmissionAction, AdmissionDecision,
)
from decision_service.config import DecisionConfig
from decision_service.history_store import HistoryStore
from decision_service.inflight_registry import InflightRegistry

logger = logging.getLogger(__name__)

MATH_TASK_NAMES = {"gsm8k", "math", "minerva_math", "mathqa", "asdiv"}
DEFAULT_MAX_GEN_TOKS = 256


class DecisionEngine:
    def __init__(self, config: DecisionConfig, history: HistoryStore, inflight: InflightRegistry):
        self.cfg = config
        self.history = history
        self.inflight = inflight

    def decide(
        self, overview: TaskOverview, stats: EngineStats, throughput_tok_s: float,
    ) -> AdmissionDecision:
        if overview.eval_request_type in ("loglikelihood", "loglikelihood_rolling"):
            return self._fast_path(overview)
        return self._slow_path(overview, stats, throughput_tok_s)

    def _fast_path(self, overview: TaskOverview) -> AdmissionDecision:
        profile = ExecutionProfile.CHAT_NO_THINK
        if overview.eval_request_type == "loglikelihood_rolling":
            profile = ExecutionProfile.RAW
        c_hat = self.history.expected_correctness(overview.eval_task_name, profile)
        ev = overview.target_reward * c_hat
        return AdmissionDecision(
            action=AdmissionAction.ASK_NOW,
            profile=profile,
            reason="fast_path",
            estimated_finish_s=0.2,
            expected_reward=ev,
        )

    def _slow_path(
        self, overview: TaskOverview, stats: EngineStats, throughput_tok_s: float,
    ) -> AdmissionDecision:
        candidates = self._candidate_profiles(overview)
        sla_ttft = self.cfg.sla_ttft(overview.target_sla)
        inflation = 1.0 + self.cfg.logprob_traffic_ratio

        backlog_decode_s = stats.running.decode_tokens_remaining / max(throughput_tok_s, 1.0)
        backlog_prefill_s = stats.waiting.compute_tokens_remaining / max(self.cfg.prefill_tok_per_s, 1.0)
        backlog_s = backlog_decode_s * inflation + backlog_prefill_s

        inflight_s = self.inflight.total_estimated_output_tokens / max(throughput_tok_s, 1.0) * inflation
        backlog_s += inflight_s * 0.5

        feasible: list[tuple[ExecutionProfile, float, float]] = []
        for profile in candidates:
            est_out = self._estimate_output_tokens(overview, profile)
            self_s = est_out / max(throughput_tok_s, 1.0) * inflation
            finish_s = backlog_s + self_s
            if finish_s * self.cfg.safety_margin > sla_ttft:
                continue
            c_hat = self.history.expected_correctness(overview.eval_task_name, profile)
            ev = overview.target_reward * c_hat
            feasible.append((profile, finish_s, ev))

        if not feasible:
            return AdmissionDecision(
                action=AdmissionAction.DROP, profile=None, reason="sla_infeasible",
            )

        profile, finish_s, ev = max(feasible, key=lambda x: x[2])
        if ev <= self.cfg.min_ev:
            return AdmissionDecision(
                action=AdmissionAction.DROP, profile=None, reason="low_ev",
            )

        return AdmissionDecision(
            action=AdmissionAction.ASK_NOW,
            profile=profile,
            reason=f"ev={ev:.2f},finish={finish_s:.2f}",
            estimated_finish_s=finish_s,
            expected_reward=ev,
        )

    def _candidate_profiles(self, overview: TaskOverview) -> list[ExecutionProfile]:
        task_lower = overview.eval_task_name.lower()
        if any(m in task_lower for m in MATH_TASK_NAMES):
            return [ExecutionProfile.CHAT_THINK, ExecutionProfile.CHAT_NO_THINK, ExecutionProfile.RAW]
        return [ExecutionProfile.CHAT_NO_THINK, ExecutionProfile.RAW]

    def _estimate_output_tokens(self, overview: TaskOverview, profile: ExecutionProfile) -> int:
        base = DEFAULT_MAX_GEN_TOKS
        if profile == ExecutionProfile.CHAT_THINK:
            return int(base * self.cfg.think_token_multiplier)
        return base
