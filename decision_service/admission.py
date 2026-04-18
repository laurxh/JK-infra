from __future__ import annotations
import asyncio
import time
import logging
from decision_service.schemas import (
    TaskOverview, InflightTask, ExecutionProfile,
)
from decision_service.candidate_pool import CandidatePool
from decision_service.inflight_registry import InflightRegistry
from decision_service.logging_utils import log_decision
from decision_service.schemas import AdmissionAction, AdmissionDecision

logger = logging.getLogger(__name__)

# Engine concurrency limit — configurable
MAX_ENGINE_TASKS = 8
# Reward threshold for thinking mode
HIGH_REWARD_THRESHOLD = 1000.0
# Max candidates to /ask per selector round
MAX_PICKS_PER_ROUND = 5
# Default output token estimate (before /ask reveals real max_gen_toks)
DEFAULT_OUTPUT_ESTIMATE = 256


class AdmissionController:
    def __init__(
        self, *, overview_queue: asyncio.Queue, exec_queue: asyncio.Queue,
        platform, stats_poller, throughput_meter,
        inflight: InflightRegistry, config,
    ):
        self._overview_q = overview_queue
        self._exec_q = exec_queue
        self._platform = platform
        self._stats = stats_poller
        self._throughput = throughput_meter
        self._inflight = inflight
        self._cfg = config
        self._pool = CandidatePool(staleness_s=60.0)

    @property
    def pool(self) -> CandidatePool:
        return self._pool

    async def run(self):
        while True:
            try:
                # 1. Drain new overviews into pool (non-blocking)
                self._drain_queue_to_pool()

                # 2. Purge stale candidates
                self._pool.purge_stale()

                # 3. Check engine gate
                stats = self._stats.latest()
                if stats is None:
                    await asyncio.sleep(0.2)
                    continue

                available_slots = MAX_ENGINE_TASKS - stats.total_tasks - self._inflight.count
                if available_slots <= 0:
                    await asyncio.sleep(0.2)
                    continue

                # 4. Pick best candidates
                picks = self._pool.pick_best(
                    max_picks=MAX_PICKS_PER_ROUND,
                    available_slots=available_slots,
                )

                if not picks:
                    await asyncio.sleep(0.1)
                    continue

                # 5. Ask each pick
                for overview in picks:
                    profile = self._select_profile(overview)
                    decision = AdmissionDecision(
                        action=AdmissionAction.ASK_NOW,
                        profile=profile,
                        reason=f"pool_pick:rt={overview.eval_request_type},reward={overview.target_reward:.0f}",
                    )
                    log_decision(overview, decision, stats)

                    try:
                        result = await self._platform.ask(
                            task_id=overview.task_id, sla=overview.target_sla,
                        )
                    except Exception as e:
                        logger.warning("ask failed for task %d: %s", overview.task_id, e)
                        self._pool.remove(overview.task_id)
                        continue

                    if result.get("status") != "accepted":
                        logger.info("ask rejected for task %d: %s",
                                    overview.task_id, result.get("reason", "?"))
                        self._pool.remove(overview.task_id)
                        continue

                    # Success — build InflightTask and push to exec_queue
                    self._pool.remove(overview.task_id)
                    task_data = result["task"]

                    est_out = self._estimate_output_tokens(overview, profile)

                    now = time.monotonic()
                    inflight_task = InflightTask(
                        task_id=overview.task_id,
                        task_data=task_data,
                        profile=profile,
                        estimated_output_tokens=est_out,
                        sla_ttft=self._cfg.sla_ttft(overview.target_sla),
                        ask_time=now,
                        absolute_deadline=now + (overview.eval_timeout_s or 600),
                    )
                    self._inflight.add(inflight_task)
                    await self._exec_q.put(inflight_task)
                    logger.info("ACCEPTED task %d profile=%s reward=%.0f",
                                overview.task_id, profile.value, overview.target_reward)

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Admission error: %s", e, exc_info=True)
                await asyncio.sleep(0.5)

    def _drain_queue_to_pool(self):
        """Non-blocking drain of overview_queue into candidate pool."""
        while True:
            try:
                overview = self._overview_q.get_nowait()
                self._pool.add(overview)
            except asyncio.QueueEmpty:
                break

    def _select_profile(self, overview: TaskOverview) -> ExecutionProfile:
        """Select execution profile based on strong features (not task_name)."""
        rt = overview.eval_request_type or "generate_until"

        # loglikelihood types — no thinking needed
        if rt == "loglikelihood":
            return ExecutionProfile.CHAT_NO_THINK
        if rt == "loglikelihood_rolling":
            return ExecutionProfile.RAW

        # generate_until — profile depends on SLA headroom + reward
        sla = self._cfg.sla_ttft(overview.target_sla)
        reward = overview.target_reward

        if sla >= 6.0 and reward >= HIGH_REWARD_THRESHOLD:
            return ExecutionProfile.CHAT_THINK
        elif sla >= 4.0:
            return ExecutionProfile.CHAT_NO_THINK
        else:
            return ExecutionProfile.RAW

    def _estimate_output_tokens(self, overview: TaskOverview, profile: ExecutionProfile) -> int:
        """Pre-ask estimate. Will be corrected after /ask with real max_gen_toks."""
        if overview.eval_request_type != "generate_until":
            return 0
        base = DEFAULT_OUTPUT_ESTIMATE
        if profile == ExecutionProfile.CHAT_THINK:
            base = int(base * self._cfg.think_token_multiplier)
        return base
