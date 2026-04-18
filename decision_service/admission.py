from __future__ import annotations
import asyncio
import time
import logging
from decision_service.schemas import (
    TaskOverview, AdmissionAction, InflightTask, ExecutionProfile,
)
from decision_service.decision_engine import DecisionEngine, DEFAULT_MAX_GEN_TOKS
from decision_service.inflight_registry import InflightRegistry
from decision_service.logging_utils import log_decision

logger = logging.getLogger(__name__)


class AdmissionController:
    def __init__(
        self, *, overview_queue: asyncio.Queue, exec_queue: asyncio.Queue,
        platform, decision_engine: DecisionEngine, stats_poller, throughput_meter,
        inflight: InflightRegistry, config,
    ):
        self._overview_q = overview_queue
        self._exec_q = exec_queue
        self._platform = platform
        self._engine = decision_engine
        self._stats = stats_poller
        self._throughput = throughput_meter
        self._inflight = inflight
        self._cfg = config

    async def run(self):
        while True:
            try:
                overview: TaskOverview = await asyncio.wait_for(
                    self._overview_q.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

            stats = self._stats.latest()
            if stats is None:
                logger.warning("No stats available, dropping task %d", overview.task_id)
                continue

            throughput = self._throughput.current_tok_per_s()
            decision = self._engine.decide(overview, stats, throughput)
            log_decision(overview, decision, stats)

            if decision.action == AdmissionAction.DROP:
                logger.info("DROP task %d: %s", overview.task_id, decision.reason)
                continue

            try:
                result = await self._platform.ask(
                    task_id=overview.task_id, sla=overview.target_sla,
                )
            except Exception as e:
                logger.warning("ask failed for task %d: %s", overview.task_id, e)
                continue

            if result.get("status") != "accepted":
                logger.info("ask rejected for task %d: %s", overview.task_id, result.get("reason", "?"))
                continue

            task_data = result["task"]
            est_out = DEFAULT_MAX_GEN_TOKS if overview.eval_request_type == "generate_until" else 0
            if decision.profile == ExecutionProfile.CHAT_THINK:
                est_out = int(est_out * self._cfg.think_token_multiplier)

            now = time.monotonic()
            inflight_task = InflightTask(
                task_id=overview.task_id,
                task_data=task_data,
                profile=decision.profile or ExecutionProfile.CHAT_NO_THINK,
                estimated_output_tokens=est_out,
                sla_ttft=self._cfg.sla_ttft(overview.target_sla),
                ask_time=now,
                absolute_deadline=now + overview.eval_timeout_s,
            )
            self._inflight.add(inflight_task)
            await self._exec_q.put(inflight_task)
            logger.info("ACCEPTED task %d profile=%s", overview.task_id, decision.profile)
