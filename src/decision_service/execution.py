from __future__ import annotations
import asyncio
import time
import logging
from decision_service.schemas import InflightTask, ExecutionProfile
from decision_service.prompt_renderer import render_prompt
from decision_service.logging_utils import log_submit

logger = logging.getLogger(__name__)


class ExecutionScheduler:
    def __init__(self, *, exec_queue: asyncio.Queue, inference, platform,
                 inflight, config, throughput_meter, history):
        self._exec_q = exec_queue
        self._inference = inference
        self._platform = platform
        self._inflight = inflight
        self._cfg = config
        self._throughput = throughput_meter
        self._history = history

    async def run(self):
        watcher = asyncio.create_task(self._deadline_watcher())
        try:
            while True:
                try:
                    task: InflightTask = await asyncio.wait_for(
                        self._exec_q.get(), timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                asyncio.create_task(self._process(task))
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    async def _deadline_watcher(self):
        """Periodically check for tasks approaching absolute timeout. Force-submit to avoid -2×R_i."""
        DEADLINE_MARGIN_S = 30  # submit this many seconds before absolute deadline
        CHECK_INTERVAL_S = 5
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL_S)
                now = time.monotonic()
                for task in self._inflight.all_tasks():
                    remaining = task.absolute_deadline - now
                    if remaining < DEADLINE_MARGIN_S:
                        logger.warning(
                            "Deadline approaching for task %d (%.1fs left), forcing submit",
                            task.task_id, remaining,
                        )
                        try:
                            await self._platform.submit(task.task_data)
                        except Exception as e:
                            logger.error("Forced submit failed for task %d: %s", task.task_id, e)
                        self._inflight.remove(task.task_id)
                        log_submit(task.task_id, False, now - task.ask_time,
                                   task.profile.value if task.profile else None)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Deadline watcher error: %s", e)

    async def _process(self, task: InflightTask):
        try:
            await self._execute_and_submit(task)
        except Exception as e:
            logger.error("Failed processing task %d: %s", task.task_id, e, exc_info=True)
        finally:
            self._inflight.remove(task.task_id)

    async def _execute_and_submit(self, task: InflightTask):
        task_data = task.task_data
        messages = task_data.get("messages", [])

        async def _infer_one(msg):
            rt = msg.get("eval_request_type", "")
            prompt = msg.get("prompt", "")
            rendered = render_prompt(prompt, task.profile)
            req_id = f"t{task.task_id}_m{msg.get('ID', 0)}"

            try:
                _msg_start = time.monotonic()
                if rt == "generate_until":
                    gen_kwargs = msg.get("eval_gen_kwargs") or {}
                    # Q98: eval_gen_kwargs contains ALL sampling params
                    result = await self._inference.generate(
                        request_id=req_id, prompt=rendered,
                        max_tokens=gen_kwargs.get("max_gen_toks", 256),
                        temperature=gen_kwargs.get("temperature", 0.0),
                        top_p=gen_kwargs.get("top_p", 1.0),
                        top_k=gen_kwargs.get("top_k", 1),
                        repetition_penalty=gen_kwargs.get("repetition_penalty", 1.0),
                        frequency_penalty=gen_kwargs.get("frequency_penalty", 0.0),
                        presence_penalty=gen_kwargs.get("presence_penalty", 0.0),
                        stop=gen_kwargs.get("until", []),
                    )
                    msg["response"] = result.get("text", "")
                    self._throughput.record(
                        completion_tokens=result.get("completion_tokens", 0),
                        wall_time_s=time.monotonic() - _msg_start,
                    )
                elif rt == "loglikelihood":
                    continuation = msg.get("eval_continuation", "")
                    result = await self._inference.loglikelihood(
                        request_id=req_id, prompt=rendered, continuation=continuation,
                    )
                    msg["accuracy"] = result.get("accuracy", 0.0)
                elif rt == "loglikelihood_rolling":
                    result = await self._inference.loglikelihood_rolling(
                        request_id=req_id, prompt=rendered,
                    )
                    msg["accuracy"] = result.get("accuracy", 0.0)
            except Exception as e:
                logger.error("Inference failed for %s: %s", req_id, e)
                if rt == "generate_until":
                    msg["response"] = ""
                else:
                    msg["accuracy"] = 0.0

        if messages:
            await asyncio.gather(*[_infer_one(msg) for msg in messages])

        elapsed = time.monotonic() - task.ask_time
        sla_met = elapsed <= task.sla_ttft
        for attempt in range(self._cfg.max_submit_retries + 1):
            try:
                await self._platform.submit(task_data)
                break
            except Exception as e:
                logger.warning("Submit attempt %d failed for task %d: %s", attempt, task.task_id, e)
                if attempt < self._cfg.max_submit_retries:
                    await asyncio.sleep(0.5)

        log_submit(task.task_id, sla_met, elapsed, task.profile.value if task.profile else None)
