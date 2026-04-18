from __future__ import annotations
import asyncio
import logging
from decision_service.config import load_config
from decision_service.platform_client import PlatformClient
from decision_service.inference_client import InferenceClient
from decision_service.engine_monitor import EngineStatsPoller, ThroughputMeter
from decision_service.inflight_registry import InflightRegistry
from decision_service.history_store import HistoryStore
from decision_service.query_harvester import QueryHarvester
from decision_service.admission import AdmissionController
from decision_service.execution import ExecutionScheduler
from decision_service.logging_utils import setup_logging
from decision_service.schemas import TaskOverview, InflightTask

logger = logging.getLogger(__name__)


async def wait_for_engine(inference: InferenceClient, timeout_s: float, poll_s: float):
    logger.info("Waiting for inference engine to be ready (timeout=%.0fs)...", timeout_s)
    elapsed = 0.0
    while elapsed < timeout_s:
        if await inference.health():
            logger.info("Inference engine is ready.")
            return
        await asyncio.sleep(poll_s)
        elapsed += poll_s
    raise RuntimeError(f"Inference engine not ready after {timeout_s}s")


async def main():
    setup_logging()
    cfg = load_config()
    logger.info("Config loaded: platform=%s inference=%s team=%s", cfg.platform_url, cfg.inference_url, cfg.team_name)

    platform = PlatformClient(base_url=cfg.platform_url, token=cfg.token, team_name=cfg.team_name)
    inference = InferenceClient(base_url=cfg.inference_url, model=cfg.model_path or cfg.model_name)

    await wait_for_engine(inference, cfg.health_timeout_s, cfg.health_poll_interval_s)

    await platform.register()
    logger.info("Registered with platform as %s", cfg.team_name)

    overview_queue: asyncio.Queue[TaskOverview] = asyncio.Queue(maxsize=100)
    exec_queue: asyncio.Queue[InflightTask] = asyncio.Queue(maxsize=50)
    inflight = InflightRegistry()
    history = HistoryStore()
    throughput = ThroughputMeter()
    stats_poller = EngineStatsPoller(inference, poll_interval_s=cfg.stats_poll_interval_s)
    harvester = QueryHarvester(
        platform, overview_queue, concurrency=cfg.query_concurrency, backoff_s=cfg.query_backoff_s,
    )
    admission = AdmissionController(
        overview_queue=overview_queue, exec_queue=exec_queue,
        platform=platform,
        stats_poller=stats_poller, throughput_meter=throughput,
        inflight=inflight, config=cfg,
    )
    execution = ExecutionScheduler(
        exec_queue=exec_queue, inference=inference, platform=platform,
        inflight=inflight, config=cfg, throughput_meter=throughput, history=history,
    )

    logger.info("Starting decision service...")
    try:
        await asyncio.gather(
            stats_poller.run(),
            harvester.run(),
            admission.run(),
            execution.run(),
        )
    except asyncio.CancelledError:
        logger.info("Shutting down...")
    finally:
        await platform.close()
        await inference.close()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
