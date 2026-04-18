from __future__ import annotations
import asyncio
import logging
from decision_service.schemas import TaskOverview

logger = logging.getLogger(__name__)


class QueryHarvester:
    def __init__(self, platform, overview_queue: asyncio.Queue, concurrency: int = 3, backoff_s: float = 0.5):
        self._platform = platform
        self._queue = overview_queue
        self._concurrency = concurrency
        self._backoff_s = backoff_s
        self._seen: set[int] = set()

    async def run(self):
        workers = [asyncio.create_task(self._worker(i)) for i in range(self._concurrency)]
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            for w in workers:
                w.cancel()
            raise

    async def _worker(self, worker_id: int):
        while True:
            try:
                overview = await self._platform.query()
                if overview is None:
                    await asyncio.sleep(self._backoff_s)
                    continue
                if overview.task_id in self._seen:
                    continue
                self._seen.add(overview.task_id)
                await self._queue.put(overview)
                logger.debug("Harvester[%d] queued task %d", worker_id, overview.task_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Harvester[%d] error: %s", worker_id, e)
                await asyncio.sleep(self._backoff_s)
