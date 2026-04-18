import asyncio
import pytest
from decision_service.query_harvester import QueryHarvester
from decision_service.schemas import TaskOverview


class FakePlatform:
    def __init__(self, overviews: list[TaskOverview | None]):
        self._overviews = list(overviews)
        self._idx = 0

    async def query(self) -> TaskOverview | None:
        if self._idx >= len(self._overviews):
            return None
        o = self._overviews[self._idx]
        self._idx += 1
        return o


@pytest.mark.asyncio
async def test_harvester_dedup(gen_overview):
    platform = FakePlatform([gen_overview, gen_overview, gen_overview])
    queue: asyncio.Queue[TaskOverview] = asyncio.Queue()
    harvester = QueryHarvester(platform, queue, concurrency=1, backoff_s=0.01)
    task = asyncio.create_task(harvester.run())
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_harvester_passes_none(gen_overview):
    platform = FakePlatform([None, gen_overview])
    queue: asyncio.Queue[TaskOverview] = asyncio.Queue()
    harvester = QueryHarvester(platform, queue, concurrency=1, backoff_s=0.01)
    task = asyncio.create_task(harvester.run())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert queue.qsize() == 1
