from __future__ import annotations
import httpx
import logging
from decision_service.schemas import TaskOverview

logger = logging.getLogger(__name__)


class PlatformClient:
    def __init__(self, base_url: str, token: str, team_name: str, timeout: float = 30.0):
        self.base_url = base_url
        self.token = token
        self.team_name = team_name
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def register(self) -> dict:
        resp = await self._client.post("/register", json={
            "name": self.team_name, "token": self.token,
        })
        resp.raise_for_status()
        return resp.json()

    async def query(self) -> TaskOverview | None:
        resp = await self._client.post("/query", json={"token": self.token})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return TaskOverview(
            task_id=data["task_id"],
            target_sla=data["target_sla"],
            target_reward=data["target_reward"],
            max_winners=data["max_winners"],
            eval_task_name=data["eval_task_name"],
            eval_request_type=data["eval_request_type"],
            eval_sampling_param=data.get("eval_sampling_param", ""),
            eval_timeout_s=data["eval_timeout_s"],
        )

    async def ask(self, task_id: int, sla: str) -> dict:
        resp = await self._client.post("/ask", json={
            "token": self.token, "task_id": task_id, "sla": sla,
        })
        resp.raise_for_status()
        return resp.json()

    async def submit(self, task_data: dict) -> dict:
        resp = await self._client.post("/submit", json={
            "user": {"name": self.team_name, "token": self.token},
            "msg": task_data,
        })
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()
