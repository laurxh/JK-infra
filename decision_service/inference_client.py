from __future__ import annotations
import httpx
import logging
from decision_service.schemas import EngineStats, RunningStats, WaitingStats

logger = logging.getLogger(__name__)


class InferenceClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def status(self) -> EngineStats:
        resp = await self._client.get("/stats")
        resp.raise_for_status()
        data = resp.json()
        qs = data.get("queue_stats", data)
        return EngineStats(
            running=RunningStats(**qs["running"]),
            waiting=WaitingStats(**qs["waiting"]),
        )

    async def generate(
        self, *, request_id: str, prompt: str, max_tokens: int,
        temperature: float, top_p: float, stop: list[str],
        top_k: int = 1, repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
    ) -> dict:
        payload = {
            "ID": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop,
        }
        resp = await self._client.post("/generate", json=payload)
        resp.raise_for_status()
        result = resp.json()
        # Normalize: teammate returns {ID, text}, we expose {id, text}
        result["id"] = result.pop("ID", request_id)
        return result

    async def loglikelihood(self, *, request_id: str, prompt: str, continuation: str) -> dict:
        resp = await self._client.post("/loglikelihood", json={
            "ID": request_id,
            "prompt": prompt,
            "eval_continuation": continuation,
            "eval_request_type": "loglikelihood",
        })
        resp.raise_for_status()
        result = resp.json()
        result["id"] = result.pop("ID", request_id)
        return result

    async def loglikelihood_rolling(self, *, request_id: str, prompt: str) -> dict:
        resp = await self._client.post("/loglikelihood_rolling", json={
            "ID": request_id,
            "prompt": prompt,
            "eval_request_type": "loglikelihood_rolling",
        })
        resp.raise_for_status()
        result = resp.json()
        result["id"] = result.pop("ID", request_id)
        return result

    async def close(self):
        await self._client.aclose()
