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
        resp = await self._client.get("/status")
        resp.raise_for_status()
        data = resp.json()
        return EngineStats(
            running=RunningStats(**data["running"]),
            waiting=WaitingStats(**data["waiting"]),
        )

    async def generate(
        self, *, request_id: str, prompt: str, max_tokens: int,
        temperature: float, top_p: float, stop: list[str],
        top_k: int = 1, repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
    ) -> dict:
        payload = {
            "request_id": request_id, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty, "stop": stop,
        }
        resp = await self._client.post("/generate", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def loglikelihood(self, *, request_id: str, prompt: str, continuation: str) -> dict:
        resp = await self._client.post("/loglikelihood", json={
            "request_id": request_id, "prompt": prompt, "continuation": continuation,
        })
        resp.raise_for_status()
        return resp.json()

    async def loglikelihood_rolling(self, *, request_id: str, prompt: str) -> dict:
        resp = await self._client.post("/loglikelihood_rolling", json={
            "request_id": request_id, "prompt": prompt,
        })
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()
