from __future__ import annotations
import re
import httpx
import logging
from decision_service.schemas import EngineStats, RunningStats, WaitingStats

logger = logging.getLogger(__name__)


class InferenceClient:
    """HTTP client for vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str = "", timeout: float = 120.0):
        self.base_url = base_url
        self.model = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def status(self) -> EngineStats:
        """Parse vLLM's Prometheus /metrics endpoint into EngineStats."""
        resp = await self._client.get("/metrics")
        resp.raise_for_status()
        text = resp.text
        running = _parse_metric(text, "vllm:num_requests_running")
        waiting = _parse_metric(text, "vllm:num_requests_waiting")
        return EngineStats(
            running=RunningStats(decode_tokens_remaining=0, task_count=int(running)),
            waiting=WaitingStats(compute_tokens_remaining=0, task_count=int(waiting)),
        )

    async def generate(
        self, *, request_id: str, prompt: str, max_tokens: int,
        temperature: float, top_p: float, stop: list[str],
        top_k: int = 1, repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
    ) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        resp = await self._client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        return {
            "id": request_id,
            "text": choice.get("text", ""),
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
        }

    async def loglikelihood(self, *, request_id: str, prompt: str, continuation: str) -> dict:
        """Compute log P(continuation | prompt) using vLLM's logprobs.

        Approach: send prompt+continuation, generate len(continuation) tokens worth of logprobs.
        This matches the verified bidder.py approach.
        """
        if not continuation:
            return {"id": request_id, "accuracy": 0.0}

        full_text = prompt + continuation
        payload = {
            "model": self.model,
            "prompt": full_text,
            "max_tokens": len(continuation),
            "logprobs": 1,
            "temperature": 0.0,
        }
        resp = await self._client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])
        total = sum(lp for lp in token_logprobs if lp is not None)
        return {"id": request_id, "accuracy": total}

    async def loglikelihood_rolling(self, *, request_id: str, prompt: str) -> dict:
        """Compute rolling log-likelihood of entire prompt.

        Approach: send prompt, generate len(prompt) tokens worth of logprobs.
        This matches the verified bidder.py approach.
        """
        if not prompt:
            return {"id": request_id, "accuracy": 0.0}

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": len(prompt),
            "logprobs": 1,
            "temperature": 0.0,
        }
        resp = await self._client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])
        total = sum(lp for lp in token_logprobs if lp is not None)
        return {"id": request_id, "accuracy": total}

    async def close(self):
        await self._client.aclose()


def _parse_metric(text: str, metric_name: str) -> float:
    """Extract a gauge value from Prometheus text format."""
    pattern = rf'^{re.escape(metric_name)}\s+([\d.eE+\-]+)'
    for line in text.split('\n'):
        m = re.match(pattern, line)
        if m:
            return float(m.group(1))
    return 0.0
