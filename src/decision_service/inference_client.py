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
        return {
            "id": request_id,
            "text": choice.get("text", ""),
        }

    async def loglikelihood(self, *, request_id: str, prompt: str, continuation: str) -> dict:
        """Compute log P(continuation | prompt) using vLLM's logprobs."""
        full_text = prompt + continuation
        payload = {
            "model": self.model,
            "prompt": full_text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 0.0,
        }
        resp = await self._client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])

        # We need to sum only the continuation portion.
        # vLLM returns logprobs for the entire prompt+continuation.
        # The continuation tokens start somewhere after the prompt tokens.
        # Since we don't have exact token boundaries, use offset mapping.
        text_offset = logprobs_data.get("text_offset", [])
        prompt_len = len(prompt)

        if text_offset:
            # Find first token that starts at or after prompt boundary
            cont_start = 0
            for i, offset in enumerate(text_offset):
                if offset >= prompt_len:
                    cont_start = i
                    break
            cont_logprobs = token_logprobs[cont_start:]
        else:
            # Fallback: use all logprobs (less accurate but won't crash)
            cont_logprobs = token_logprobs

        total = sum(lp for lp in cont_logprobs if lp is not None)
        return {"id": request_id, "accuracy": total}

    async def loglikelihood_rolling(self, *, request_id: str, prompt: str) -> dict:
        """Compute rolling log-likelihood of entire prompt."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 0,
            "echo": True,
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
