import pytest
import httpx
import respx

from decision_service.inference_client import InferenceClient, _parse_metric


@pytest.fixture
def client():
    return InferenceClient(base_url="http://mock:8000", model="Qwen3-32B")


@respx.mock
@pytest.mark.asyncio
async def test_health_ok(client):
    respx.get("http://mock:8000/health").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    assert await client.health() is True


@respx.mock
@pytest.mark.asyncio
async def test_health_not_ready(client):
    respx.get("http://mock:8000/health").mock(return_value=httpx.Response(503))
    assert await client.health() is False


@respx.mock
@pytest.mark.asyncio
async def test_status_from_metrics(client):
    metrics_text = (
        "# HELP vllm:num_requests_running Number of requests running\n"
        "# TYPE vllm:num_requests_running gauge\n"
        "vllm:num_requests_running 5\n"
        "# HELP vllm:num_requests_waiting Number of requests waiting\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        "vllm:num_requests_waiting 3\n"
    )
    respx.get("http://mock:8000/metrics").mock(
        return_value=httpx.Response(200, text=metrics_text)
    )
    stats = await client.status()
    assert stats.running.task_count == 5
    assert stats.waiting.task_count == 3


@respx.mock
@pytest.mark.asyncio
async def test_generate(client):
    respx.post("http://mock:8000/v1/completions").mock(return_value=httpx.Response(200, json={
        "choices": [{"text": "42", "finish_reason": "stop"}],
    }))
    result = await client.generate(
        request_id="req_1", prompt="What is 6*7?",
        max_tokens=256, temperature=0.0, top_p=1.0, stop=["\n\n"],
    )
    assert result["id"] == "req_1"
    assert result["text"] == "42"


@respx.mock
@pytest.mark.asyncio
async def test_loglikelihood(client):
    # bidder.py approach: send prompt+continuation, generate len(continuation) tokens
    respx.post("http://mock:8000/v1/completions").mock(return_value=httpx.Response(200, json={
        "choices": [{
            "logprobs": {
                "token_logprobs": [-1.0, -0.3, -0.7],
            },
        }],
    }))
    result = await client.loglikelihood(
        request_id="req_2", prompt="Context", continuation="choice A",
    )
    assert result["accuracy"] == pytest.approx(-2.0)
    assert result["id"] == "req_2"


@respx.mock
@pytest.mark.asyncio
async def test_loglikelihood_rolling(client):
    respx.post("http://mock:8000/v1/completions").mock(return_value=httpx.Response(200, json={
        "choices": [{
            "logprobs": {
                "token_logprobs": [None, -2.0, -3.0, -4.0],
            },
        }],
    }))
    result = await client.loglikelihood_rolling(
        request_id="req_3", prompt="A long doc...",
    )
    # sum(-2.0, -3.0, -4.0) = -9.0 (None skipped)
    assert result["accuracy"] == pytest.approx(-9.0)
    assert result["id"] == "req_3"


@respx.mock
@pytest.mark.asyncio
async def test_generate_503_raises(client):
    respx.post("http://mock:8000/v1/completions").mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        await client.generate(
            request_id="req_err", prompt="test", max_tokens=10,
            temperature=0.0, top_p=1.0, stop=[],
        )


def test_parse_metric():
    text = (
        "# HELP vllm:num_requests_running help\n"
        "# TYPE vllm:num_requests_running gauge\n"
        "vllm:num_requests_running 12\n"
        "vllm:num_requests_waiting 0\n"
    )
    assert _parse_metric(text, "vllm:num_requests_running") == 12.0
    assert _parse_metric(text, "vllm:num_requests_waiting") == 0.0
    assert _parse_metric(text, "nonexistent") == 0.0
