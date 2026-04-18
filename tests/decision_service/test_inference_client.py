import pytest
import httpx
import respx

from decision_service.inference_client import InferenceClient


@pytest.fixture
def client():
    return InferenceClient(base_url="http://mock:8000")


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
async def test_status(client):
    respx.get("http://mock:8000/status").mock(return_value=httpx.Response(200, json={
        "running": {"decode_tokens_remaining": 1785, "task_count": 8},
        "waiting": {"compute_tokens_remaining": 0, "task_count": 0},
    }))
    stats = await client.status()
    assert stats.running.decode_tokens_remaining == 1785
    assert stats.waiting.task_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_generate(client):
    respx.post("http://mock:8000/generate").mock(return_value=httpx.Response(200, json={
        "id": "req_1", "text": "42", "finish_reason": "stop",
        "prompt_tokens": 10, "completion_tokens": 1,
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
    respx.post("http://mock:8000/loglikelihood").mock(return_value=httpx.Response(200, json={
        "id": "req_2", "accuracy": -2.345, "prompt_tokens": 100, "continuation_tokens": 4,
    }))
    result = await client.loglikelihood(request_id="req_2", prompt="Context", continuation="choice A")
    assert result["accuracy"] == pytest.approx(-2.345)


@respx.mock
@pytest.mark.asyncio
async def test_loglikelihood_rolling(client):
    respx.post("http://mock:8000/loglikelihood_rolling").mock(return_value=httpx.Response(200, json={
        "id": "req_3", "accuracy": -1234.56, "prompt_tokens": 4096,
    }))
    result = await client.loglikelihood_rolling(request_id="req_3", prompt="A long doc...")
    assert result["accuracy"] == pytest.approx(-1234.56)


@respx.mock
@pytest.mark.asyncio
async def test_generate_503_raises(client):
    respx.post("http://mock:8000/generate").mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        await client.generate(
            request_id="req_err", prompt="test", max_tokens=10,
            temperature=0.0, top_p=1.0, stop=[],
        )
