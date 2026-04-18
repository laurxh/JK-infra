import pytest
import httpx
import respx

from decision_service.platform_client import PlatformClient


@pytest.fixture
def platform():
    return PlatformClient(base_url="http://mock:8003", token="test_tok", team_name="test_team")


@respx.mock
@pytest.mark.asyncio
async def test_register(platform):
    respx.post("http://mock:8003/register").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    result = await platform.register()
    assert result == {"status": "ok"}


@respx.mock
@pytest.mark.asyncio
async def test_query_returns_overview(platform):
    respx.post("http://mock:8003/query").mock(
        return_value=httpx.Response(200, json={
            "task_id": 42, "target_sla": "Gold", "target_reward": 1.5,
            "max_winners": 1, "eval_task_name": "svamp",
            "eval_request_type": "generate_until",
            "eval_sampling_param": "Deterministic", "eval_timeout_s": 600,
        })
    )
    overview = await platform.query()
    assert overview is not None
    assert overview.task_id == 42
    assert overview.target_sla == "Gold"


@respx.mock
@pytest.mark.asyncio
async def test_query_returns_none_on_404(platform):
    respx.post("http://mock:8003/query").mock(
        return_value=httpx.Response(404, json={"detail": "no tasks"})
    )
    overview = await platform.query()
    assert overview is None


@respx.mock
@pytest.mark.asyncio
async def test_ask_accepted(platform):
    task_data = {"overview": {}, "messages": [{"ID": 0, "prompt": "test"}]}
    respx.post("http://mock:8003/ask").mock(
        return_value=httpx.Response(200, json={"status": "accepted", "task": task_data})
    )
    result = await platform.ask(task_id=42, sla="Gold")
    assert result["status"] == "accepted"
    assert result["task"]["messages"][0]["prompt"] == "test"


@respx.mock
@pytest.mark.asyncio
async def test_ask_rejected(platform):
    respx.post("http://mock:8003/ask").mock(
        return_value=httpx.Response(200, json={"status": "rejected", "reason": "expired"})
    )
    result = await platform.ask(task_id=42, sla="Gold")
    assert result["status"] == "rejected"


@respx.mock
@pytest.mark.asyncio
async def test_submit(platform):
    respx.post("http://mock:8003/submit").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    task_data = {"overview": {}, "messages": []}
    result = await platform.submit(task_data)
    assert result == {"status": "ok"}
