import os
import pytest
import httpx
import respx
from openai_usage_mcp.client import OpenAIUsageClient, BASE_URL


def test_client_init_with_key():
    client = OpenAIUsageClient(api_key="sk-admin-test123")
    assert client.api_key == "sk-admin-test123"


def test_client_init_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-envkey")
    client = OpenAIUsageClient()
    assert client.api_key == "sk-admin-envkey"


def test_client_init_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_ADMIN_KEY"):
        OpenAIUsageClient()


@pytest.mark.asyncio
@respx.mock
async def test_request_single_page():
    respx.get(f"{BASE_URL}/costs").mock(
        return_value=httpx.Response(200, json={
            "object": "page",
            "data": [{"object": "bucket", "start_time": 1000, "end_time": 2000, "results": []}],
            "next_page": None,
        })
    )
    client = OpenAIUsageClient(api_key="sk-admin-test")
    result = await client.get("/costs", params={"start_time": 1000})
    assert len(result) == 1
    assert result[0]["object"] == "bucket"


@pytest.mark.asyncio
@respx.mock
async def test_request_paginates():
    respx.get(f"{BASE_URL}/costs").mock(
        side_effect=[
            httpx.Response(200, json={
                "object": "page",
                "data": [{"object": "bucket", "start_time": 1000, "end_time": 2000, "results": []}],
                "next_page": "cursor_abc",
            }),
            httpx.Response(200, json={
                "object": "page",
                "data": [{"object": "bucket", "start_time": 2000, "end_time": 3000, "results": []}],
                "next_page": None,
            }),
        ]
    )
    client = OpenAIUsageClient(api_key="sk-admin-test")
    result = await client.get("/costs", params={"start_time": 1000})
    assert len(result) == 2


@pytest.mark.asyncio
@respx.mock
async def test_request_auth_error():
    respx.get(f"{BASE_URL}/costs").mock(
        return_value=httpx.Response(403, json={"error": {"message": "Invalid admin key"}})
    )
    client = OpenAIUsageClient(api_key="sk-admin-bad")
    with pytest.raises(httpx.HTTPStatusError):
        await client.get("/costs", params={"start_time": 1000})


@pytest.mark.asyncio
@respx.mock
async def test_request_rate_limit_retry():
    respx.get(f"{BASE_URL}/costs").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "Rate limited"}}),
            httpx.Response(200, json={
                "object": "page",
                "data": [{"object": "bucket", "start_time": 1000, "end_time": 2000, "results": []}],
                "next_page": None,
            }),
        ]
    )
    client = OpenAIUsageClient(api_key="sk-admin-test")
    result = await client.get("/costs", params={"start_time": 1000})
    assert len(result) == 1
