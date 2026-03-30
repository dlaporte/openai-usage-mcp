import os
import pytest
from openai_usage_mcp.client import OpenAIUsageClient


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
