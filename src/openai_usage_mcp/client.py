"""HTTP client for the OpenAI Usage and Costs APIs."""

import os
from typing import Any, Optional

import httpx

BASE_URL = "https://api.openai.com/v1/organization"


class OpenAIUsageClient:
    """Async client for OpenAI's Usage and Costs API endpoints."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_ADMIN_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENAI_ADMIN_KEY environment variable is required. "
                "Create an admin key at platform.openai.com/settings/organization/admin-keys"
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
