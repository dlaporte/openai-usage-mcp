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

    async def get(self, path: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Make a GET request, handling pagination automatically.

        Returns a flat list of all data items across all pages.
        """
        params = dict(params or {})
        all_data: list[dict[str, Any]] = []

        async with httpx.AsyncClient() as http:
            while True:
                response = await self._request(http, path, params)
                body = response.json()
                all_data.extend(body.get("data", []))

                next_page = body.get("next_page")
                if not next_page:
                    break
                params["page"] = next_page

        return all_data

    async def _request(self, http: httpx.AsyncClient, path: str, params: dict[str, Any]) -> httpx.Response:
        """Execute a single request with one retry on 429."""
        import asyncio

        url = f"{BASE_URL}{path}"
        response = await http.get(url, headers=self._headers(), params=params)

        if response.status_code == 429:
            await asyncio.sleep(1)
            response = await http.get(url, headers=self._headers(), params=params)

        response.raise_for_status()
        return response
