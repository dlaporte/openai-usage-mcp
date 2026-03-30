"""HTTP client for the OpenAI Usage and Costs APIs."""

import os
from typing import Any, Optional

import httpx

BASE_URL = "https://api.openai.com/v1/organization"
MAX_PAGES = 50
REQUEST_TIMEOUT = 30.0


class OpenAIUsageClient:
    """Async client for OpenAI's Usage and Costs API endpoints."""

    _project_cache: Optional[dict[str, str]] = None

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
        Pagination is capped at MAX_PAGES to prevent runaway loops.
        """
        params = dict(params or {})
        all_data: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as http:
            for _ in range(MAX_PAGES):
                response = await self._request(http, path, params)
                body = response.json()
                all_data.extend(body.get("data", []))

                next_page = body.get("next_page")
                if not next_page:
                    break
                params["page"] = next_page

        return all_data

    async def list_projects(self) -> dict[str, str]:
        """Fetch all projects and return a mapping of project_id -> project_name.

        Results are cached for the lifetime of the process to avoid repeated API calls.
        """
        if OpenAIUsageClient._project_cache is not None:
            return OpenAIUsageClient._project_cache

        projects: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as http:
            params: dict[str, Any] = {"limit": 100, "include_archived": True}
            for _ in range(MAX_PAGES):
                response = await self._request(http, "/projects", params)
                body = response.json()
                for proj in body.get("data", []):
                    projects[proj["id"]] = proj["name"]
                if not body.get("has_more"):
                    break
                last_id = body.get("last_id")
                if not last_id:
                    break
                params["after"] = last_id

        OpenAIUsageClient._project_cache = projects
        return projects

    async def _request(self, http: httpx.AsyncClient, path: str, params: dict[str, Any]) -> httpx.Response:
        """Execute a single request with one retry on 429."""
        import asyncio

        url = f"{BASE_URL}{path}"
        response = await http.get(url, headers=self._headers(), params=params)

        if response.status_code == 429:
            await asyncio.sleep(1)
            response = await http.get(url, headers=self._headers(), params=params)

        if response.status_code >= 400:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            raise RuntimeError(f"OpenAI API error {response.status_code}: {detail}")

        return response
