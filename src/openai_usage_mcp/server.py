"""FastMCP server exposing OpenAI Usage and Costs tools."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastmcp import Context, FastMCP

from openai_usage_mcp.client import OpenAIUsageClient

mcp = FastMCP(
    name="openai-usage-mcp",
    instructions=(
        "OpenAI Usage and Costs MCP Server. Provides two tools:\n"
        "- costs: Query dollar-amount spend data, grouped by line item or project\n"
        "- usage: Query token/request usage data for any OpenAI service type\n\n"
        "Requires an OpenAI Admin API key (OPENAI_ADMIN_KEY env var).\n"
        "Dates should be provided in YYYY-MM-DD format."
    ),
)


def parse_date_to_unix(date_str: Optional[str]) -> Optional[int]:
    """Convert a YYYY-MM-DD date string to Unix timestamp (UTC)."""
    if date_str is None:
        return None
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def unix_to_date(ts: int) -> str:
    """Convert a Unix timestamp to YYYY-MM-DD."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


VALID_SERVICE_TYPES = [
    "completions", "embeddings", "images", "audio_speeches",
    "audio_transcriptions", "moderations", "vector_stores",
    "code_interpreter_sessions",
]


def _format_result_fields(result: dict[str, Any], service_type: str) -> list[str]:
    """Extract and format the relevant fields from a usage result."""
    parts = []

    model = result.get("model")
    if model:
        parts.append(f"model={model}")

    project = result.get("project_id")
    if project:
        parts.append(f"project={project}")

    api_key = result.get("api_key_id")
    if api_key:
        parts.append(f"api_key={api_key}")

    for key in [
        "input_tokens", "output_tokens", "input_cached_tokens",
        "input_audio_tokens", "output_audio_tokens",
        "num_model_requests", "num_images", "num_sessions",
        "characters", "seconds",
    ]:
        val = result.get(key)
        if val is not None and val > 0:
            parts.append(f"{key}={val:,}")

    return parts


def format_usage_response(buckets: list[dict[str, Any]], service_type: str) -> str:
    """Format raw usage buckets into a readable text summary."""
    if not buckets:
        return f"No usage data found for {service_type} in the specified period."

    lines = []

    for bucket in buckets:
        date = unix_to_date(bucket["start_time"])
        results = bucket.get("results", [])
        if not results:
            continue

        lines.append(f"\n## {date}")
        for r in results:
            fields = _format_result_fields(r, service_type)
            if fields:
                lines.append(f"  {', '.join(fields)}")

    header = f"# OpenAI Usage — {service_type}\n"
    return header + "\n".join(lines)


def format_costs_response(buckets: list[dict[str, Any]]) -> str:
    """Format raw cost buckets into a readable text summary."""
    if not buckets:
        return "No cost data found for the specified period."

    lines = []
    grand_total = 0.0

    for bucket in buckets:
        date = unix_to_date(bucket["start_time"])
        results = bucket.get("results", [])
        if not results:
            continue

        day_total = sum(r["amount"]["value"] for r in results)
        grand_total += day_total

        lines.append(f"\n## {date} (${day_total:.2f})")
        for r in sorted(results, key=lambda x: x["amount"]["value"], reverse=True):
            label = r.get("line_item") or r.get("project_id") or "unknown"
            amount = r["amount"]["value"]
            if amount > 0.005:
                lines.append(f"  {label}: ${amount:.2f}")

    header = f"# OpenAI Costs — Total: ${grand_total:.2f}\n"
    return header + "\n".join(lines)


@mcp.tool(
    name="costs",
    description=(
        "Query OpenAI dollar-amount spend data. Returns daily cost breakdowns "
        "grouped by line item (model/service) or project. "
        "Dates in YYYY-MM-DD format. Uses the OpenAI Costs API."
    ),
)
async def costs_tool(
    ctx: Context,
    start_time: str,
    end_time: Optional[str] = None,
    group_by: Optional[str] = None,
    limit: int = 30,
) -> str:
    """Query OpenAI costs for a date range.

    Args:
        start_time: Start date (YYYY-MM-DD)
        end_time: End date (YYYY-MM-DD), defaults to today
        group_by: JSON array of grouping fields: "project_id", "line_item". Defaults to ["line_item"]
        limit: Number of daily buckets to return (1-180, default 30)
    """
    try:
        client = OpenAIUsageClient()

        params: dict[str, Any] = {
            "start_time": parse_date_to_unix(start_time),
            "bucket_width": "1d",
            "limit": min(max(limit, 1), 180),
        }

        if end_time:
            params["end_time"] = parse_date_to_unix(end_time)

        group_by_list = json.loads(group_by) if group_by else ["line_item"]
        for g in group_by_list:
            params.setdefault("group_by[]", [])
            params["group_by[]"].append(g)

        await ctx.info(f"Querying OpenAI costs from {start_time} to {end_time or 'now'}")
        buckets = await client.get("/costs", params=params)
        return format_costs_response(buckets)
    except Exception as e:
        return f"Error querying OpenAI costs: {e}"


@mcp.tool(
    name="usage",
    description=(
        "Query OpenAI token and request usage data. Supports all service types: "
        "completions, embeddings, images, audio_speeches, audio_transcriptions, "
        "moderations, vector_stores, code_interpreter_sessions. "
        "Dates in YYYY-MM-DD format. Supports 1m/1h/1d granularity."
    ),
)
async def usage_tool(
    ctx: Context,
    service_type: str,
    start_time: str,
    end_time: Optional[str] = None,
    bucket_width: str = "1d",
    group_by: Optional[str] = None,
    models: Optional[str] = None,
    project_ids: Optional[str] = None,
    limit: int = 30,
) -> str:
    """Query OpenAI usage for a service type and date range.

    Args:
        service_type: One of: completions, embeddings, images, audio_speeches, audio_transcriptions, moderations, vector_stores, code_interpreter_sessions
        start_time: Start date (YYYY-MM-DD)
        end_time: End date (YYYY-MM-DD), defaults to today
        bucket_width: Granularity — "1m", "1h", or "1d" (default "1d")
        group_by: JSON array of grouping fields (e.g. '["model", "project_id"]')
        models: JSON array of model names to filter (e.g. '["gpt-4o"]')
        project_ids: JSON array of project IDs to filter
        limit: Number of buckets to return (default 30)
    """
    if service_type not in VALID_SERVICE_TYPES:
        return f"Invalid service_type '{service_type}'. Must be one of: {', '.join(VALID_SERVICE_TYPES)}"

    if bucket_width not in ("1m", "1h", "1d"):
        return f"Invalid bucket_width '{bucket_width}'. Must be one of: 1m, 1h, 1d"

    try:
        client = OpenAIUsageClient()

        params: dict[str, Any] = {
            "start_time": parse_date_to_unix(start_time),
            "bucket_width": bucket_width,
            "limit": limit,
        }

        if end_time:
            params["end_time"] = parse_date_to_unix(end_time)

        if group_by:
            for g in json.loads(group_by):
                params.setdefault("group_by[]", [])
                params["group_by[]"].append(g)

        if models:
            for m in json.loads(models):
                params.setdefault("models[]", [])
                params["models[]"].append(m)

        if project_ids:
            for p in json.loads(project_ids):
                params.setdefault("project_ids[]", [])
                params["project_ids[]"].append(p)

        await ctx.info(f"Querying OpenAI {service_type} usage from {start_time} to {end_time or 'now'}")
        buckets = await client.get(f"/usage/{service_type}", params=params)
        return format_usage_response(buckets, service_type)
    except Exception as e:
        return f"Error querying OpenAI {service_type} usage: {e}"


def main():
    """Entry point for the MCP server."""
    if not os.environ.get("OPENAI_ADMIN_KEY"):
        raise SystemExit(
            "OPENAI_ADMIN_KEY environment variable is required. "
            "Create an admin key at platform.openai.com/settings/organization/admin-keys"
        )
    mcp.run()


if __name__ == "__main__":
    main()
