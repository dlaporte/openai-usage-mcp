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


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
