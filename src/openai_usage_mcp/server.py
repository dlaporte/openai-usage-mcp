"""FastMCP server exposing OpenAI Usage and Costs tools."""

import json
import os
from collections import defaultdict
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
        "Dates should be provided in YYYY-MM-DD format.\n"
        "Both tools default to 'summary' mode which returns compact aggregated data.\n"
        "Use detail_level='daily' or 'raw' for granular breakdowns."
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


def _parse_list_param(raw: Optional[str], default: list[str] | None = None) -> list[str]:
    """Parse a list parameter from various LLM-provided formats.

    Handles:
    - None -> default (or [])
    - JSON array: '["a", "b"]' -> ["a", "b"]
    - Bare string: "a" -> ["a"]
    - Comma-separated: "a, b" -> ["a", "b"]
    """
    if raw is None:
        return default or []
    raw = raw.strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [s.strip() for s in raw.split(",")]


VALID_SERVICE_TYPES = [
    "completions", "embeddings", "images", "audio_speeches",
    "audio_transcriptions", "moderations", "vector_stores",
    "code_interpreter_sessions",
]

USAGE_NUMERIC_FIELDS = [
    "input_tokens", "output_tokens", "input_cached_tokens",
    "input_audio_tokens", "output_audio_tokens",
    "num_model_requests", "num_images", "num_sessions",
    "characters", "seconds",
]


# ---------------------------------------------------------------------------
# Cost formatters
# ---------------------------------------------------------------------------

def format_costs_summary(buckets: list[dict[str, Any]], top_n: int = 10) -> str:
    """Aggregate cost buckets into a compact summary with top-N breakdown."""
    if not buckets:
        return "No cost data found for the specified period."

    totals: defaultdict[str, float] = defaultdict(float)
    grand_total = 0.0

    for bucket in buckets:
        for r in bucket.get("results", []):
            amount = r["amount"]["value"]
            label = r.get("line_item") or r.get("project_id") or "unknown"
            totals[label] += amount
            grand_total += amount

    if grand_total == 0:
        return "No cost data found for the specified period."

    date_start = unix_to_date(buckets[0]["start_time"])
    date_end = unix_to_date(buckets[-1]["start_time"])
    num_days = len(buckets)

    sorted_items = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_n]
    other_items = sorted_items[top_n:]

    lines = [
        f"# OpenAI Costs Summary: {date_start} to {date_end}",
        f"**Total: ${grand_total:,.2f}** ({num_days} days, ${grand_total / max(num_days, 1):,.2f}/day avg)",
        "",
        "| Line Item | Amount | % |",
        "|-----------|--------|---|",
    ]

    for label, amount in top_items:
        pct = (amount / grand_total) * 100
        lines.append(f"| {label} | ${amount:,.2f} | {pct:.1f}% |")

    if other_items:
        other_total = sum(amount for _, amount in other_items)
        other_pct = (other_total / grand_total) * 100
        lines.append(f"| Other ({len(other_items)} items) | ${other_total:,.2f} | {other_pct:.1f}% |")

    return "\n".join(lines)


def format_costs_daily(buckets: list[dict[str, Any]], top_n: int = 10) -> str:
    """Format cost buckets as daily breakdown, capped to top-N per day."""
    if not buckets:
        return "No cost data found for the specified period."

    lines = []
    grand_total = 0.0

    for bucket in buckets:
        results = bucket.get("results", [])
        if not results:
            continue

        day_total = sum(r["amount"]["value"] for r in results)
        if day_total < 0.01:
            continue

        grand_total += day_total
        date = unix_to_date(bucket["start_time"])

        sorted_results = sorted(results, key=lambda x: x["amount"]["value"], reverse=True)
        top_results = sorted_results[:top_n]
        other_results = sorted_results[top_n:]

        lines.append(f"\n## {date} (${day_total:,.2f})")
        for r in top_results:
            label = r.get("line_item") or r.get("project_id") or "unknown"
            amount = r["amount"]["value"]
            if amount > 0.005:
                lines.append(f"  {label}: ${amount:.2f}")

        if other_results:
            other_total = sum(r["amount"]["value"] for r in other_results)
            if other_total > 0.005:
                lines.append(f"  Other ({len(other_results)} items): ${other_total:.2f}")

    header = f"# OpenAI Costs — Total: ${grand_total:,.2f}\n"
    return header + "\n".join(lines)


def format_costs_response(buckets: list[dict[str, Any]]) -> str:
    """Format raw cost buckets into a readable text summary (legacy raw mode)."""
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


# ---------------------------------------------------------------------------
# Usage formatters
# ---------------------------------------------------------------------------

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

    for key in USAGE_NUMERIC_FIELDS:
        val = result.get(key)
        if val is not None and val > 0:
            parts.append(f"{key}={val:,}")

    return parts


def format_usage_summary(buckets: list[dict[str, Any]], service_type: str, top_n: int = 10) -> str:
    """Aggregate usage buckets into a compact summary by model."""
    if not buckets:
        return f"No usage data found for {service_type} in the specified period."

    # Aggregate by model
    model_totals: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    for bucket in buckets:
        for r in bucket.get("results", []):
            model = r.get("model") or "unknown"
            for key in USAGE_NUMERIC_FIELDS:
                val = r.get(key)
                if val is not None and val > 0:
                    model_totals[model][key] += val

    if not model_totals:
        return f"No usage data found for {service_type} in the specified period."

    date_start = unix_to_date(buckets[0]["start_time"])
    date_end = unix_to_date(buckets[-1]["start_time"])

    # Determine which columns have data
    all_keys: set[str] = set()
    for fields in model_totals.values():
        all_keys.update(fields.keys())

    # Sort models by total requests or input tokens
    sort_key = "num_model_requests" if "num_model_requests" in all_keys else "input_tokens"
    sorted_models = sorted(model_totals.items(), key=lambda x: x[1].get(sort_key, 0), reverse=True)

    top_models = sorted_models[:top_n]
    other_models = sorted_models[top_n:]

    # Pick display columns (most useful subset)
    display_cols = [k for k in ["input_tokens", "output_tokens", "input_cached_tokens",
                                 "num_model_requests", "num_images", "characters", "seconds"]
                    if k in all_keys]

    col_labels = {
        "input_tokens": "Input Tokens",
        "output_tokens": "Output Tokens",
        "input_cached_tokens": "Cached",
        "num_model_requests": "Requests",
        "num_images": "Images",
        "characters": "Characters",
        "seconds": "Seconds",
    }

    header_cols = " | ".join(col_labels.get(c, c) for c in display_cols)
    separator = " | ".join("---" for _ in display_cols)

    lines = [
        f"# OpenAI Usage Summary: {service_type} ({date_start} to {date_end})",
        "",
        f"| Model | {header_cols} |",
        f"|-------|{separator}|",
    ]

    for model, fields in top_models:
        vals = " | ".join(f"{fields.get(c, 0):,}" for c in display_cols)
        lines.append(f"| {model} | {vals} |")

    if other_models:
        other_totals: defaultdict[str, int] = defaultdict(int)
        for _, fields in other_models:
            for c in display_cols:
                other_totals[c] += fields.get(c, 0)
        vals = " | ".join(f"{other_totals[c]:,}" for c in display_cols)
        lines.append(f"| Other ({len(other_models)} models) | {vals} |")

    # Grand totals
    grand: defaultdict[str, int] = defaultdict(int)
    for _, fields in model_totals.items():
        for c in display_cols:
            grand[c] += fields.get(c, 0)
    total_parts = [f"{grand[c]:,} {col_labels.get(c, c).lower()}" for c in display_cols if grand[c] > 0]
    lines.append(f"\n**Totals:** {', '.join(total_parts)}")

    return "\n".join(lines)


def format_usage_response(buckets: list[dict[str, Any]], service_type: str) -> str:
    """Format raw usage buckets into a readable text summary (legacy raw mode)."""
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="costs",
    description=(
        "Query OpenAI dollar-amount spend data. "
        "Default 'summary' mode returns a compact total + top-N breakdown (~20 lines). "
        "Use detail_level='daily' for per-day breakdown, or 'raw' for full unprocessed data. "
        "For queries spanning more than a week, prefer 'summary' to avoid large responses. "
        "Dates in YYYY-MM-DD format."
    ),
)
async def costs_tool(
    ctx: Context,
    start_time: str,
    end_time: Optional[str] = None,
    detail_level: str = "summary",
    group_by: Optional[str] = None,
    top_n: int = 10,
    limit: int = 180,
) -> str:
    """Query OpenAI costs for a date range.

    Args:
        start_time: Start date (YYYY-MM-DD)
        end_time: End date (YYYY-MM-DD), defaults to today
        detail_level: "summary" (default, compact totals), "daily" (per-day), or "raw" (full detail)
        group_by: Grouping fields: "line_item", "project_id", or both. Accepts bare string, comma-separated, or JSON array.
        top_n: Number of top items to show (default 10, used in summary and daily modes)
        limit: Max daily buckets to fetch (1-180, default 180)
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

        group_by_list = _parse_list_param(group_by, ["line_item"])
        for g in group_by_list:
            params.setdefault("group_by[]", [])
            params["group_by[]"].append(g)

        await ctx.info(f"Querying OpenAI costs from {start_time} to {end_time or 'now'}")
        buckets = await client.get("/costs", params=params)

        if detail_level == "daily":
            return format_costs_daily(buckets, top_n)
        elif detail_level == "raw":
            return format_costs_response(buckets)
        else:
            return format_costs_summary(buckets, top_n)
    except Exception as e:
        return f"Error querying OpenAI costs: {e}"


@mcp.tool(
    name="usage",
    description=(
        "Query OpenAI token and request usage data. "
        "Default 'summary' mode returns a compact table aggregated by model. "
        "Use detail_level='daily' or 'raw' for granular breakdowns. "
        "Supports service types: completions, embeddings, images, audio_speeches, "
        "audio_transcriptions, moderations, vector_stores, code_interpreter_sessions. "
        "Dates in YYYY-MM-DD format."
    ),
)
async def usage_tool(
    ctx: Context,
    service_type: str,
    start_time: str,
    end_time: Optional[str] = None,
    detail_level: str = "summary",
    bucket_width: str = "1d",
    group_by: Optional[str] = None,
    models: Optional[str] = None,
    project_ids: Optional[str] = None,
    top_n: int = 10,
    limit: int = 180,
) -> str:
    """Query OpenAI usage for a service type and date range.

    Args:
        service_type: One of: completions, embeddings, images, audio_speeches, audio_transcriptions, moderations, vector_stores, code_interpreter_sessions
        start_time: Start date (YYYY-MM-DD)
        end_time: End date (YYYY-MM-DD), defaults to today
        detail_level: "summary" (default, compact totals by model), "daily" (per-day), or "raw" (full detail)
        bucket_width: Granularity — "1m", "1h", or "1d" (default "1d")
        group_by: Grouping fields (e.g. "model", "project_id"). Accepts bare string, comma-separated, or JSON array.
        models: Model names to filter (e.g. "gpt-4o" or '["gpt-4o"]')
        project_ids: Project IDs to filter
        top_n: Number of top models to show (default 10, used in summary mode)
        limit: Max buckets to fetch (default 180)
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

        for g in _parse_list_param(group_by):
            params.setdefault("group_by[]", [])
            params["group_by[]"].append(g)

        for m in _parse_list_param(models):
            params.setdefault("models[]", [])
            params["models[]"].append(m)

        for p in _parse_list_param(project_ids):
            params.setdefault("project_ids[]", [])
            params["project_ids[]"].append(p)

        await ctx.info(f"Querying OpenAI {service_type} usage from {start_time} to {end_time or 'now'}")
        buckets = await client.get(f"/usage/{service_type}", params=params)

        if detail_level == "raw":
            return format_usage_response(buckets, service_type)
        else:
            return format_usage_summary(buckets, service_type, top_n)
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
