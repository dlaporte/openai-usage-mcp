"""FastMCP server exposing OpenAI Usage and Costs tools."""

import calendar
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from fastmcp import Context, FastMCP

from openai_usage_mcp.client import OpenAIUsageClient

# ---------------------------------------------------------------------------
# Tool descriptions (AWS-style: USE FOR / DON'T USE FOR / DEFAULTS / EXAMPLES)
# ---------------------------------------------------------------------------

COSTS_DESCRIPTION = (
    "Query OpenAI dollar-amount spend data.\n\n"
    "USE THIS TOOL FOR:\n"
    "- Total spend over a date range (summary mode, default)\n"
    "- Daily cost breakdown by line item or project\n"
    "- Month-to-date spend with projected month-end forecast\n"
    "- Identifying cost anomalies (spike days highlighted in summary)\n\n"
    "DO NOT USE THIS TOOL FOR:\n"
    "- Token or request counts (use 'usage' tool instead)\n"
    "- Comparing two different months side by side (use 'cost-comparison' tool)\n\n"
    "DETAIL LEVELS:\n"
    "- summary (default): Compact total + top-N breakdown table (~20 lines). "
    "Includes projected month-end spend and anomaly detection when applicable.\n"
    "- daily: Per-day breakdown with per-item amounts.\n"
    "- raw: Full unprocessed data, every line item every day.\n\n"
    "DEFAULTS: detail_level='summary', group_by='line_item', top_n=10, end_time=today\n\n"
    "EXAMPLES:\n"
    '- This month\'s spend: start_time="2026-03-01"\n'
    '- Last 7 days by project: start_time="2026-03-23", group_by="project_id"\n'
    '- Daily breakdown for February: start_time="2026-02-01", end_time="2026-03-01", detail_level="daily"\n\n'
    "Dates in YYYY-MM-DD format."
)

USAGE_DESCRIPTION = (
    "Query OpenAI token and request usage data by service type.\n\n"
    "USE THIS TOOL FOR:\n"
    "- Token consumption (input, output, cached) by model\n"
    "- Request counts over time\n"
    "- Usage breakdown by model, project, or API key\n"
    "- Image generation counts, audio seconds, etc.\n\n"
    "DO NOT USE THIS TOOL FOR:\n"
    "- Dollar-amount costs (use 'costs' tool instead)\n"
    "- Month-over-month cost comparisons (use 'cost-comparison' tool)\n\n"
    "SERVICE TYPES: completions, embeddings, images, audio_speeches, "
    "audio_transcriptions, moderations, vector_stores, code_interpreter_sessions\n\n"
    "DETAIL LEVELS:\n"
    "- summary (default): Compact table aggregated by model with totals.\n"
    "- daily: Per-day breakdown.\n"
    "- raw: Full unprocessed data.\n\n"
    "DEFAULTS: detail_level='summary', bucket_width='1d', top_n=10, end_time=today\n\n"
    "EXAMPLES:\n"
    '- GPT-4o usage this month: service_type="completions", start_time="2026-03-01", models="gpt-4o"\n'
    '- All completions last week: service_type="completions", start_time="2026-03-23"\n'
    '- Embeddings by project: service_type="embeddings", start_time="2026-03-01", group_by="project_id"\n\n'
    "Dates in YYYY-MM-DD format."
)

COST_COMPARISON_DESCRIPTION = (
    "Compare OpenAI costs between two months to identify spending changes.\n\n"
    "USE THIS TOOL FOR:\n"
    "- Month-over-month cost variance analysis (e.g., February vs March)\n"
    "- Identifying which line items increased or decreased the most\n"
    "- Executive-level cost change summaries\n\n"
    "DO NOT USE THIS TOOL FOR:\n"
    "- Single-month cost analysis (use 'costs' tool instead)\n"
    "- Token or request usage data (use 'usage' tool instead)\n"
    "- Arbitrary date range comparisons (this tool compares full calendar months only)\n\n"
    "PARAMETERS:\n"
    '- baseline_month: Earlier month in YYYY-MM format (e.g., "2026-02")\n'
    '- comparison_month: Later month in YYYY-MM format (e.g., "2026-03")\n'
    '- group_by: "line_item" (default), "project_id", or both\n'
    "- top_n: Number of items to show (default 10)\n\n"
    "OUTPUT includes:\n"
    "- Total spend for each month with overall delta and % change\n"
    "- Per-line-item comparison table with delta and % change\n"
    "- Biggest movers section highlighting largest increase and decrease\n\n"
    "EXAMPLES:\n"
    '- February vs March: baseline_month="2026-02", comparison_month="2026-03"\n'
    '- By project: baseline_month="2026-02", comparison_month="2026-03", group_by="project_id"'
)

mcp = FastMCP(
    name="openai-usage-mcp",
    instructions=(
        "OpenAI Usage and Costs MCP Server. Provides three tools:\n"
        "- costs: Query dollar-amount spend data with summary, daily, or raw detail levels\n"
        "- usage: Query token/request usage data for any OpenAI service type\n"
        "- cost-comparison: Compare costs between two months\n\n"
        "Requires an OpenAI Admin API key (OPENAI_ADMIN_KEY env var).\n"
        "Dates should be provided in YYYY-MM-DD format (or YYYY-MM for cost-comparison).\n"
        "Both costs and usage tools default to 'summary' mode which returns compact aggregated data.\n"
        "Use detail_level='daily' or 'raw' for granular breakdowns."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_DETAIL_LEVELS = ("summary", "daily", "raw")


def parse_date_to_unix(date_str: Optional[str]) -> Optional[int]:
    """Convert a YYYY-MM-DD date string to Unix timestamp (UTC)."""
    if date_str is None:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(
            f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD (e.g., '2026-03-01')"
        )
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
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
            raise ValueError(f"Expected JSON array of strings, got: {raw}")
        return parsed
    return [s.strip() for s in raw.split(",") if s.strip()]


def _month_to_range(month_str: str) -> tuple[int, int]:
    """Convert 'YYYY-MM' to (start_unix, end_unix) spanning that full month."""
    try:
        dt = datetime.strptime(month_str, "%Y-%m").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(
            f"Invalid month '{month_str}'. Expected format: YYYY-MM (e.g., '2026-03')"
        )
    year, month = dt.year, dt.month
    start = int(dt.timestamp())
    # Roll to first of next month
    if month == 12:
        end_dt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end_dt = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    end = int(end_dt.timestamp())
    return start, end


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
# Forecast and anomaly helpers
# ---------------------------------------------------------------------------

def _compute_forecast(buckets: list[dict[str, Any]], grand_total: float) -> str | None:
    """Compute projected month-end spend from partial-month data.

    Returns a forecast line if the data starts on the 1st of a month and
    covers fewer days than the full month. Returns None otherwise.
    """
    if not buckets or grand_total <= 0:
        return None

    first_dt = datetime.fromtimestamp(buckets[0]["start_time"], tz=timezone.utc)
    if first_dt.day != 1:
        return None

    days_elapsed = len(buckets)
    total_days = calendar.monthrange(first_dt.year, first_dt.month)[1]

    if days_elapsed >= total_days:
        return None

    daily_avg = grand_total / days_elapsed
    projected = daily_avg * total_days
    remaining = total_days - days_elapsed

    return (
        f"\n**Projected month-end: ${projected:,.2f}** "
        f"(based on ${daily_avg:,.2f}/day avg, {remaining} days remaining)"
    )


def _detect_anomalies(
    buckets: list[dict[str, Any]], threshold_sigma: float = 2.0,
) -> str | None:
    """Detect daily spending anomalies using standard deviation.

    Returns a compact table of spike days, or None if no anomalies found.
    Requires at least 7 days of data and a mean > $1/day to avoid noise.
    """
    if len(buckets) < 7:
        return None

    daily: list[tuple[str, float]] = []
    for b in buckets:
        date = unix_to_date(b["start_time"])
        total = sum(r["amount"]["value"] for r in b.get("results", []))
        daily.append((date, total))

    amounts = [a for _, a in daily]
    mean = statistics.mean(amounts)

    if mean < 1.0:
        return None

    try:
        stdev = statistics.stdev(amounts)
    except statistics.StatisticsError:
        return None

    if stdev == 0:
        return None

    threshold = mean + threshold_sigma * stdev
    anomalies = [(date, amt) for date, amt in daily if amt > threshold]

    if not anomalies:
        return None

    # Sort by deviation magnitude, cap at 5
    anomalies.sort(key=lambda x: x[1], reverse=True)
    anomalies = anomalies[:5]

    lines = [
        f"\n### Anomalies (>{threshold_sigma}σ from mean)",
        "| Date | Amount | vs Avg |",
        "|------|--------|--------|",
    ]
    for date, amt in anomalies:
        pct_above = ((amt - mean) / mean) * 100
        lines.append(f"| {date} | ${amt:,.2f} | +{pct_above:.0f}% |")

    return "\n".join(lines)


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

    # Append forecast if applicable
    forecast = _compute_forecast(buckets, grand_total)
    if forecast:
        lines.append(forecast)

    # Append anomaly detection if applicable
    anomalies = _detect_anomalies(buckets)
    if anomalies:
        lines.append(anomalies)

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
# Cost comparison formatter
# ---------------------------------------------------------------------------

def format_cost_comparison(
    baseline_buckets: list[dict[str, Any]],
    comparison_buckets: list[dict[str, Any]],
    baseline_label: str,
    comparison_label: str,
    top_n: int = 10,
) -> str:
    """Compare costs between two periods and format as a delta table."""
    if not baseline_buckets and not comparison_buckets:
        return "No cost data found for either period."

    def _aggregate(buckets: list[dict[str, Any]]) -> tuple[defaultdict[str, float], float]:
        totals: defaultdict[str, float] = defaultdict(float)
        grand = 0.0
        for bucket in buckets:
            for r in bucket.get("results", []):
                amount = r["amount"]["value"]
                label = r.get("line_item") or r.get("project_id") or "unknown"
                totals[label] += amount
                grand += amount
        return totals, grand

    base_totals, base_grand = _aggregate(baseline_buckets)
    comp_totals, comp_grand = _aggregate(comparison_buckets)

    overall_delta = comp_grand - base_grand
    overall_pct = ((overall_delta / base_grand) * 100) if base_grand > 0 else 0.0
    delta_sign = "+" if overall_delta >= 0 else ""

    # Build per-item deltas
    all_labels = set(base_totals.keys()) | set(comp_totals.keys())
    items: list[tuple[str, float, float, float, str]] = []
    for label in all_labels:
        b_amt = base_totals.get(label, 0.0)
        c_amt = comp_totals.get(label, 0.0)
        delta = c_amt - b_amt
        if b_amt > 0:
            pct_str = f"{(delta / b_amt) * 100:+.1f}%"
        elif c_amt > 0:
            pct_str = "new"
        else:
            pct_str = "—"
        items.append((label, b_amt, c_amt, delta, pct_str))

    # Sort by absolute delta descending
    items.sort(key=lambda x: abs(x[3]), reverse=True)

    top_items = items[:top_n]
    other_items = items[top_n:]

    lines = [
        f"# Cost Comparison: {baseline_label} vs {comparison_label}",
        f"**{baseline_label}: ${base_grand:,.2f}** | **{comparison_label}: ${comp_grand:,.2f}** | "
        f"**Delta: {delta_sign}${abs(overall_delta):,.2f} ({delta_sign}{abs(overall_pct):.1f}%)**",
        "",
        f"| Line Item | {baseline_label} | {comparison_label} | Delta | % Change |",
        "|-----------|----------|------------|-------|----------|",
    ]

    def _fmt_delta(delta: float) -> str:
        sign = "+" if delta >= 0 else "-"
        return f"{sign}${abs(delta):,.2f}"

    for label, b_amt, c_amt, delta, pct_str in top_items:
        lines.append(
            f"| {label} | ${b_amt:,.2f} | ${c_amt:,.2f} | {_fmt_delta(delta)} | {pct_str} |"
        )

    if other_items:
        o_base = sum(x[1] for x in other_items)
        o_comp = sum(x[2] for x in other_items)
        o_delta = o_comp - o_base
        o_pct = f"{(o_delta / o_base) * 100:+.1f}%" if o_base > 0 else ("new" if o_comp > 0 else "—")
        lines.append(
            f"| Other ({len(other_items)} items) | ${o_base:,.2f} | ${o_comp:,.2f} | {_fmt_delta(o_delta)} | {o_pct} |"
        )

    # Biggest movers
    increases = [(l, d) for l, _, _, d, _ in items if d > 0]
    decreases = [(l, d) for l, _, _, d, _ in items if d < 0]

    if increases or decreases:
        lines.append("\n### Biggest Movers")
        if increases:
            top_inc = increases[0]
            lines.append(f"**Largest increase:** {top_inc[0]} (+${top_inc[1]:,.2f})")
        if decreases:
            top_dec = decreases[-1]  # most negative
            lines.append(f"**Largest decrease:** {top_dec[0]} (-${abs(top_dec[1]):,.2f})")

    return "\n".join(lines)


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

@mcp.tool(name="costs", description=COSTS_DESCRIPTION)
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
    if detail_level not in VALID_DETAIL_LEVELS:
        return f"Invalid detail_level '{detail_level}'. Must be one of: {', '.join(VALID_DETAIL_LEVELS)}"

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


@mcp.tool(name="usage", description=USAGE_DESCRIPTION)
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

    if detail_level not in VALID_DETAIL_LEVELS:
        return f"Invalid detail_level '{detail_level}'. Must be one of: {', '.join(VALID_DETAIL_LEVELS)}"

    if bucket_width not in ("1m", "1h", "1d"):
        return f"Invalid bucket_width '{bucket_width}'. Must be one of: 1m, 1h, 1d"

    try:
        client = OpenAIUsageClient()

        params: dict[str, Any] = {
            "start_time": parse_date_to_unix(start_time),
            "bucket_width": bucket_width,
            "limit": min(max(limit, 1), 180),
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


@mcp.tool(name="cost-comparison", description=COST_COMPARISON_DESCRIPTION)
async def cost_comparison_tool(
    ctx: Context,
    baseline_month: str,
    comparison_month: str,
    group_by: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """Compare OpenAI costs between two months.

    Args:
        baseline_month: Earlier month (YYYY-MM, e.g. "2026-02")
        comparison_month: Later month (YYYY-MM, e.g. "2026-03")
        group_by: Grouping fields: "line_item" (default), "project_id", or both
        top_n: Number of top items to show (default 10)
    """
    try:
        client = OpenAIUsageClient()

        base_start, base_end = _month_to_range(baseline_month)
        comp_start, comp_end = _month_to_range(comparison_month)

        group_by_list = _parse_list_param(group_by, ["line_item"])

        def _build_params(start: int, end: int) -> dict[str, Any]:
            p: dict[str, Any] = {
                "start_time": start,
                "end_time": end,
                "bucket_width": "1d",
                "limit": 31,
            }
            for g in group_by_list:
                p.setdefault("group_by[]", [])
                p["group_by[]"].append(g)
            return p

        await ctx.info(f"Comparing costs: {baseline_month} vs {comparison_month}")
        baseline_buckets = await client.get("/costs", params=_build_params(base_start, base_end))
        comparison_buckets = await client.get("/costs", params=_build_params(comp_start, comp_end))

        return format_cost_comparison(
            baseline_buckets, comparison_buckets,
            baseline_month, comparison_month, top_n,
        )
    except Exception as e:
        return f"Error comparing OpenAI costs: {e}"


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
