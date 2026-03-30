# OpenAI Usage and Cost Management MCP Server

MCP server for accessing OpenAI platform usage and cost data through the [OpenAI Admin API](https://platform.openai.com/docs/api-reference/usage).

> **Note:** This server accesses cost and usage data from the OpenAI Admin API. All API calls are performed using the caller's admin key and are subject to OpenAI's rate limits.

## Features

### Cost Analysis
- **Spend summaries**: Total and per-line-item cost breakdowns with top-N ranking
- **Daily breakdowns**: Per-day cost tracking by model or project
- **Projected spend**: Automatic month-end projection based on current daily average
- **Anomaly detection**: Flags daily spending spikes (>2σ from mean)

### Month-over-Month Comparison
- **Cost variance analysis**: Compare any two months side by side
- **Delta tracking**: Per-line-item changes with dollar and percentage deltas
- **Biggest movers**: Highlights the largest cost increases and decreases

### Usage Tracking
- **Token consumption**: Input, output, and cached token counts by model
- **Request volumes**: API request counts over time
- **Multi-service support**: Completions, embeddings, images, audio, moderations, vector stores, and more
- **Model-level breakdown**: Usage aggregated by model with compact summary tables

## Prerequisites

1. Python 3.11 or newer
2. [uv](https://docs.astral.sh/uv/) package manager
3. An OpenAI Admin API key ([create one here](https://platform.openai.com/settings/organization/admin-keys))

## Installation

Add to your MCP client configuration (e.g., Claude Desktop, Claude Code):

### Using uv

```json
{
  "mcpServers": {
    "openai-usage-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/openai-usage-mcp", "openai-usage-mcp"],
      "env": {
        "OPENAI_ADMIN_KEY": "sk-admin-..."
      }
    }
  }
}
```

### Using uvx (from PyPI)

```json
{
  "mcpServers": {
    "openai-usage-mcp": {
      "command": "uvx",
      "args": ["openai-usage-mcp"],
      "env": {
        "OPENAI_ADMIN_KEY": "sk-admin-..."
      }
    }
  }
}
```

## Tools

### `costs`

Query OpenAI dollar-amount spend data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_time` | string | *(required)* | Start date (YYYY-MM-DD) |
| `end_time` | string | today | End date (YYYY-MM-DD) |
| `detail_level` | string | `"summary"` | `"summary"`, `"daily"`, or `"raw"` |
| `group_by` | string | `"line_item"` | `"line_item"`, `"project_id"`, or both |
| `top_n` | int | 10 | Number of top items to show |
| `limit` | int | 180 | Max daily buckets to fetch (1-180) |

**Detail levels:**
- **summary** (default): Compact total + top-N breakdown table (~20 lines). Includes projected month-end spend and anomaly detection when applicable.
- **daily**: Per-day breakdown with per-item amounts.
- **raw**: Full unprocessed data, every line item every day.

**Examples:**
```
# This month's spend
costs(start_time="2026-03-01")

# Last 7 days by project
costs(start_time="2026-03-23", group_by="project_id")

# Daily breakdown for February
costs(start_time="2026-02-01", end_time="2026-03-01", detail_level="daily")
```

### `cost-comparison`

Compare OpenAI costs between two calendar months.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseline_month` | string | *(required)* | Earlier month (YYYY-MM) |
| `comparison_month` | string | *(required)* | Later month (YYYY-MM) |
| `group_by` | string | `"line_item"` | `"line_item"`, `"project_id"`, or both |
| `top_n` | int | 10 | Number of top items to show |

**Output includes:**
- Total spend for each month with overall delta and percentage change
- Per-line-item comparison table sorted by largest absolute change
- Biggest movers section highlighting the largest increase and decrease

**Examples:**
```
# February vs March
cost-comparison(baseline_month="2026-02", comparison_month="2026-03")

# By project
cost-comparison(baseline_month="2026-02", comparison_month="2026-03", group_by="project_id")
```

### `usage`

Query OpenAI token and request usage data by service type.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `service_type` | string | *(required)* | See supported types below |
| `start_time` | string | *(required)* | Start date (YYYY-MM-DD) |
| `end_time` | string | today | End date (YYYY-MM-DD) |
| `detail_level` | string | `"summary"` | `"summary"` or `"raw"` |
| `bucket_width` | string | `"1d"` | `"1m"`, `"1h"`, or `"1d"` |
| `group_by` | string | — | `"model"`, `"project_id"`, etc. |
| `models` | string | — | Filter by model name(s) |
| `project_ids` | string | — | Filter by project ID(s) |
| `top_n` | int | 10 | Number of top models to show |
| `limit` | int | 180 | Max buckets to fetch |

**Supported service types:** `completions`, `embeddings`, `images`, `audio_speeches`, `audio_transcriptions`, `moderations`, `vector_stores`, `code_interpreter_sessions`

**Examples:**
```
# GPT-4o usage this month
usage(service_type="completions", start_time="2026-03-01", models="gpt-4o")

# All completions last week
usage(service_type="completions", start_time="2026-03-23")

# Embeddings by project
usage(service_type="embeddings", start_time="2026-03-01", group_by="project_id")
```

## Authentication

This server requires an OpenAI Admin API key set via the `OPENAI_ADMIN_KEY` environment variable.

Admin keys can be created at [platform.openai.com/settings/organization/admin-keys](https://platform.openai.com/settings/organization/admin-keys).

The key needs the **Usage** read permission to access cost and usage data.

## Development

```bash
# Clone and install
git clone https://github.com/dlaporte/openai-usage-mcp.git
cd openai-usage-mcp
uv sync --dev

# Run tests
uv run pytest -v

# Run the server locally
OPENAI_ADMIN_KEY=sk-admin-... uv run openai-usage-mcp
```

## License

MIT
