import pytest
from openai_usage_mcp.server import (
    _parse_list_param,
    format_costs_response,
    format_costs_summary,
    format_costs_daily,
    format_usage_response,
    format_usage_summary,
    parse_date_to_unix,
)


# ---------------------------------------------------------------------------
# parse_date_to_unix
# ---------------------------------------------------------------------------

def test_parse_date_to_unix():
    ts = parse_date_to_unix("2026-03-01")
    assert ts == 1772323200


def test_parse_date_to_unix_none_returns_none():
    assert parse_date_to_unix(None) is None


# ---------------------------------------------------------------------------
# _parse_list_param
# ---------------------------------------------------------------------------

def test_parse_list_param_none_returns_default():
    assert _parse_list_param(None, ["line_item"]) == ["line_item"]


def test_parse_list_param_none_no_default():
    assert _parse_list_param(None) == []


def test_parse_list_param_bare_string():
    assert _parse_list_param("line_item") == ["line_item"]


def test_parse_list_param_comma_separated():
    assert _parse_list_param("line_item, project_id") == ["line_item", "project_id"]


def test_parse_list_param_json_array():
    assert _parse_list_param('["line_item", "project_id"]') == ["line_item", "project_id"]


def test_parse_list_param_json_single():
    assert _parse_list_param('["line_item"]') == ["line_item"]


# ---------------------------------------------------------------------------
# Cost formatters
# ---------------------------------------------------------------------------

COST_BUCKETS = [
    {
        "object": "bucket",
        "start_time": 1772323200,  # 2026-03-01
        "end_time": 1772409600,
        "results": [
            {
                "object": "organization.costs.result",
                "amount": {"value": 12.50, "currency": "usd"},
                "line_item": "GPT-4o",
                "project_id": None,
            },
            {
                "object": "organization.costs.result",
                "amount": {"value": 3.25, "currency": "usd"},
                "line_item": "GPT-4o mini",
                "project_id": None,
            },
        ],
    },
    {
        "object": "bucket",
        "start_time": 1772409600,  # 2026-03-02
        "end_time": 1772496000,
        "results": [
            {
                "object": "organization.costs.result",
                "amount": {"value": 8.00, "currency": "usd"},
                "line_item": "GPT-4o",
                "project_id": None,
            },
            {
                "object": "organization.costs.result",
                "amount": {"value": 1.50, "currency": "usd"},
                "line_item": "Embeddings",
                "project_id": None,
            },
        ],
    },
]


def test_format_costs_response_by_line_item():
    output = format_costs_response(COST_BUCKETS[:1])
    assert "2026-03-01" in output
    assert "GPT-4o" in output
    assert "$12.50" in output
    assert "$3.25" in output
    assert "$15.75" in output


def test_format_costs_response_empty():
    output = format_costs_response([])
    assert "No cost data" in output


def test_format_costs_summary_basic():
    output = format_costs_summary(COST_BUCKETS)
    assert "Summary" in output
    assert "$25.25" in output  # grand total
    assert "GPT-4o" in output
    assert "GPT-4o mini" in output
    assert "Embeddings" in output
    assert "2026-03-01" in output
    assert "2026-03-02" in output


def test_format_costs_summary_top_n():
    output = format_costs_summary(COST_BUCKETS, top_n=1)
    # GPT-4o should be #1 (20.50), rest should be "Other"
    assert "GPT-4o" in output
    assert "Other (2 items)" in output


def test_format_costs_summary_percentages():
    output = format_costs_summary(COST_BUCKETS)
    # GPT-4o = 20.50/25.25 = 81.2%
    assert "81.2%" in output


def test_format_costs_summary_empty():
    output = format_costs_summary([])
    assert "No cost data" in output


def test_format_costs_daily_basic():
    output = format_costs_daily(COST_BUCKETS)
    assert "2026-03-01" in output
    assert "2026-03-02" in output
    assert "$25.25" in output  # grand total


def test_format_costs_daily_top_n():
    output = format_costs_daily(COST_BUCKETS, top_n=1)
    assert "Other" in output


def test_format_costs_daily_skips_zero():
    zero_bucket = [{
        "object": "bucket",
        "start_time": 1772323200,
        "end_time": 1772409600,
        "results": [
            {"amount": {"value": 0.0}, "line_item": "nothing"},
        ],
    }]
    output = format_costs_daily(zero_bucket)
    assert "2026-03-01" not in output


# ---------------------------------------------------------------------------
# Usage formatters
# ---------------------------------------------------------------------------

USAGE_BUCKETS = [
    {
        "object": "bucket",
        "start_time": 1772323200,
        "end_time": 1772409600,
        "results": [
            {
                "object": "organization.usage.completions.result",
                "input_tokens": 50000,
                "output_tokens": 10000,
                "input_cached_tokens": 5000,
                "input_audio_tokens": 0,
                "output_audio_tokens": 0,
                "num_model_requests": 120,
                "model": "gpt-4o",
                "project_id": None,
                "user_id": None,
                "api_key_id": None,
                "batch": None,
            },
        ],
    },
    {
        "object": "bucket",
        "start_time": 1772409600,
        "end_time": 1772496000,
        "results": [
            {
                "object": "organization.usage.completions.result",
                "input_tokens": 30000,
                "output_tokens": 8000,
                "input_cached_tokens": 2000,
                "input_audio_tokens": 0,
                "output_audio_tokens": 0,
                "num_model_requests": 80,
                "model": "gpt-4o",
                "project_id": None,
                "user_id": None,
                "api_key_id": None,
                "batch": None,
            },
        ],
    },
]


def test_format_usage_response_completions():
    output = format_usage_response(USAGE_BUCKETS[:1], "completions")
    assert "2026-03-01" in output
    assert "50,000" in output
    assert "10,000" in output
    assert "5,000" in output
    assert "120" in output
    assert "gpt-4o" in output


def test_format_usage_response_empty():
    output = format_usage_response([], "completions")
    assert "No usage data" in output


def test_format_usage_summary_completions():
    output = format_usage_summary(USAGE_BUCKETS, "completions")
    assert "Summary" in output
    assert "gpt-4o" in output
    # Aggregated: 80,000 input tokens
    assert "80,000" in output
    # Aggregated: 18,000 output tokens
    assert "18,000" in output
    # Aggregated: 200 requests
    assert "200" in output


def test_format_usage_summary_empty():
    output = format_usage_summary([], "completions")
    assert "No usage data" in output


def test_format_usage_summary_top_n():
    # Add a second model to test top_n
    buckets = [
        {
            "object": "bucket",
            "start_time": 1772323200,
            "end_time": 1772409600,
            "results": [
                {"model": "gpt-4o", "input_tokens": 50000, "num_model_requests": 100},
                {"model": "gpt-4o-mini", "input_tokens": 30000, "num_model_requests": 80},
                {"model": "gpt-3.5", "input_tokens": 10000, "num_model_requests": 20},
            ],
        }
    ]
    output = format_usage_summary(buckets, "completions", top_n=1)
    assert "gpt-4o" in output
    assert "Other (2 models)" in output
