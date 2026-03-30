import pytest
from openai_usage_mcp.server import (
    _compute_forecast,
    _detect_anomalies,
    _month_to_range,
    _parse_list_param,
    format_cost_comparison,
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
# _month_to_range
# ---------------------------------------------------------------------------

def test_month_to_range_basic():
    start, end = _month_to_range("2026-03")
    assert start == parse_date_to_unix("2026-03-01")
    assert end == parse_date_to_unix("2026-04-01")


def test_month_to_range_december():
    start, end = _month_to_range("2026-12")
    assert start == parse_date_to_unix("2026-12-01")
    assert end == parse_date_to_unix("2027-01-01")


# ---------------------------------------------------------------------------
# Cost fixtures
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


def _make_daily_buckets(amounts: list[float], start_ts: int = 1772323200) -> list[dict]:
    """Helper: create cost buckets with one line item per day."""
    buckets = []
    for i, amt in enumerate(amounts):
        buckets.append({
            "object": "bucket",
            "start_time": start_ts + i * 86400,
            "end_time": start_ts + (i + 1) * 86400,
            "results": [
                {"amount": {"value": amt}, "line_item": "GPT-4o"},
            ],
        })
    return buckets


# ---------------------------------------------------------------------------
# Cost formatters
# ---------------------------------------------------------------------------

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
    assert "$25.25" in output


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
# Forecast
# ---------------------------------------------------------------------------

def test_compute_forecast_partial_month():
    # 15 days of $10/day starting March 1
    buckets = _make_daily_buckets([10.0] * 15)
    result = _compute_forecast(buckets, 150.0)
    assert result is not None
    assert "Projected" in result
    # 31 days * $10/day = $310
    assert "$310.00" in result
    assert "16 days remaining" in result


def test_compute_forecast_full_month_returns_none():
    # 31 days = full March
    buckets = _make_daily_buckets([10.0] * 31)
    result = _compute_forecast(buckets, 310.0)
    assert result is None


def test_compute_forecast_not_first_returns_none():
    # Starts on March 5 (not 1st)
    buckets = _make_daily_buckets([10.0] * 10, start_ts=1772323200 + 4 * 86400)
    result = _compute_forecast(buckets, 100.0)
    assert result is None


def test_compute_forecast_single_day():
    buckets = _make_daily_buckets([50.0])
    result = _compute_forecast(buckets, 50.0)
    assert result is not None
    assert "Projected" in result


def test_format_costs_summary_includes_forecast():
    # Partial month starting March 1
    buckets = _make_daily_buckets([10.0] * 10)
    output = format_costs_summary(buckets)
    assert "Projected" in output


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def test_detect_anomalies_with_spike():
    # 9 days at $10, 1 day at $50 (well above 2σ)
    amounts = [10.0] * 9 + [50.0]
    buckets = _make_daily_buckets(amounts)
    result = _detect_anomalies(buckets)
    assert result is not None
    assert "Anomalies" in result
    assert "$50.00" in result


def test_detect_anomalies_no_spikes():
    # All uniform — no anomalies
    buckets = _make_daily_buckets([10.0] * 10)
    result = _detect_anomalies(buckets)
    assert result is None


def test_detect_anomalies_too_few_days():
    buckets = _make_daily_buckets([10.0] * 5)
    result = _detect_anomalies(buckets)
    assert result is None


def test_detect_anomalies_all_zero():
    buckets = _make_daily_buckets([0.0] * 10)
    result = _detect_anomalies(buckets)
    assert result is None  # mean < $1


def test_format_costs_summary_includes_anomalies():
    # 9 days at $10, 1 day at $50
    amounts = [10.0] * 9 + [50.0]
    buckets = _make_daily_buckets(amounts)
    output = format_costs_summary(buckets)
    assert "Anomalies" in output


# ---------------------------------------------------------------------------
# Cost comparison
# ---------------------------------------------------------------------------

COST_BUCKETS_MONTH_A = [
    {
        "object": "bucket",
        "start_time": 1769644800,  # 2026-02-01
        "end_time": 1769731200,
        "results": [
            {"amount": {"value": 100.0}, "line_item": "GPT-4o"},
            {"amount": {"value": 50.0}, "line_item": "Embeddings"},
        ],
    },
]

COST_BUCKETS_MONTH_B = [
    {
        "object": "bucket",
        "start_time": 1772323200,  # 2026-03-01
        "end_time": 1772409600,
        "results": [
            {"amount": {"value": 130.0}, "line_item": "GPT-4o"},
            {"amount": {"value": 35.0}, "line_item": "Embeddings"},
            {"amount": {"value": 20.0}, "line_item": "Images"},
        ],
    },
]


def test_format_cost_comparison_basic():
    output = format_cost_comparison(
        COST_BUCKETS_MONTH_A, COST_BUCKETS_MONTH_B, "2026-02", "2026-03",
    )
    assert "Comparison" in output
    assert "GPT-4o" in output
    assert "Embeddings" in output
    assert "2026-02" in output
    assert "2026-03" in output
    # Overall: $150 -> $185
    assert "$150.00" in output
    assert "$185.00" in output


def test_format_cost_comparison_new_item():
    output = format_cost_comparison(
        COST_BUCKETS_MONTH_A, COST_BUCKETS_MONTH_B, "2026-02", "2026-03",
    )
    # Images is new in comparison month
    assert "new" in output


def test_format_cost_comparison_top_n():
    output = format_cost_comparison(
        COST_BUCKETS_MONTH_A, COST_BUCKETS_MONTH_B, "2026-02", "2026-03", top_n=1,
    )
    assert "Other (2 items)" in output


def test_format_cost_comparison_biggest_movers():
    output = format_cost_comparison(
        COST_BUCKETS_MONTH_A, COST_BUCKETS_MONTH_B, "2026-02", "2026-03",
    )
    assert "Biggest Movers" in output
    assert "Largest increase" in output
    assert "Largest decrease" in output


def test_format_cost_comparison_empty():
    output = format_cost_comparison([], [], "2026-02", "2026-03")
    assert "No cost data" in output


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
    assert "80,000" in output
    assert "18,000" in output
    assert "200" in output


def test_format_usage_summary_empty():
    output = format_usage_summary([], "completions")
    assert "No usage data" in output


def test_format_usage_summary_top_n():
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
