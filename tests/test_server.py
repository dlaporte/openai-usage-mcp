import pytest
from openai_usage_mcp.server import format_costs_response, parse_date_to_unix


def test_parse_date_to_unix():
    ts = parse_date_to_unix("2026-03-01")
    # 2026-03-01 00:00:00 UTC
    assert ts == 1772323200


def test_parse_date_to_unix_none_returns_none():
    assert parse_date_to_unix(None) is None


def test_format_costs_response_by_line_item():
    raw = [
        {
            "object": "bucket",
            "start_time": 1772323200,
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
        }
    ]
    output = format_costs_response(raw)
    assert "2026-03-01" in output
    assert "GPT-4o" in output
    assert "$12.50" in output
    assert "$3.25" in output
    assert "$15.75" in output  # daily total


def test_format_costs_response_empty():
    output = format_costs_response([])
    assert "No cost data" in output


from openai_usage_mcp.server import format_usage_response

VALID_SERVICE_TYPES = [
    "completions", "embeddings", "images", "audio_speeches",
    "audio_transcriptions", "moderations", "vector_stores",
    "code_interpreter_sessions",
]


def test_format_usage_response_completions():
    raw = [
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
        }
    ]
    output = format_usage_response(raw, "completions")
    assert "2026-03-01" in output
    assert "50,000" in output  # input tokens formatted
    assert "10,000" in output  # output tokens
    assert "5,000" in output   # cached tokens
    assert "120" in output     # requests
    assert "gpt-4o" in output


def test_format_usage_response_embeddings():
    raw = [
        {
            "object": "bucket",
            "start_time": 1772323200,
            "end_time": 1772409600,
            "results": [
                {
                    "object": "organization.usage.embeddings.result",
                    "input_tokens": 200000,
                    "num_model_requests": 50,
                    "model": "text-embedding-3-small",
                    "project_id": None,
                    "user_id": None,
                    "api_key_id": None,
                },
            ],
        }
    ]
    output = format_usage_response(raw, "embeddings")
    assert "200,000" in output
    assert "text-embedding-3-small" in output


def test_format_usage_response_empty():
    output = format_usage_response([], "completions")
    assert "No usage data" in output
