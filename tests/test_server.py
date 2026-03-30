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
