"""Tests for the built-in MCP server "calc": exact arithmetic, dates and units with a
parser of its own (no eval), bounded input and results, read-only and closed-world."""

import re
import time

import pytest

from app import calc
from app.calc import CalcError, calculate, convert

# --- arithmetic ---


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("12 * (3 + 4) / 2", "42"),
        ("0.1 + 0.2", "0.3"),
        ("1/3", "1/3 (about 0.333333333333)"),
        ("10 / 4", "2.5"),
        ("2^10", "1024"),
        ("2**-2", "0.25"),
        ("-3^2", "-9"),
        ("(-3)^2", "9"),
        ("2 - -3", "5"),
        ("7 // 2", "3"),
        ("-7 // 2", "-4"),
        ("7 % 3", "1"),
        ("1e3 + 1", "1001"),
        ("sqrt(16)", "4"),
        ("sqrt(2)", "about 1.41421356237"),
        ("2^0.5", "about 1.41421356237"),
        ("abs(-2.5)", "2.5"),
        ("2^64", "18446744073709551616"),
        ("  ((((1)))) ", "1"),
    ],
)
def test_arithmetic_is_exact(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression, message",
    [
        ("", "empty"),
        ("1/0", "division by zero"),
        ("5 % 0", "division by zero"),
        ("0^-1", "division by zero"),
        ("sqrt(-1)", "negative"),
        ("2^(1/3)", "whole exponent"),
        ("1 +", "ends too early"),
        ("(1 + 2", "expected ')'"),
        ("1 2", "unexpected '2'"),
        ("foo(1)", "unknown name 'foo'"),
        ("__import__('os').system('ls')", "unexpected character '_'"),
        ("open('/etc/passwd')", "unexpected character \"'\""),
        ("open(1)", "unknown name 'open'"),
        ("1; 2", "unexpected character ';'"),
        ("[1]", "unexpected character '['"),
    ],
)
def test_an_invalid_or_code_like_input_is_refused(expression, message):
    with pytest.raises(CalcError, match=re.escape(message)):
        calculate(expression)


def test_the_input_and_every_result_are_bounded():
    with pytest.raises(CalcError, match="longer than 200 characters"):
        calculate("1+" * 100 + "1")
    assert calculate("1+" * 99 + "1") == "100"
    with pytest.raises(CalcError, match="nested levels"):
        calculate("(" * 31 + "1" + ")" * 31)
    assert calculate("(" * 30 + "1" + ")" * 30) == "1"
    with pytest.raises(CalcError, match="longer than 60 characters"):
        calculate("1" * 61)
    started = time.monotonic()
    for huge in ("1e999999999", "9^9^9", "10^1001", "2^99999999", "(10^999)*(10^999)"):
        with pytest.raises(CalcError, match="more than 1000 digits"):
            calculate(huge)
    assert time.monotonic() - started < 1, "refused before being computed"
    assert len(calculate("10^999")) == 1000


# --- dates ---


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2026-09-29 + 3 days", "2026-10-02 (Friday)"),
        ("2026-10-04 - 2 weeks + 1 day", "2026-09-21 (Monday)"),
        ("2026-01-31 + 1 month", "2026-02-28 (Saturday)"),
        ("2024-02-29 + 1 year", "2025-02-28 (Friday)"),
        ("2026-12-25 - 2026-10-04", "82 days (11 weeks and 5 days)"),
        ("2026-10-04 - 2026-12-25", "-82 days (-11 weeks and 5 days)"),
        ("weekday 2026-10-04", "2026-10-04 (Sunday)"),
    ],
)
def test_date_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression, message",
    [
        ("2026-02-30 + 1 day", "not a valid date"),
        ("2026-10-04 + 3 fortnights", "unknown date unit"),
        ("9999-12-31 + 1 day", "years 1 to 9999"),
        ("9999-12-31 + 1 month", "years 1 to 9999"),
        ("2026-10-04 + 9999999 days", "too large"),
    ],
)
def test_an_invalid_date_expression_is_refused(expression, message):
    with pytest.raises(CalcError, match=message):
        calculate(expression)


# --- units ---


@pytest.mark.parametrize(
    "args, expected",
    [
        ((42, "km", "miles"), "42 km = about 26.097590074 mi"),
        ((1, "mile", "km"), "1 mi = 1.609344 km"),
        ((100, "C", "F"), "100 C = 212 F"),
        ((-40, "fahrenheit", "celsius"), "-40 F = -40 C"),
        ((0, "K", "C"), "0 K = -273.15 C"),
        ((1, "GiB", "MB"), "1 GiB = 1073.741824 MB"),
        ((5, "feet", "m"), "5 ft = 1.524 m"),
        (("3.5", "hours", "min"), "3.5 h = 210 min"),
        ((1, "lb", "g"), "1 lb = 453.59237 g"),
        ((1, "acre", "m2"), "1 acre = 4046.8564224 m2"),
        ((1, "uk gallon", "L"), "1 ukgal = 4.54609 L"),
    ],
)
def test_conversions_are_exact(args, expected):
    assert convert(*args) == expected


@pytest.mark.parametrize(
    "args, message",
    [
        ((1, "kg", "m"), "cannot be converted"),
        ((1, "C", "km"), "cannot be converted"),
        ((1, "parsec", "m"), "unknown unit 'parsec'"),
        (("abc", "kg", "g"), "not a number"),
        (("nan", "kg", "g"), "not a number"),
        (("1e5000", "kg", "g"), "more than 1000 digits"),
    ],
)
def test_an_invalid_conversion_is_refused(args, message):
    with pytest.raises(CalcError, match=message):
        convert(*args)


# --- the server ---


def test_the_calc_server_is_read_only_closed_world_and_gets_no_setting():
    from app.mcp import builtin
    from app.mcp.builtin_servers import calc_server

    assert builtin.REGISTRY["calc"] == "app.mcp.builtin_servers.calc_server"
    assert builtin.builtin_environment("calc") == {}
    for name in ("calculate", "convert"):
        tool = calc_server.mcp._tool_manager.get_tool(name)
        assert (tool.annotations.readOnlyHint, tool.annotations.openWorldHint) == (True, False)
    assert calc_server.calculate("1/0") == "error: division by zero"
    assert calc_server.convert("1", "kg", "m").startswith("error: kg (mass) cannot")


async def test_a_call_round_trips_through_the_real_calc_server():
    from app.db.models import McpTransport
    from app.mcp.manager import ManagedServer, ServerConfig

    server = ManagedServer(
        ServerConfig(
            name="calc", protocol=McpTransport.STDIO, builtin_id="calc", timeout_seconds=20,
            concurrency_limit=1, result_max_bytes=100_000,
        )
    )  # fmt: skip
    try:
        assert sorted(t.name for t in await server.list_tools()) == ["calculate", "convert"]
        assert await server.call_tool("calculate", {"expression": "12 * (3 + 4) / 2"}) == (
            "42",
            False,
        )
        text, _ = await server.call_tool("convert", {"value": "42", "from_unit": "km",
                                                     "to_unit": "miles"})  # fmt: skip
        assert text == "42 km = about 26.097590074 mi"
    finally:
        await server.disconnect()


def test_the_module_never_evaluates_code():
    source = open(calc.__file__).read()
    forbidden_calls = (
        r"\beval\(", r"\bexec\(", r"(?<!re\.)\bcompile\(", "__import__", "import ast",
    )
    for forbidden in forbidden_calls:
        assert not re.search(forbidden, source), forbidden


def test_every_built_in_server_module_exists():
    """A registry entry whose module is missing cannot start (the "filesystem" entry, removed)."""
    import importlib.util

    from app.mcp.builtin import REGISTRY

    assert all(importlib.util.find_spec(module) for module in REGISTRY.values()), REGISTRY
