"""The built-in MCP server "calc": exact arithmetic, date arithmetic and unit
conversion (app/calc.py). A parser of its own, never `eval`; no network, no file. Launched over
stdio by app.mcp.manager with a cleared environment; both tools are read-only and closed-world,
so their default policy is "allow".
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from app import calc

mcp = FastMCP("calc")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def calculate(expression: str) -> str:
    """Compute an expression exactly: numbers, + - * / // %, ^ for a power, parentheses,
    sqrt() and abs(), e.g. "12 * (3 + 4) / 2". Also dates: "2026-09-29 + 3 days" (days, weeks,
    months, years), "2026-12-25 - 2026-10-04" (days between), "weekday 2026-10-04"."""
    try:
        return calc.calculate(expression)
    except calc.CalcError as exc:
        return f"error: {exc}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def convert(value: str, from_unit: str, to_unit: str) -> str:
    """Convert a value between units of one kind: length (km, miles, ft...), mass (kg, lb...),
    volume (L, gal...), area, time, speed (km/h, mph...), data (MB, GiB...) or temperature
    (C, F, K)."""
    try:
        return calc.convert(value, from_unit, to_unit)
    except calc.CalcError as exc:
        return f"error: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
