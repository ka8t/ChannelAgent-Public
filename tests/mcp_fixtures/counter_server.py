"""Test-only MCP server: counts the calls it really receives, so a test can
show that a refused call never reached it. The count and the description of `bump`
live in COUNTER_DIR (a declared env var), which also lets a test change that
description between two connections (definition pinning). Never registered in
app.mcp.builtin.REGISTRY; tests reach it by monkeypatching that registry.

Tools: `bump` has no annotations (default policy "confirm", M4); `peek` is read-only
and closed-world (default "allow"); `drop` says it is destructive.
"""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

DIR = Path(os.environ["COUNTER_DIR"])
COUNT = DIR / "count"
DESCRIPTION = DIR / "bump_description"

mcp = FastMCP("counter")


def _count() -> int:
    return int(COUNT.read_text()) if COUNT.exists() else 0


def _bump() -> int:
    value = _count() + 1
    COUNT.write_text(str(value))
    return value


def bump(note: str = "") -> str:
    return f"count {_bump()}"


mcp.add_tool(
    bump,
    name="bump",
    description=DESCRIPTION.read_text() if DESCRIPTION.exists() else "Add one to the counter.",
)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def peek(note: str = "") -> str:
    """Read the counter (counts as a call too)."""
    return f"count {_bump()}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def drop(note: str = "") -> str:
    """Reset the counter."""
    return f"count {_bump()}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
