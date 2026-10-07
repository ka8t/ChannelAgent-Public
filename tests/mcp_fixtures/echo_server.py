"""Test-only MCP server: `echo` returns the text of ECHO_DIR/reply (a declared env
var), so a test can make a tool result carry whatever it needs, and counts its calls in
ECHO_DIR/count. Never registered in app.mcp.builtin.REGISTRY; tests monkeypatch it in."""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

DIR = Path(os.environ["ECHO_DIR"])
mcp = FastMCP("echo")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def echo(query: str = "") -> str:
    """Return the prepared reply."""
    count = DIR / "count"
    count.write_text(str(int(count.read_text()) + 1 if count.exists() else 1))
    reply = DIR / "reply"
    return reply.read_text() if reply.exists() else "nothing"


if __name__ == "__main__":
    mcp.run(transport="stdio")
