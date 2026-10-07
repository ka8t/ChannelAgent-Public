"""Test-only MCP server: a controllable `sleep` tool for the manager's
timeout, concurrency and backoff tests, so those exercise a real subprocess and a
real stdio round trip instead of a mock session. Never registered in
app.mcp.builtin.REGISTRY; tests reach it by monkeypatching that registry.
"""

import asyncio

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slow")


@mcp.tool()
async def sleep(seconds: float) -> str:
    """Sleep for a number of seconds, then answer."""
    await asyncio.sleep(seconds)
    return f"slept {seconds}s"


@mcp.tool()
def big(bytes: int) -> str:
    """Return a string of the given length immediately."""
    return "x" * bytes


if __name__ == "__main__":
    mcp.run(transport="stdio")
