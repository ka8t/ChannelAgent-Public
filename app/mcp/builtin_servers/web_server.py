"""The built-in MCP server "web": read one web page as Markdown (app/web_fetch.py).

Launched over stdio by app.mcp.manager with a cleared environment holding only the
WEB_FETCH_* settings (app.mcp.builtin.BUILTIN_SETTINGS), never the rest of `.env`. Its tool is
read-only and open-world: its results come from outside (untrusted), and declared with
egress "internet" it is outbound too, so a second page asked for after a first one in the same
turn waits for the user's yes.
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from app.web_fetch import FetchError, Limits, fetch_markdown

mcp = FastMCP("web")

MAX_RESULT_CHARS = 20000


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def fetch_page(url: str) -> str:
    """Read one public web page (an https address) and return its main text as Markdown."""
    try:
        final, markdown = await fetch_markdown(url, Limits.from_environment())
    except FetchError as exc:
        return f"error: {exc}"
    if len(markdown) > MAX_RESULT_CHARS:
        markdown = markdown[:MAX_RESULT_CHARS] + "\n\n[... the page goes on, cut here]"
    return f"Source: {final}\n\n{markdown}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
