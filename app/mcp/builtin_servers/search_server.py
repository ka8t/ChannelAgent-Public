"""The built-in MCP server "search": a web search through the owner's own SearXNG
instance (app/search.py), results as title, link and snippet; a page is then read with
`fetch_page` (the server "web").

Launched over stdio by app.mcp.manager with a cleared environment holding only SEARXNG_URL,
SEARCH_MAX_RESULTS and WEB_FETCH_TIMEOUT_SECONDS (app.mcp.builtin.BUILTIN_SETTINGS). Its tool is
read-only and open-world: the results come from outside (untrusted), and the query leaves
for the search engines, so the server is declared with egress "internet" (outbound).
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from app.search import SearchError
from app.search import web_search as search

mcp = FastMCP("search")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def web_search(query: str) -> str:
    """Search the web: the title, link and a short extract of the best results. Read a result's
    page afterwards with fetch_page."""
    try:
        return await search(query)
    except SearchError as exc:
        return f"error: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
