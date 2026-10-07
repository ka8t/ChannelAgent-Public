"""The built-in MCP server "feeds": the recent items of an RSS or Atom feed
(app/feeds.py), fetched through the outbound guard with the WEB_FETCH_* limits of "web".

Launched over stdio by app.mcp.manager with a cleared environment holding only the WEB_FETCH_*
settings (app.mcp.builtin.BUILTIN_SETTINGS). Its tool is read-only and open-world: its results
come from outside (untrusted). Each item carries an `id:` line; in a scheduled task's turn
the application removes the items that task already delivered (app/feed_memory.py).
"""

from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from app import feeds
from app.web_fetch import FetchError, Limits, fetch_bytes

mcp = FastMCP("feeds")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def read_feed(url: str, since: str | None = None, limit: int = 10) -> str:
    """Read the recent items of an RSS or Atom feed (an https address): title, date, link and a
    short summary of each, newest first. `since` (YYYY-MM-DD) keeps the items published on or
    after that day; `limit` is at most 50."""
    after = None
    if since:
        try:
            after = datetime.fromisoformat(since.strip()).replace(tzinfo=UTC)
        except ValueError:
            return "error: since is a date written YYYY-MM-DD"
    try:
        final, data = await fetch_bytes(url, Limits.from_environment(), feeds.FEED_TYPES)
        title, items = feeds.parse(data)
    except (FetchError, feeds.FeedError) as exc:
        return f"error: {exc}"
    return feeds.render(title, final, feeds.select(items, after, int(limit or 10)))


if __name__ == "__main__":
    mcp.run(transport="stdio")
