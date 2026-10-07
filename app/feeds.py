"""Reading RSS and Atom feeds: the items of a feed as title, link, date and summary.

Parsed with the standard library (xml.etree on expat, 2.7.4 in the locked Python), no extra
dependency: expat refuses exponential entity expansion ("billion laughs") since 2.4.0, and
ElementTree resolves no external entity. A document that declares entities (`<!ENTITY`) is
refused before parsing all the same, since a feed never needs one. Summaries are reduced to
plain text (tags removed, entities decoded) and cut; every item gets an id, the first 16 hex
of the SHA-256 of its link (or of its title and date when it has no link), which is what a
task remembers to deliver an item only once (app/feed_memory.py).

This module never reads `.env`: it runs inside the built-in MCP server "feeds".
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser

MAX_ITEMS = 50
MAX_SUMMARY_CHARS = 300  # A long result is slow for a local model (measured)
MAX_TITLE_CHARS = 300
ITEM_ID = re.compile(r"^id: ([0-9a-f]{16})$", re.MULTILINE)
FEED_TYPES = (
    "application/rss+xml",
    "application/atom+xml",
    "application/xml",
    "text/xml",
    "application/rdf+xml",
    "application/x-rss+xml",
    "text/plain",
    # Served for a feed by real sites (blog.python.org, measured 2026-09-28); the parser still
    # refuses anything that is not RSS or Atom.
    "application/octet-stream",
)
_ATOM = "{http://www.w3.org/2005/Atom}"
_CONTENT = "{http://purl.org/rss/1.0/modules/content/}encoded"
_DC_DATE = "{http://purl.org/dc/elements/1.1/}date"
_RSS1 = "{http://purl.org/rss/1.0/}"


class FeedError(Exception):
    """The document is not a feed this module reads; the message is safe to show."""


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    link: str
    published: datetime | None
    summary: str


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(html: str) -> str:
    parser = _Text()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        return " ".join((html or "").split())
    return " ".join(" ".join(parser.parts).split())


def _cut(text: str, size: int) -> str:
    return text if len(text) <= size else text[: size - 3].rstrip() + "..."


def _date(text: str | None) -> datetime | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = parsedate_to_datetime(text)  # RSS: RFC 822
    except (TypeError, ValueError, IndexError):
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))  # Atom: RFC 3339
        except ValueError:
            return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def item_id(link: str, title: str, published: datetime | None) -> str:
    key = link or f"{title}|{published.isoformat() if published else ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _find_text(node, *tags) -> str:
    for tag in tags:
        child = node.find(tag)
        if child is not None and (child.text or "").strip():
            return child.text.strip()
    return ""


def parse(data: bytes) -> tuple[str, list[Item]]:
    """(feed title, items newest first) of an RSS 2.0, RSS 1.0 or Atom document."""
    if b"<!ENTITY" in data[:65536].upper():
        raise FeedError("the document declares entities, which a feed never needs")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise FeedError(f"not a readable XML feed ({exc})") from None
    items: list[Item] = []
    if root.tag == f"{_ATOM}feed":
        title = _find_text(root, f"{_ATOM}title")
        for entry in root.iter(f"{_ATOM}entry"):
            link = ""
            for candidate in entry.findall(f"{_ATOM}link"):
                if candidate.get("rel", "alternate") == "alternate" and candidate.get("href"):
                    link = candidate.get("href").strip()
                    break
            summary = _find_text(entry, f"{_ATOM}summary", f"{_ATOM}content")
            published = _date(_find_text(entry, f"{_ATOM}published", f"{_ATOM}updated"))
            items.append(_item(_find_text(entry, f"{_ATOM}title"), link, published, summary))
    elif root.tag in ("rss", "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF"):
        channel = root.find("channel")
        if channel is None:
            channel = root.find(f"{_RSS1}channel")
        title = _find_text(channel, "title", f"{_RSS1}title") if channel is not None else ""
        entries = list(root.iter("item")) + list(root.iter(f"{_RSS1}item"))
        for entry in entries:
            link = _find_text(entry, "link", f"{_RSS1}link", "guid")
            summary = _find_text(entry, "description", f"{_RSS1}description", _CONTENT)
            published = _date(_find_text(entry, "pubDate", _DC_DATE))
            entry_title = _find_text(entry, "title", f"{_RSS1}title")
            items.append(_item(entry_title, link, published, summary))
    else:
        raise FeedError("not an RSS or Atom feed")
    epoch = datetime.min.replace(tzinfo=UTC)
    items.sort(key=lambda i: i.published or epoch, reverse=True)
    return plain_text(title), items


def _item(title: str, link: str, published: datetime | None, summary: str) -> Item:
    title = _cut(plain_text(title), MAX_TITLE_CHARS) or "(no title)"
    return Item(
        id=item_id(link, title, published),
        title=title,
        link=link,
        published=published,
        summary=_cut(plain_text(summary), MAX_SUMMARY_CHARS),
    )


def select(items: list[Item], since: datetime | None, limit: int) -> list[Item]:
    chosen = [i for i in items if since is None or (i.published and i.published >= since)]
    return chosen[: max(1, min(limit, MAX_ITEMS))]


def render(title: str, url: str, items: list[Item]) -> str:
    """The tool's text: one block per item, each with its `id:` line (what a task remembers)."""
    lines = [f"Feed: {title or '(untitled)'}", f"Source: {url}", f"Items: {len(items)}"]
    for item in items:
        when = item.published.strftime("%Y-%m-%d %H:%M UTC") if item.published else "no date"
        lines += ["", f"### {item.title}", f"id: {item.id}", f"date: {when}"]
        if item.link:
            lines.append(f"link: {item.link}")
        if item.summary:
            lines.append(item.summary)
    return "\n".join(lines)


def split_blocks(text: str) -> tuple[str, list[tuple[str | None, str]]]:
    """The header and the item blocks of a rendered result, each block with its id."""
    head, *blocks = text.split("\n\n### ")
    out = []
    for block in blocks:
        match = ITEM_ID.search(block)
        out.append((match.group(1) if match else None, "### " + block))
    return head, out
