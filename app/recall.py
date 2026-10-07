"""Recall archive: lexical search over the turns
the model no longer sees.

The checkpoint keeps a conversation's whole history; only what is sent to the model is cut
to the window, and the summary keeps the gist, not the detail. When part of the
conversation has left the window, the turn offers one more tool, `recall_search`: it looks
for words in the messages outside the window, and in the other conversations of the same
user with the same agent (the user's other channels). Never another user's or another
agent's: the archive is per (user, agent).

Zero dependency: words of three letters or more, lowercased, accents removed; each message
is scored by the query words it holds, a rare word weighing more than a common one (inverse
document frequency), the more recent message first on a tie.
"""

import json
import logging
import math
import re
import unicodedata
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger("channelagent")

TOOL_NAME = "recall_search"
GUIDANCE = (
    "Earlier parts of this conversation are no longer shown to you. When the user refers to "
    "something said before that you do not see, call recall_search before answering that "
    "you do not know."
)
RESULT_LIMIT = 5
SNIPPET_CHARS = 400
MAX_QUERY_CHARS = 500


def tool_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Search the earlier part of your conversations with this user that you no "
                "longer see (older turns, and the user's other channels). Returns the best "
                "matching messages with their place in the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "words to look for"},
                },
                "required": ["query"],
            },
        },
    }


def words(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return [w for w in re.findall(r"\w+", plain) if len(w) > 2]


def _snippet(text: str, wanted: set[str]) -> str:
    if len(text) <= SNIPPET_CHARS:
        return text
    lowered = text.lower()
    first = min((lowered.find(w) for w in wanted if lowered.find(w) >= 0), default=0)
    start = max(0, first - SNIPPET_CHARS // 4)
    part = text[start : start + SNIPPET_CHARS]
    return ("..." if start else "") + part + ("..." if start + SNIPPET_CHARS < len(text) else "")


def search(
    archive: Sequence[tuple[str, int, int, BaseMessage]], query: str, limit: int = RESULT_LIMIT
) -> list[tuple[str, int, int, BaseMessage]]:
    """The `limit` best (label, position, total, message) of `archive` for `query`."""
    wanted = set(words(query[:MAX_QUERY_CHARS]))
    if not wanted:
        return []
    bags = [set(words(str(message.content))) for *_rest, message in archive]
    count = len(bags) or 1
    frequency = {w: sum(1 for bag in bags if w in bag) for w in wanted}
    scored = []
    for order, (entry, bag) in enumerate(zip(archive, bags, strict=True)):
        score = sum(math.log(1 + count / frequency[w]) for w in wanted if w in bag)
        if score:
            scored.append((score, order, entry))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [entry for _score, _order, entry in scored[:limit]]


def _speaker(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "you"
    return message.type


def format_results(found, query: str) -> str:
    if not found:
        return "nothing found"
    wanted = set(words(query))
    return "\n".join(
        f"[{label}, message {position} of {total}, {_speaker(message)}] "
        f"{_snippet(str(message.content), wanted)}"
        for label, position, total, message in found
    )


async def _other_threads(user_id: int, agent_id: int, current: str) -> list[str]:
    from sqlalchemy import select

    from app.db.models import ChannelIdentity
    from app.db.session import session_scope
    from app.graph import thread_id_from_key

    async with session_scope() as session:
        identities = (
            (
                await session.execute(
                    select(ChannelIdentity).where(ChannelIdentity.user_id == user_id)
                )
            )
            .scalars()
            .all()
        )
        threads = [thread_id_from_key(i.channel, i.external_id, agent_id) for i in identities]
    return [t for t in threads if t != current]


async def _thread_messages(thread_id: str) -> list[BaseMessage]:
    from app.graph import get_graph

    graph = await get_graph()
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return list(state.values.get("messages", [])) if state and state.values else []


def make_executor(
    user_id: int, agent_id: int, thread_id: str, out_of_window: list[BaseMessage], total: int
):
    """`executor(name, raw_arguments)` for `recall_search` in one turn. `out_of_window` is
    the current conversation's messages before the window; the user's other conversations
    with this agent are read when the tool is called. Never raises (app.tools' contract)."""

    async def run(name: str, raw_arguments: str) -> str:
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            return "error: invalid arguments: not JSON"
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if not isinstance(query, str) or not query.strip():
            return "error: query is the words to look for"
        archive = [
            ("this conversation", position, total, message)
            for position, message in enumerate(out_of_window, start=1)
            if isinstance(message, HumanMessage | AIMessage)
        ]
        for other in await _other_threads(user_id, agent_id, thread_id):
            messages = await _thread_messages(other)
            label = f"conversation {other.split('_', 1)[0]}"
            archive.extend(
                (label, position, len(messages), message)
                for position, message in enumerate(messages, start=1)
                if isinstance(message, HumanMessage | AIMessage)
            )
        found = search(archive, query)
        logger.info("Recall search: %d of %d archived messages matched", len(found), len(archive))
        return format_results(found, query)

    return run
