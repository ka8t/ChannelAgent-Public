"""How a tool call reaches the user for a "confirm" policy, without app.mcp
knowing any channel. app.channels.dispatch sets `current_confirmer` from the event's
own `confirm` callable before the turn runs; the executor of app.mcp.catalogue reads
it. A turn with none (the Admin API, a test, a channel that cannot ask) gets None,
and a "confirm" tool is then refused: nobody said yes.
"""

import contextvars
import re
from collections.abc import Awaitable, Callable

# confirm(question, timeout_seconds) -> True (yes), False (no), None (no answer in time)
Confirmer = Callable[[str, float], Awaitable[bool | None]]

current_confirmer: contextvars.ContextVar[Confirmer | None] = contextvars.ContextVar(
    "mcp_confirmer", default=None
)

# A scheduled task's standing approval: {catalogue name: approved definition sha256}.
# app.tasks.execute sets it around the task's turn; None in every other turn.
current_standing: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "mcp_standing", default=None
)

# The web addresses written in that task's prompt, set beside `current_standing`: a
# read-only tool may read one of them after untrusted content; any other address stays refused.
current_task_addresses: contextvars.ContextVar[frozenset[str] | None] = contextvars.ContextVar(
    "mcp_task_addresses", default=None
)

_ADDRESS = re.compile(r"https?://[^\s<>\"'`]+")


def prompt_addresses(prompt: str) -> frozenset[str]:
    """The http(s) addresses of a task prompt, without the punctuation that ends a sentence."""
    return frozenset(m.rstrip(".,;:!?)]}") for m in _ADDRESS.findall(prompt or ""))
