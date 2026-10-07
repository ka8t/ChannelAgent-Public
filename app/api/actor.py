"""Who is acting, for the admin events.

The script says who it is with an `X-Client: cli:<operating-system user>` header, the admin
UI with `ui:web`; anything else is recorded as `api`. Until named administrators
exist (deferred) the key is shared, so this label is what tells a script action from a raw
API call. It is a label, not a proof: whoever holds the key can send any value that matches
the pattern.
"""

import re
from contextvars import ContextVar

API_ACTOR = "api"
_CLI = re.compile(r"(cli|ui):[A-Za-z0-9._-]{1,32}")
_current: ContextVar[str] = ContextVar("actor", default=API_ACTOR)


def actor_from_header(value: str | None) -> str:
    return value if value and _CLI.fullmatch(value) else API_ACTOR


def set_actor(actor: str) -> None:
    _current.set(actor)


def current_actor() -> str:
    return _current.get()
