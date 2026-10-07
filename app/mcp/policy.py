"""Permissions, policies, definition pinning and argument redaction for MCP tools
.

Pure functions, no database and no network: app.mcp.catalogue applies them to a
turn, app.admin.mcp to the administration.
"""

import enum
import hashlib
import json
import re

from app.logging_setup import REDACTED, scrub


class Policy(enum.StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


POLICIES = tuple(p.value for p in Policy)


class Decision(enum.StrEnum):
    """What was decided before a call, stored on its McpCall row (at most 16 characters)."""

    ALLOWED = "allowed"  # policy allow
    CONFIRMED = "confirmed"  # policy confirm, the user said yes
    DECLINED = "declined"  # policy confirm, the user said no
    TIMEOUT = "timeout"  # policy confirm, no answer in time
    NO_CHANNEL = "no_channel"  # policy confirm, but nobody can be asked (no channel)
    DENIED = "denied"  # policy deny
    NOT_GRANTED = "not_granted"  # no grant for this user, agent and tool
    UNAPPROVED = "unapproved"  # definition never approved, or changed since
    CREDENTIALS = "credentials"  # holds a credential but is not flagged shared (M5)
    NOT_OFFERED = "not_offered"  # not in this turn's catalogue for another reason
    INVALID = "invalid"  # arguments are not a JSON object
    # Nobody can be asked (a scheduled task's turn), and the task holds a standing
    # approval for this tool and its current approved definition.
    STANDING = "standing"
    # A standing approval exists, but for a definition an administrator has replaced.
    STALE = "stale_approval"
    # The arguments carry a secret, an internal address, a traversal or a shell
    # injection (app.mcp.guard.check_arguments).
    BLOCKED = "blocked"
    # The user's tools are suspended after repeated threats.
    SUSPENDED = "suspended"


def tool_definition(tool) -> dict:
    """What an administrator approves: everything the model reads about a tool, and
    the annotations the default policy is derived from. A change to any of it changes
    the hash."""
    data = tool.model_dump(mode="json", exclude_none=True, by_alias=True)
    keys = ("name", "title", "description", "inputSchema", "outputSchema", "annotations")
    return {key: data[key] for key in keys if key in data}


def definition_hash(definition: dict) -> str:
    canonical = json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def default_policy(definition: dict) -> Policy:
    """M4: `allow` only for a tool that says it is read-only and not open-world;
    `confirm` for everything else, a tool with no annotations included (treated as
    destructive). Annotations are hints written by the server, so they only choose a
    default: they never make a tool callable without a grant and an approval."""
    annotations = definition.get("annotations") or {}
    if annotations.get("readOnlyHint") is True and annotations.get("openWorldHint") is False:
        return Policy.ALLOW
    return Policy.CONFIRM


# What a tool can do, for the read-then-write rule and the exposure view.
CLASSES = ("private", "untrusted", "outbound")


def default_classes(definition: dict, egress: str) -> dict[str, bool]:
    """Derived from the tool's annotations and its server's egress label:

    - private: the tool works on a closed domain (`openWorldHint` false): the user's own data;
    - untrusted: its results may come from outside (`openWorldHint` not false, the MCP default
      being true) or its server may reach the internet;
    - outbound: it changes something (`readOnlyHint` not true, the MCP default being false) or
      its server may reach the internet.

    Hints are written by the server: an administrator can override each class per tool."""
    annotations = definition.get("annotations") or {}
    open_world = annotations.get("openWorldHint") is not False
    read_only = annotations.get("readOnlyHint") is True
    internet = egress == "internet"
    return {
        "private": not open_world,
        "untrusted": open_world or internet,
        "outbound": not read_only or internet,
    }


def classes(tool_name: str, definition: dict, egress: str, tool_classes: dict) -> dict[str, bool]:
    """The classes in force: the derived ones, then the administrator's overrides."""
    result = default_classes(definition, egress)
    result.update(tool_classes.get(tool_name) or {})
    return result


def effective_policy(tool_name: str, definition: dict, tool_policies: dict) -> Policy:
    explicit = tool_policies.get(tool_name)
    return Policy(explicit) if explicit in POLICIES else default_policy(definition)


def is_granted(grants: frozenset, server: str, tool: str) -> bool:
    """`grants` holds (server, tool) pairs, tool None meaning every tool of the server."""
    return (server, tool) in grants or (server, None) in grants


# A key that names a secret has its value replaced whatever the value looks like.
_SENSITIVE_KEY = re.compile(
    r"pass|secret|token|api[_-]?key|apikey|auth|credential|cookie|session|private[_-]?key",
    re.IGNORECASE,
)
MAX_STORED_ARGUMENTS = 4000


def redact(value, key: str = ""):
    if key and _SENSITIVE_KEY.search(key):
        return REDACTED
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return scrub(value)
    return value


def redacted_arguments(arguments) -> str:
    """The JSON stored on the audit row: secret-named keys and every configured secret
    replaced, then capped."""
    text = json.dumps(redact(arguments), ensure_ascii=False, sort_keys=True)
    if len(text) > MAX_STORED_ARGUMENTS:
        text = text[:MAX_STORED_ARGUMENTS] + "...(truncated)"
    return text
