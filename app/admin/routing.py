"""Model-routing service: which model answers a turn when the agent itself
has none (Agent.model, decision layer 1 — the caller decides that layer,
not this module). Layer 2 is a short ordered list of rules the admin configures
through the Admin API (app/api/routing_routes.py); layer 3 is the default model.
No classifier (D7: rules only, evaluated on what is
known about the message before any model call — no extra latency per turn.

Tool budget: a turn is shown at most `max_tools` MCP tools (0 = no cap). When the agent
may use more, the tools named by the tool rules whose keyword is in the message come first,
then the others by how many words of the message their name and description share; the rest
are not shown this turn. `tool_model`, when set, answers the turns of an agent that uses MCP
tools (after the sender's `/model` choice and the agent's own model, before the rules).
"""

import fnmatch
import re

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import InvalidInputError, _clean_model, record_admin_event
from app.db.models import RoutingConfig, RoutingRule

# Singleton row: one routing table for the whole deployment, not per user or agent.
CONFIG_ID = 1
MATCH_TYPES = ("min_length", "command_prefix")
MAX_MATCH_VALUE_LENGTH = 200
MAX_RULES = 50
# T: the value the benchmark supports.
DEFAULT_MAX_TOOLS = 5
MAX_TOOLS_LIMIT = 100
MAX_TOOL_RULES = 50
MAX_TOOL_PATTERNS = 50
# Words of 4 letters or more: "the", "for" and "get" say nothing about the tool.
_WORD = re.compile(r"[a-z0-9]{4,}")


def _clean_match_value(match_type: str, value) -> str:
    if match_type == "min_length":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidInputError("A min_length rule needs a positive integer match_value")
        return str(value)
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_MATCH_VALUE_LENGTH:
        raise InvalidInputError(
            f"A command_prefix rule needs a non-empty match_value up to "
            f"{MAX_MATCH_VALUE_LENGTH} characters"
        )
    return value


def _clean_rule(rule) -> dict:
    if not isinstance(rule, dict) or set(rule) - {"match_type", "match_value", "model"}:
        raise InvalidInputError("A rule has only match_type, match_value and model")
    match_type = rule.get("match_type")
    if match_type not in MATCH_TYPES:
        raise InvalidInputError(f"match_type is one of: {', '.join(MATCH_TYPES)}")
    model = _clean_model(rule.get("model"))
    if model is None:
        raise InvalidInputError("A rule needs a model")
    return {
        "match_type": match_type,
        "match_value": _clean_match_value(match_type, rule.get("match_value")),
        "model": model,
    }


def _clean_rules(rules) -> list[dict]:
    if not isinstance(rules, list) or len(rules) > MAX_RULES:
        raise InvalidInputError(f"At most {MAX_RULES} rules")
    return [_clean_rule(rule) for rule in rules]


def _clean_model_ctx_sizes(value) -> dict[str, int]:
    if not isinstance(value, dict):
        raise InvalidInputError("model_ctx_sizes is an object of model name to context size")
    cleaned: dict[str, int] = {}
    for name, size in value.items():
        model = _clean_model(name)
        if model is None or isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise InvalidInputError("model_ctx_sizes maps a valid model name to a positive integer")
        cleaned[model] = size
    return cleaned


def _clean_max_tools(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_TOOLS_LIMIT:
        raise InvalidInputError(f"max_tools is a number from 0 (no cap) to {MAX_TOOLS_LIMIT}")
    return value


def _clean_tool_rules(rules) -> list[dict]:
    if not isinstance(rules, list) or len(rules) > MAX_TOOL_RULES:
        raise InvalidInputError(f"At most {MAX_TOOL_RULES} tool rules")
    cleaned = []
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"keyword", "tools"}:
            raise InvalidInputError("A tool rule has a keyword and tools")
        keyword, tools = rule["keyword"], rule["tools"]
        if not isinstance(keyword, str) or not keyword.strip() or len(keyword) > 100:
            raise InvalidInputError("A tool rule's keyword is 1 to 100 characters")
        if (
            not isinstance(tools, list)
            or not 0 < len(tools) <= MAX_TOOL_PATTERNS
            or not all(isinstance(t, str) and 0 < len(t) <= 200 for t in tools)
        ):
            raise InvalidInputError(
                f"A tool rule names 1 to {MAX_TOOL_PATTERNS} tools (mcp__server__tool, * allowed)"
            )
        cleaned.append({"keyword": keyword.strip(), "tools": tools})
    return cleaned


async def get_routing(session: AsyncSession) -> dict:
    """The routing table as it stands: the default model, the rules in order, and
    the per-model context sizes. Empty defaults when nothing was ever set, so a
    turn behaves exactly as without routing until an admin configures something.
    """
    config = await session.get(RoutingConfig, CONFIG_ID)
    rows = (
        (await session.execute(select(RoutingRule).order_by(RoutingRule.position))).scalars().all()
    )
    return {
        "default_model": config.default_model if config else None,
        "model_ctx_sizes": dict(config.model_ctx_sizes) if config else {},
        "max_tools": config.max_tools if config else DEFAULT_MAX_TOOLS,
        "tool_rules": list(config.tool_rules or []) if config else [],
        "tool_model": config.tool_model if config else None,
        "rules": [
            {"match_type": r.match_type, "match_value": r.match_value, "model": r.model}
            for r in rows
        ],
    }


async def set_routing(
    session: AsyncSession,
    *,
    default_model=None,
    rules=None,
    model_ctx_sizes=None,
    max_tools=DEFAULT_MAX_TOOLS,
    tool_rules=None,
    tool_model=None,
    actor: str,
) -> dict:
    """Replace the whole routing table: simpler and safer to reason about than a
    partial patch of an ordered list, and the admin UI always sends the full
    state anyway (mirrors PUT semantics).
    """
    cleaned_default = _clean_model(default_model)
    cleaned_rules = _clean_rules(rules if rules is not None else [])
    cleaned_ctx = _clean_model_ctx_sizes(model_ctx_sizes if model_ctx_sizes is not None else {})
    cleaned_max_tools = _clean_max_tools(max_tools)
    cleaned_tool_rules = _clean_tool_rules(tool_rules if tool_rules is not None else [])
    cleaned_tool_model = _clean_model(tool_model)

    config = await session.get(RoutingConfig, CONFIG_ID)
    if config is None:
        config = RoutingConfig(id=CONFIG_ID)
        session.add(config)
    config.default_model = cleaned_default
    config.model_ctx_sizes = cleaned_ctx
    config.max_tools = cleaned_max_tools
    config.tool_rules = cleaned_tool_rules
    config.tool_model = cleaned_tool_model

    await session.execute(delete(RoutingRule))
    for position, rule in enumerate(cleaned_rules):
        session.add(RoutingRule(position=position, **rule))
    await session.flush()

    await record_admin_event(
        session,
        actor=actor,
        action="routing.set",
        target_type="routing",
        details={
            "default_model": cleaned_default,
            "rules": len(cleaned_rules),
            "model_ctx_sizes": cleaned_ctx,
            "max_tools": cleaned_max_tools,
            "tool_rules": len(cleaned_tool_rules),
            "tool_model": cleaned_tool_model,
        },
    )
    return await get_routing(session)


def select_model(*, text: str, rules: list[dict], default_model: str | None) -> str | None:
    """Decision layers 2 and 3 (layer 1, the agent's own model, is decided by the
    caller, app.graph.run_turn, before this is even called): the first rule that
    matches what is known about the message without a model call, or the default.
    """
    for rule in rules:
        if _rule_matches(rule, text):
            return rule["model"]
    return default_model


def _rule_matches(rule: dict, text: str) -> bool:
    if rule["match_type"] == "min_length":
        return len(text) >= int(rule["match_value"])
    if rule["match_type"] == "command_prefix":
        return text.startswith(rule["match_value"])
    return False


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower().replace("_", " ")))


def _shared(message: set[str], tool: set[str]) -> int:
    """How many words of the message a tool's words share, a word that begins another counting
    ("remind" and "reminder")."""
    return sum(any(v.startswith(w) or w.startswith(v) for v in tool) for w in message)


def select_tools(*, text: str, tools: list[dict], rules: list[dict], max_tools: int) -> list[dict]:
    """The tools a turn is shown, in the OpenAI `tools` format, at most `max_tools`
    of them (0 = all): first those of the tool rules whose keyword is in the message (in rule
    order), then the others by the number of words of the message their name and description
    share (a word that begins another counts; ties keep the catalogue order)."""
    if not max_tools or len(tools) <= max_tools:
        return tools
    lowered = text.lower()
    names = [t["function"]["name"] for t in tools]
    chosen: list[str] = []
    for rule in rules:
        if rule["keyword"].lower() in lowered:
            for name in names:
                if name not in chosen and any(fnmatch.fnmatchcase(name, p) for p in rule["tools"]):
                    chosen.append(name)
    words = _words(text)
    rest = sorted(
        (t for t in tools if t["function"]["name"] not in chosen),
        key=lambda t: -_shared(
            words, _words(t["function"]["name"] + " " + (t["function"].get("description") or ""))
        ),
    )
    chosen += [t["function"]["name"] for t in rest]
    keep = set(chosen[:max_tools])
    return [t for t in tools if t["function"]["name"] in keep]
