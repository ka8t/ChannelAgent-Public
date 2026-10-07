"""Agent templates: what the agent builder starts from, written by an administrator as
data, never as code.

A template has a name, a description (the builder picks a template by it), guidance for the
builder's model, instructions added at the end of the agent's own, an optional fixed memory
mode, the tools it attaches (the part of a tool name after `mcp__<server>__`, attached only when
the user may attach them: a template never widens a right), whether the agent needs a schedule,
and the only questions the builder may ask while it follows it, by what is missing
(`purpose`, `schedule`, `task_prompt`). Every change is a new version holding the whole
template (`agent_template_versions`), and an admin event.

The starter templates (news digest, web page watch, reminder, daily summary) are data in
`app/admin/agent_templates_starter.json`, installed on an administrator's request; one already
present is left as it is.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import ConflictError, InvalidInputError, NotFoundError, record_admin_event
from app.db.models import AgentTemplate, AgentTemplateVersion

NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
TOOL = re.compile(r"[A-Za-z0-9_.-]{1,64}")
MEMORY_MODES = ("off", "ondemand", "always", "search")
QUESTION_FIELDS = ("purpose", "schedule", "task_prompt")
MAX_DESCRIPTION = 200
MAX_GUIDANCE = 4000
MAX_INSTRUCTIONS = 2000
MAX_QUESTION = 300
MAX_TOOLS = 20
STARTER_FILE = Path(__file__).with_name("agent_templates_starter.json")
FIELDS = (
    "description", "guidance", "agent_instructions", "memory_mode", "tools", "needs_schedule",
    "questions", "enabled",
)  # fmt: skip


class TemplateNotFoundError(NotFoundError):
    pass


def _text(value, field: str, limit: int, required: bool = False) -> str:
    if not isinstance(value, str):
        raise InvalidInputError(f"{field} is a text")
    text = value.strip()
    if required and not text:
        raise InvalidInputError(f"A template needs a {field}")
    if len(text) > limit:
        raise InvalidInputError(f"{field} is at most {limit} characters")
    return text


def clean(fields: dict, *, creating: bool) -> dict:
    """The template's fields, checked; every refusal names the field."""
    unknown = sorted(set(fields) - {"name", *FIELDS})
    if unknown:
        raise InvalidInputError(f"Unknown template field(s): {', '.join(unknown)}")
    out: dict = {}
    if "name" in fields or creating:
        name = fields.get("name")
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise InvalidInputError("A template name is lowercase letters, digits and -, up to 40")
        out["name"] = name
    if "description" in fields or creating:
        text = " ".join(_text(fields.get("description"), "description", 10_000, True).split())
        if len(text) > MAX_DESCRIPTION:
            raise InvalidInputError(f"description is at most {MAX_DESCRIPTION} characters")
        out["description"] = text
    if "guidance" in fields:
        out["guidance"] = _text(fields["guidance"], "guidance", MAX_GUIDANCE)
    if "agent_instructions" in fields:
        out["agent_instructions"] = _text(
            fields["agent_instructions"], "agent_instructions", MAX_INSTRUCTIONS
        )
    if "memory_mode" in fields:
        mode = fields["memory_mode"]
        if mode is not None and mode not in MEMORY_MODES:
            raise InvalidInputError(f"memory_mode is null or one of: {', '.join(MEMORY_MODES)}")
        out["memory_mode"] = mode
    if "tools" in fields:
        tools = fields["tools"]
        if (
            not isinstance(tools, list)
            or len(tools) > MAX_TOOLS
            or not all(isinstance(t, str) and TOOL.fullmatch(t) for t in tools)
        ):
            raise InvalidInputError(
                f"tools is a list of at most {MAX_TOOLS} tool names (the part after "
                "mcp__<server>__, for example read_feed)"
            )
        out["tools"] = sorted(set(tools))
    for flag in ("needs_schedule", "enabled"):
        if flag in fields:
            if not isinstance(fields[flag], bool):
                raise InvalidInputError(f"{flag} is a boolean")
            out[flag] = fields[flag]
    if "questions" in fields:
        questions = fields["questions"]
        if not isinstance(questions, dict):
            raise InvalidInputError(
                f"questions maps what is missing ({', '.join(QUESTION_FIELDS)}) to a question"
            )
        bad = sorted(set(questions) - set(QUESTION_FIELDS))
        if bad:
            raise InvalidInputError(
                f"questions are asked for {', '.join(QUESTION_FIELDS)} only, not {', '.join(bad)}"
            )
        out["questions"] = {
            key: _text(value, f"the {key} question", MAX_QUESTION, True)
            for key, value in sorted(questions.items())
        }
    return out


def snapshot(template: AgentTemplate) -> dict:
    return {"name": template.name, **{field: getattr(template, field) for field in FIELDS}}


async def list_templates(session: AsyncSession, enabled_only: bool = False) -> list[AgentTemplate]:
    stmt = select(AgentTemplate).order_by(AgentTemplate.name)
    if enabled_only:
        stmt = stmt.where(AgentTemplate.enabled.is_(True))
    return list((await session.execute(stmt)).scalars())


async def get_template(session: AsyncSession, name: str) -> AgentTemplate:
    found = await session.execute(select(AgentTemplate).where(AgentTemplate.name == name))
    template = found.scalar_one_or_none()
    if template is None:
        raise TemplateNotFoundError(f"No agent template named {name!r}")
    return template


async def versions(session: AsyncSession, name: str) -> list[AgentTemplateVersion]:
    template = await get_template(session, name)
    stmt = select(AgentTemplateVersion).where(AgentTemplateVersion.template_id == template.id)
    return list((await session.execute(stmt.order_by(AgentTemplateVersion.version))).scalars())


async def create_template(
    session: AsyncSession, fields: dict, *, actor: str, source: str = "api"
) -> AgentTemplate:
    cleaned = clean(fields, creating=True)
    exists = await session.execute(
        select(AgentTemplate.id).where(AgentTemplate.name == cleaned["name"])
    )
    if exists.scalar_one_or_none() is not None:
        raise ConflictError(f"An agent template named {cleaned['name']!r} already exists")
    defaults = {"guidance": "", "agent_instructions": "", "memory_mode": None, "tools": [],
                "needs_schedule": False, "questions": {}, "enabled": True}  # fmt: skip
    template = AgentTemplate(**{**defaults, **cleaned}, version=1, source=source)
    session.add(template)
    await session.flush()
    session.add(AgentTemplateVersion(template_id=template.id, version=1, data=snapshot(template)))
    await record_admin_event(
        session, actor=actor, action="agent_template.create", target_type="agent_template",
        target_id=template.id, details={"name": template.name, "version": 1, "source": source},
    )  # fmt: skip
    await session.flush()
    return template


async def update_template(
    session: AsyncSession, name: str, fields: dict, *, actor: str
) -> AgentTemplate:
    template = await get_template(session, name)
    if "name" in fields:
        raise InvalidInputError("A template keeps its name; create another one to rename it")
    cleaned = clean(fields, creating=False)
    changed = sorted(k for k, v in cleaned.items() if getattr(template, k) != v)
    for key in changed:
        setattr(template, key, cleaned[key])
    if changed:
        template.version += 1
        template.updated_at = datetime.now(UTC)
        session.add(AgentTemplateVersion(template_id=template.id, version=template.version,
                                         data=snapshot(template)))  # fmt: skip
    await record_admin_event(
        session, actor=actor, action="agent_template.update", target_type="agent_template",
        target_id=template.id,
        details={"name": template.name, "version": template.version, "fields": changed},
    )  # fmt: skip
    await session.flush()
    return template


async def delete_template(session: AsyncSession, name: str, *, actor: str) -> None:
    template = await get_template(session, name)
    await session.delete(template)
    await record_admin_event(
        session, actor=actor, action="agent_template.delete", target_type="agent_template",
        target_id=template.id, details={"name": name},
    )  # fmt: skip
    await session.flush()


def starter_templates() -> list[dict]:
    data = json.loads(STARTER_FILE.read_text(encoding="utf-8"))
    for fields in data:
        clean(fields, creating=True)
    return data


async def install_starters(session: AsyncSession, *, actor: str) -> dict:
    """Create the starter templates that are missing; one already present stays as it is."""
    present = {t.name for t in await list_templates(session)}
    created, kept = [], []
    for fields in starter_templates():
        if fields["name"] in present:
            kept.append(fields["name"])
            continue
        await create_template(session, fields, actor=actor, source="starter")
        created.append(fields["name"])
    return {"created": created, "kept": kept}
