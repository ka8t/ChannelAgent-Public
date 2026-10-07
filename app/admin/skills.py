"""Local skills (decision M3): instructions an agent loads when it needs them.

A skill has a name, a one-line description, a body and the tools it expects. It is created
through the Admin API or imported from a folder of SKILLS_DIR holding a `SKILL.md`:

    ---
    name: invoice-reminder
    description: Draft a polite reminder for an unpaid invoice.
    tools: [time.now]
    ---
    The instructions, in Markdown.

Only the names and the descriptions of the skills granted to an agent enter its prompt, as a
short index; the body is read by the agent's `load_skill` tool, which refuses a skill not
granted to that agent. Every change of the description or the body is a new version, kept in
`skill_versions`.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import (
    ConflictError,
    InvalidInputError,
    NotFoundError,
    record_admin_event,
)
from app.db.models import Agent, Skill, SkillVersion

NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
MAX_DESCRIPTION = 120
MAX_BODY = 20_000
MAX_TOOLS = 50
LOAD_TOOL = "load_skill"
INDEX_HEADER = "Skills you can use (call load_skill with the name to read one before using it):"


class SkillNotFoundError(NotFoundError):
    pass


def _clean(name=None, description=None, body=None, tools=None) -> dict:
    out = {}
    if name is not None:
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise InvalidInputError("A skill name is lowercase letters, digits and -, up to 40")
        out["name"] = name
    if description is not None:
        if not isinstance(description, str) or not description.strip():
            raise InvalidInputError("A skill needs a description")
        text = " ".join(description.split())
        if len(text) > MAX_DESCRIPTION:
            raise InvalidInputError(f"The description is at most {MAX_DESCRIPTION} characters")
        out["description"] = text
    if body is not None:
        if not isinstance(body, str) or not body.strip() or len(body) > MAX_BODY:
            raise InvalidInputError(f"The body is 1 to {MAX_BODY} characters")
        out["body"] = body.strip()
    if tools is not None:
        if (
            not isinstance(tools, list)
            or len(tools) > MAX_TOOLS
            or not all(isinstance(t, str) and 0 < len(t) <= 200 for t in tools)
        ):
            raise InvalidInputError(f"tools is a list of at most {MAX_TOOLS} tool names")
        out["tools"] = tools
    return out


async def list_skills(session: AsyncSession) -> list[Skill]:
    return list((await session.execute(select(Skill).order_by(Skill.name))).scalars())


async def get_skill(session: AsyncSession, name: str) -> Skill:
    skill = (await session.execute(select(Skill).where(Skill.name == name))).scalar_one_or_none()
    if skill is None:
        raise SkillNotFoundError(f"No skill named {name!r}")
    return skill


async def versions(session: AsyncSession, name: str) -> list[SkillVersion]:
    skill = await get_skill(session, name)
    stmt = select(SkillVersion).where(SkillVersion.skill_id == skill.id)
    return list((await session.execute(stmt.order_by(SkillVersion.version))).scalars())


async def create_skill(
    session: AsyncSession,
    *,
    name: str,
    description: str,
    body: str,
    tools: list | None = None,
    source: str = "api",
    self_service: bool = False,
    actor: str,
) -> Skill:
    fields = _clean(name=name, description=description, body=body, tools=tools or [])
    if not isinstance(self_service, bool):
        raise InvalidInputError("self_service is a boolean")
    exists = await session.execute(select(Skill.id).where(Skill.name == fields["name"]))
    if exists.scalar_one_or_none() is not None:
        raise ConflictError(f"A skill named {name!r} already exists")
    skill = Skill(**fields, version=1, source=source, self_service=self_service)
    session.add(skill)
    await session.flush()
    session.add(SkillVersion(skill_id=skill.id, version=1, description=skill.description,
                             body=skill.body))  # fmt: skip
    await record_admin_event(
        session, actor=actor, action="skill.create", target_type="skill", target_id=skill.id,
        details={"name": skill.name, "version": 1, "source": source, "self_service": self_service},
    )  # fmt: skip
    await session.flush()
    return skill


async def update_skill(
    session: AsyncSession, name: str, *, actor: str, description=None, body=None, tools=None,
    source: str | None = None, self_service: bool | None = None,
) -> Skill:  # fmt: skip
    skill = await get_skill(session, name)
    fields = _clean(description=description, body=body, tools=tools)
    if self_service is not None:
        if not isinstance(self_service, bool):
            raise InvalidInputError("self_service is a boolean")
        fields["self_service"] = self_service
    new_text = any(k in fields and fields[k] != getattr(skill, k) for k in ("description", "body"))
    for key, value in fields.items():
        setattr(skill, key, value)
    if source is not None:
        skill.source = source
    if new_text:
        skill.version += 1
        session.add(SkillVersion(skill_id=skill.id, version=skill.version,
                                 description=skill.description, body=skill.body))  # fmt: skip
    skill.updated_at = datetime.now(UTC)
    await record_admin_event(
        session, actor=actor, action="skill.update", target_type="skill", target_id=skill.id,
        details={"name": skill.name, "version": skill.version, "fields": sorted(fields)},
    )  # fmt: skip
    await session.flush()
    return skill


async def delete_skill(session: AsyncSession, name: str, *, actor: str) -> None:
    skill = await get_skill(session, name)
    for agent in (await session.execute(select(Agent))).scalars():
        if name in (agent.skills or []):
            agent.skills = [s for s in agent.skills if s != name]
    await session.delete(skill)
    await record_admin_event(
        session, actor=actor, action="skill.delete", target_type="skill", target_id=skill.id,
        details={"name": name},
    )  # fmt: skip
    await session.flush()


async def set_agent_skills(
    session: AsyncSession, agent_id: int, names: list, *, actor: str
) -> list[str]:
    agent = await session.get(Agent, agent_id)
    if agent is None:
        raise NotFoundError(f"No agent {agent_id}")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise InvalidInputError("skills is a list of skill names")
    known = {s.name for s in await list_skills(session)}
    unknown = sorted(set(names) - known)
    if unknown:
        raise InvalidInputError(f"Unknown skill(s): {', '.join(unknown)}")
    agent.skills = sorted(set(names))
    await record_admin_event(
        session, actor=actor, action="agent.skills", target_type="agent", target_id=agent_id,
        details={"skills": agent.skills},
    )  # fmt: skip
    await session.flush()
    return agent.skills


# --- import of a SKILL.md folder ---


def parse_skill_md(text: str) -> dict:
    """The front matter (name, description, tools) and the body of a SKILL.md."""
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", text, re.S)
    if not match:
        raise InvalidInputError("A SKILL.md starts with a --- front matter --- block")
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise InvalidInputError(f"The front matter is not valid YAML: {exc}") from None
    if not isinstance(meta, dict):
        raise InvalidInputError("The front matter is a mapping of name, description, tools")
    return {
        "name": meta.get("name"),
        "description": meta.get("description"),
        "tools": meta.get("tools") or [],
        "body": match.group(2),
    }


def _folder(skills_dir: str, folder: str) -> Path:
    if not NAME.fullmatch(folder or ""):
        raise InvalidInputError("folder is a folder name of SKILLS_DIR, not a path")
    base = Path(skills_dir).resolve()
    path = base / folder / "SKILL.md"
    if not path.is_file() or path.is_symlink() or path.resolve().parent.parent != base:
        raise SkillNotFoundError(f"No {folder}/SKILL.md in SKILLS_DIR")
    return path


async def import_folder(
    session: AsyncSession, skills_dir: str, folder: str, *, actor: str
) -> Skill:
    """Create or update (a new version) the skill of `SKILLS_DIR/<folder>/SKILL.md`."""
    path = _folder(skills_dir, folder)
    meta = parse_skill_md(path.read_text(encoding="utf-8"))
    if meta["name"] != folder:
        raise InvalidInputError(f"The skill's name ({meta['name']!r}) is not its folder's")
    source = f"folder:{folder}"
    exists = await session.execute(select(Skill.id).where(Skill.name == folder))
    if exists.scalar_one_or_none() is None:
        return await create_skill(session, source=source, actor=actor, **meta)
    return await update_skill(
        session, folder, actor=actor, description=meta["description"], body=meta["body"],
        tools=meta["tools"], source=source,
    )  # fmt: skip


# --- what an agent sees ---


def index(skills: list[tuple[str, str]]) -> str:
    """The lines an agent's prompt gets for its granted skills: name and description only."""
    if not skills:
        return ""
    return "\n".join([INDEX_HEADER, *(f"- {name}: {text}" for name, text in skills)])


def tool_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": LOAD_TOOL,
            "description": "Read the full instructions of one of your skills, by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "the skill's name"}},
                "required": ["name"],
            },
        },
    }


async def granted(session: AsyncSession, agent: Agent) -> list[tuple[str, str]]:
    names = list(agent.skills or [])
    if not names:
        return []
    rows = await session.execute(
        select(Skill.name, Skill.description).where(Skill.name.in_(names)).order_by(Skill.name)
    )
    return [(name, description) for name, description in rows.all()]


def make_executor(agent_id: int):
    """`executor(name, raw_arguments)` of load_skill for one agent. Never raises: a refusal
    is the tool's text. The grant is read again at each call."""
    from app.db.session import session_scope

    async def run(_name: str, raw_arguments: str) -> str:
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            return "error: invalid arguments: not JSON"
        wanted = arguments.get("name") if isinstance(arguments, dict) else None
        async with session_scope() as session:
            agent = await session.get(Agent, agent_id)
            if agent is None or not isinstance(wanted, str) or wanted not in (agent.skills or []):
                return f"error: no skill named {wanted!r} is available to you"
            skill = (
                await session.execute(select(Skill).where(Skill.name == wanted))
            ).scalar_one_or_none()
            if skill is None:
                return f"error: no skill named {wanted!r} is available to you"
            return f"# Skill {skill.name} (version {skill.version})\n{skill.body}"

    return run
