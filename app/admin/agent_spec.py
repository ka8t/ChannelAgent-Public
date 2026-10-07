"""An agent a user creates themselves (agent builder.

An agent specification is the only thing the agent builder produces and the only thing this
module accepts: a name, a purpose, the per-agent settings, skills, and optionally a
schedule with the prompt a task sends and the identity it delivers on. It is checked
against the rights the user already has, then the agent and its task are created in one
transaction with one admin event.

The check never widens a right. A tool must be covered by one of the user's grants for every
agent, since the agent does not exist yet; a skill must be marked self-service by an
administrator; the delivery identity must be the user's own. A specification asking for more is
refused with the reason, and nothing here creates a grant, approves a definition or grants a
skill that is not self-service. Administrators keep their own routes (`POST
/users/{id}/agents`), which this module does not limit.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import tasks
from app.admin import service
from app.admin.service import InvalidInputError, record_admin_event
from app.config import get_settings
from app.db.models import Agent, McpGrant, ScheduledTask, Skill
from app.mcp.catalogue import NAME_PREFIX
from app.mcp.policy import is_granted

MAX_PURPOSE = 200
SPEC_FIELDS = (
    "name",
    "purpose",
    "system_prompt",
    "model",
    "memory_mode",
    "tools",
    "skills",
    "task_prompt",
    "schedule_kind",
    "schedule_expr",
    "delivery_identity_id",
    "unattended",
)
SCHEDULE_FIELDS = ("task_prompt", "schedule_kind", "schedule_expr")


class SpecRefusedError(InvalidInputError):
    """Every reason a specification is refused, not only the first."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("The agent specification is refused: " + "; ".join(errors))


def _split_tool(name: str) -> tuple[str, str] | None:
    if not name.startswith(NAME_PREFIX):
        return None
    server, sep, tool = name[len(NAME_PREFIX) :].partition("__")
    return (server, tool) if sep and server and tool else None


def _clean_purpose(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise InvalidInputError("purpose is text")
    value = " ".join(value.split())
    if len(value) > MAX_PURPOSE:
        raise InvalidInputError(f"purpose is at most {MAX_PURPOSE} characters")
    return value or None


async def _agent_wide_grants(session: AsyncSession, user_id: int) -> frozenset:
    """The grants that cover every agent of the user, the only ones a new agent gets."""
    rows = await session.execute(
        select(McpGrant.server_name, McpGrant.tool_name).where(
            McpGrant.user_id == user_id, McpGrant.agent_id.is_(None)
        )
    )
    return frozenset((server, tool) for server, tool in rows.all())


async def _count(session: AsyncSession, model, user_id: int) -> int:
    stmt = select(func.count()).select_from(model).where(model.user_id == user_id)
    return (await session.execute(stmt)).scalar_one()


async def check_spec(
    session: AsyncSession,
    user_id: int,
    spec: dict,
    now: datetime | None = None,
    editing: Agent | None = None,
) -> tuple[dict, list[str]]:
    """The cleaned specification and every reason it is refused (empty when it is valid).
    Writes nothing. Raises UserNotFoundError for an unknown user.

    `editing`Checks a change to that agent of the user: its own name is not taken,
    the agent cap does not apply, grants bound to it count, and the tools and skills it already
    has stay allowed (an administrator may have set them); only what is added is checked."""
    user = await service._require_user(session, user_id)
    kept_tools = set(editing.tools or []) if editing is not None else set()
    kept_skills = set(editing.skills or []) if editing is not None else set()
    errors: list[str] = []
    cleaned: dict = {}

    unknown = sorted(set(spec) - set(SPEC_FIELDS))
    if unknown:
        errors.append(f"unknown fields: {', '.join(unknown)}")

    def clean(name: str, cleaner, *args):
        try:
            cleaned[name] = cleaner(spec.get(name), *args)
        except InvalidInputError as exc:
            errors.append(f"{name}: {exc}")

    try:
        cleaned["name"] = service._clean_agent_name(spec.get("name") or "")
    except InvalidInputError as exc:
        errors.append(f"name: {exc}")
    else:
        named = await service._agent_named(session, user_id, cleaned["name"])
        if named is not None and (editing is None or named.id != editing.id):
            errors.append(f"name: you already have an agent named {cleaned['name']!r}")
    clean("purpose", _clean_purpose)
    clean("system_prompt", service._clean_system_prompt)
    clean("model", service._clean_model)
    cleaned["memory_mode"] = spec.get("memory_mode") or "off"
    try:
        service._clean_memory_mode(cleaned["memory_mode"])
    except InvalidInputError as exc:
        errors.append(f"memory_mode: {exc}")
    clean("tools", lambda v: service._clean_tools(v if v is not None else []))
    skills = spec.get("skills") if spec.get("skills") is not None else []
    if not isinstance(skills, list) or not all(isinstance(n, str) for n in skills):
        errors.append("skills: a list of skill names")
        skills = []
    cleaned["skills"] = sorted(set(skills))

    # Rights: tools covered by a grant for every agent of the user.
    grants = await _agent_wide_grants(session, user_id)
    if editing is not None:
        from app.admin.mcp import grants_for

        grants = grants | await grants_for(session, user_id, editing.id)
    for name in cleaned.get("tools", []):
        pair = _split_tool(name)
        if name in kept_tools:
            continue
        if pair is None:
            errors.append(f"tools: {name!r} is not an MCP tool name (mcp__<server>__<tool>)")
        elif not is_granted(grants, *pair):
            errors.append(
                f"tools: {name!r} is not granted to you for all your agents; ask an administrator"
            )

    # Rights: skills an administrator marked self-service.
    if set(cleaned["skills"]) - kept_skills:
        rows = await session.execute(
            select(Skill.name, Skill.self_service).where(Skill.name.in_(cleaned["skills"]))
        )
        known = dict(rows.all())
        for name in cleaned["skills"]:
            if name in kept_skills:
                continue
            if name not in known:
                errors.append(f"skills: no skill named {name!r}")
            elif not known[name]:
                errors.append(
                    f"skills: {name!r} is granted by an administrator only; ask an administrator"
                )

    # Optional schedule: the three fields together or none of them.
    given = [f for f in SCHEDULE_FIELDS if spec.get(f) not in (None, "")]
    if given and len(given) != len(SCHEDULE_FIELDS):
        missing = [f for f in SCHEDULE_FIELDS if f not in given]
        together = ", ".join(SCHEDULE_FIELDS)
        errors.append(f"schedule: {', '.join(missing)} missing (all of {together} or none)")
    elif given:
        try:
            cleaned["task_prompt"] = tasks._clean_prompt(spec["task_prompt"])
        except InvalidInputError as exc:
            errors.append(f"task_prompt: {exc}")
        try:
            kind, expr = tasks.check_schedule(spec["schedule_kind"], spec["schedule_expr"])
            # A spec is what a user asks for themselves (TASK_MIN_INTERVAL_MINUTES).
            tasks.check_self_service_interval(kind, expr)
            # Also refuses a schedule that never falls due (`0 0 31 2 *`).
            cleaned["next_run_at"] = tasks.next_time(
                kind, expr, user.timezone, now or datetime.now(UTC)
            )
            cleaned["schedule_kind"], cleaned["schedule_expr"] = kind, expr
        except InvalidInputError as exc:
            errors.append(f"schedule: {exc}")
        identity_id = spec.get("delivery_identity_id")
        if identity_id is not None and not isinstance(identity_id, int):
            errors.append("delivery_identity_id: an identity id")
        else:
            try:
                identity = await tasks._identity_for(session, user_id, identity_id)
                cleaned["delivery_identity_id"] = identity.id
            except (InvalidInputError, service.NotFoundError) as exc:
                errors.append(f"delivery_identity_id: {exc}")
        has_task = editing is not None and await first_task(session, editing.id) is not None
        count = await _count(session, ScheduledTask, user_id)
        if not has_task and count >= tasks.MAX_TASKS_PER_USER:
            errors.append(f"schedule: you already have {tasks.MAX_TASKS_PER_USER} tasks")
    elif spec.get("delivery_identity_id") is not None:
        errors.append("delivery_identity_id: only with a schedule")

    # Standing approval: tools of this agent its scheduled runs may use without asking.
    unattended = spec.get("unattended") if spec.get("unattended") is not None else []
    if not isinstance(unattended, list) or not all(isinstance(n, str) for n in unattended):
        errors.append("unattended: a list of tool names")
        unattended = []
    cleaned["unattended"] = sorted(set(unattended))
    if cleaned["unattended"] and not given:
        errors.append("unattended: only with a schedule")
    elif cleaned["unattended"]:
        approved = await tasks.approved_hashes(session, cleaned["unattended"])
        for name in cleaned["unattended"]:
            if name not in cleaned.get("tools", []):
                errors.append(f"unattended: {name!r} is not one of the agent's tools")
            elif approved.get(name) is None:
                errors.append(f"unattended: {name!r} has no approved definition")

    # Limit: agents a user may have when creating one themselves (SELF_SERVICE_MAX_AGENTS).
    cap = get_settings().self_service_max_agents
    if editing is not None:
        pass  # a change creates no agent
    elif cap == 0:
        errors.append("users cannot create agents themselves here (SELF_SERVICE_MAX_AGENTS=0)")
    elif await _count(session, Agent, user_id) >= cap:
        errors.append(f"you already have {cap} agents, the most you may create yourself")

    return cleaned, errors


async def create_agent_from_spec(
    session: AsyncSession, user_id: int, spec: dict, *, actor: str, now: datetime | None = None
) -> tuple[Agent, ScheduledTask | None]:
    """Create the agent, and its task when the specification has a schedule, in the caller's
    transaction, with one admin event that holds names and switches, never the free text."""
    cleaned, errors = await check_spec(session, user_id, spec, now)
    if errors:
        raise SpecRefusedError(errors)
    agent = Agent(
        user_id=user_id,
        name=cleaned["name"],
        purpose=cleaned["purpose"],
        system_prompt=cleaned["system_prompt"],
        model=cleaned["model"],
        memory_mode=cleaned["memory_mode"],
        tools=cleaned["tools"],
        skills=cleaned["skills"],
    )
    session.add(agent)
    await session.flush()
    task = None
    if "schedule_kind" in cleaned:
        kind, expr = cleaned["schedule_kind"], cleaned["schedule_expr"]
        task = ScheduledTask(
            user_id=user_id,
            agent_id=agent.id,
            channel_identity_id=cleaned["delivery_identity_id"],
            prompt=cleaned["task_prompt"],
            kind=kind,
            expr=expr,
            enabled=True,
            next_run_at=cleaned["next_run_at"],
            standing_tools=await tasks.standing_approval(session, agent, cleaned["unattended"]),
        )
        session.add(task)
        await session.flush()
    await record_admin_event(
        session,
        actor=actor,
        action="agent.create_from_spec",
        target_type="agent",
        target_id=agent.id,
        details={
            "name": agent.name,
            "owner_user_id": user_id,
            "model": agent.model,
            "memory_mode": agent.memory_mode,
            "tools": agent.tools,
            "skills": agent.skills,
            **({"task_id": task.id, "kind": task.kind, "expr": task.expr,
                "standing_tools": sorted(task.standing_tools)} if task else {}),  # fmt: skip
        },
    )
    return agent, task


# --- an agent the user already has ---


async def own_agent(session: AsyncSession, user_id: int, agent_id: int) -> Agent:
    """The user's agent, or AgentNotFoundError worded the same whether it does not exist or
    is another user's."""
    agent = await session.get(Agent, agent_id)
    if agent is None or agent.user_id != user_id:
        raise service.AgentNotFoundError(f"No agent {agent_id}")
    return agent


async def first_task(session: AsyncSession, agent_id: int) -> ScheduledTask | None:
    """The task a specification describes: the agent's first (a spec has one schedule)."""
    stmt = (
        select(ScheduledTask).where(ScheduledTask.agent_id == agent_id).order_by(ScheduledTask.id)
    )
    return (await session.execute(stmt)).scalars().first()


async def spec_of(session: AsyncSession, agent: Agent) -> dict:
    """The specification of an existing agent and of its first task."""
    task = await first_task(session, agent.id)
    spec = {
        "name": agent.name,
        "purpose": agent.purpose,
        "system_prompt": agent.system_prompt,
        "model": agent.model,
        "memory_mode": agent.memory_mode,
        "tools": list(agent.tools or []),
        "skills": list(agent.skills or []),
        "task_prompt": task.prompt if task else None,
        "schedule_kind": task.kind if task else None,
        "schedule_expr": task.expr if task else None,
        "delivery_identity_id": task.channel_identity_id if task else None,
        "unattended": sorted((task.standing_tools or {}) if task else []),
    }
    return spec


async def update_agent_from_spec(
    session: AsyncSession,
    user_id: int,
    agent_id: int,
    spec: dict,
    *,
    actor: str,
    now: datetime | None = None,
) -> tuple[Agent, ScheduledTask | None]:
    """Change one of the user's agents to `spec`, in the caller's transaction: its
    settings, and its first task (changed, created, or deleted when the spec has no schedule).
    Checked like a creation, within the rights the user has; one admin event without the free
    text. Returns the agent and its task after the change."""
    agent = await own_agent(session, user_id, agent_id)
    cleaned, errors = await check_spec(session, user_id, spec, now, editing=agent)
    if errors:
        raise SpecRefusedError(errors)
    before = await spec_of(session, agent)
    for field in ("name", "purpose", "system_prompt", "model", "memory_mode", "tools", "skills"):
        setattr(agent, field, cleaned[field])
    await session.flush()
    task = await first_task(session, agent.id)
    if "schedule_kind" in cleaned:
        standing = await tasks.standing_approval(session, agent, cleaned["unattended"])
        if task is None:
            task = ScheduledTask(user_id=user_id, agent_id=agent.id, enabled=True)
            session.add(task)
        task.prompt = cleaned["task_prompt"]
        task.kind, task.expr = cleaned["schedule_kind"], cleaned["schedule_expr"]
        task.channel_identity_id = cleaned["delivery_identity_id"]
        task.standing_tools = standing
        if task.enabled:
            task.next_run_at = cleaned["next_run_at"]
        await session.flush()
    elif task is not None:
        await tasks.delete_task(session, task.id, actor=actor, user_id=user_id)
        task = None
    after = await spec_of(session, agent)
    changed = sorted(k for k in after if after[k] != before.get(k))
    await record_admin_event(
        session, actor=actor, action="agent.update_from_spec", target_type="agent",
        target_id=agent.id,
        details={"owner_user_id": user_id, "fields": changed, "tools": agent.tools,
                 **({"task_id": task.id, "kind": task.kind, "expr": task.expr} if task else {})},
    )  # fmt: skip
    return agent, task
