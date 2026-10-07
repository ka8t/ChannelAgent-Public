"""Agents a user creates themselves (agent builder.

One operation, `app.admin.agent_spec.create_agent_from_spec`, used by the agent builder in
process and exposed here, so `start.sh` and the admin UI get it from the manifest. It checks the
specification against the rights the user already has (grants for every agent, self-service
skills, their own identities, SELF_SERVICE_MAX_AGENTS) and never widens them. The task prompt and
the purpose are user text, so both routes need the admin scope, like the task routes.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app import tasks
from app.admin import agent_spec, my_agents
from app.admin.jobs import Job, registry
from app.admin.service import InvalidInputError
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.routes import _agent_out
from app.api.schemas import AgentOut, JobOut, MemoryMode
from app.api.scopes import Principal, Scope, get_principal, require

router = APIRouter()


class AgentSpecIn(BaseModel):
    """An agent and, optionally, its schedule: `task_prompt`, `schedule_kind` (cron, every,
    daily) and `schedule_expr` together, delivered on `delivery_identity_id` (the user's only
    Telegram or email identity when left out)."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    purpose: str | None = Field(default=None, max_length=agent_spec.MAX_PURPOSE)
    system_prompt: str | None = Field(default=None, max_length=20000)
    model: str | None = Field(default=None, max_length=200)
    memory_mode: MemoryMode = "off"
    tools: list[str] = Field(
        default_factory=list, max_length=100, description="mcp__<server>__<tool> names"
    )
    skills: list[str] = Field(default_factory=list, max_length=100)
    task_prompt: str | None = Field(default=None, max_length=tasks.MAX_PROMPT)
    schedule_kind: str | None = Field(default=None, max_length=16)
    schedule_expr: str | None = Field(default=None, max_length=tasks.MAX_EXPR)
    delivery_identity_id: int | None = None
    unattended: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="tools of the agent its scheduled runs may use without asking",
    )


class AgentSpecCheckOut(BaseModel):
    valid: bool
    errors: list[str]


class AgentFromSpecOut(BaseModel):
    agent: AgentOut
    task_id: int | None


@router.post(
    "/users/{user_id}/agent-specs/check",
    dependencies=[require(Scope.ADMIN)],
    response_model=AgentSpecCheckOut,
    tags=["agents"],
    responses=error_responses(404, 409),
)
async def check_agent_spec(
    user_id: int, body: AgentSpecIn, session: AsyncSession = Depends(get_db_session)
) -> AgentSpecCheckOut:
    """Check an agent specification for a user, without creating anything.

    Returns every reason the specification would be refused."""
    _, errors = await agent_spec.check_spec(session, user_id, body.model_dump())
    await session.rollback()
    return AgentSpecCheckOut(valid=not errors, errors=errors)


@router.post(
    "/users/{user_id}/agent-specs",
    dependencies=[require(Scope.ADMIN)],
    response_model=AgentFromSpecOut,
    status_code=status.HTTP_201_CREATED,
    tags=["agents"],
    responses=error_responses(404, 409),
)
async def create_agent_from_spec(
    user_id: int,
    body: AgentSpecIn,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> AgentFromSpecOut:
    """Create an agent and its scheduled task, within the rights the user already has.

    Refused with 422 and every reason otherwise; nothing is written then."""
    agent, task = await agent_spec.create_agent_from_spec(
        session, user_id, body.model_dump(), actor=current_actor()
    )
    await session.commit()
    return AgentFromSpecOut(agent=_agent_out(agent, principal), task_id=task.id if task else None)


# --- a user's own agents ---

MINE = "/users/{user_id}/agents/{agent_id}"


class AgentTaskOverview(BaseModel):
    task_id: int
    kind: str
    expr: str
    enabled: bool
    next_run_at: datetime | None
    last_run_at: datetime | None
    last_status: str | None


class AgentOverview(BaseModel):
    agent_id: int
    name: str
    is_active: bool
    purpose: str | None
    tasks: list[AgentTaskOverview]


class PausedOut(BaseModel):
    agent_id: int
    changed_task_ids: list[int]


@router.get(
    "/users/{user_id}/agent-overview",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[AgentOverview],
    tags=["agents"],
    responses=error_responses(404),
)
async def agent_overview(
    user_id: int, session: AsyncSession = Depends(get_db_session)
) -> list[AgentOverview]:
    """A user's agents with their purpose and their tasks' schedule, next and last run."""
    return [AgentOverview(**row) for row in await my_agents.overview(session, user_id)]


@router.get(
    MINE + "/spec",
    dependencies=[require(Scope.ADMIN)],
    response_model=AgentSpecIn,
    tags=["agents"],
    responses=error_responses(404),
)
async def get_agent_spec(
    user_id: int, agent_id: int, session: AsyncSession = Depends(get_db_session)
) -> AgentSpecIn:
    """The specification of one of a user's agents and of its first task."""
    agent = await agent_spec.own_agent(session, user_id, agent_id)
    return AgentSpecIn(**await agent_spec.spec_of(session, agent))


@router.put(
    MINE + "/spec",
    dependencies=[require(Scope.ADMIN)],
    response_model=AgentFromSpecOut,
    tags=["agents"],
    responses=error_responses(404, 409),
)
async def update_agent_spec(
    user_id: int,
    agent_id: int,
    body: AgentSpecIn,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> AgentFromSpecOut:
    """Change one of a user's agents and its task to a specification, within the user's rights.

    Refused with 422 and every reason otherwise; nothing is written then."""
    agent, task = await agent_spec.update_agent_from_spec(
        session, user_id, agent_id, body.model_dump(), actor=current_actor()
    )
    await session.commit()
    return AgentFromSpecOut(agent=_agent_out(agent, principal), task_id=task.id if task else None)


async def _pause(user_id, agent_id, paused, session) -> PausedOut:
    changed = await my_agents.set_paused(session, user_id, agent_id, paused, actor=current_actor())
    await session.commit()
    return PausedOut(agent_id=agent_id, changed_task_ids=changed)


@router.post(
    MINE + "/pause",
    dependencies=[require(Scope.OPERATE)],
    response_model=PausedOut,
    tags=["agents"],
    responses=error_responses(404, 409),
)
async def pause_agent(
    user_id: int, agent_id: int, session: AsyncSession = Depends(get_db_session)
) -> PausedOut:
    """Stop the scheduled tasks of one of a user's agents (it still answers in chat)."""
    return await _pause(user_id, agent_id, True, session)


@router.post(
    MINE + "/resume",
    dependencies=[require(Scope.OPERATE)],
    response_model=PausedOut,
    tags=["agents"],
    responses=error_responses(404, 409),
)
async def resume_agent(
    user_id: int, agent_id: int, session: AsyncSession = Depends(get_db_session)
) -> PausedOut:
    """Start again the scheduled tasks of one of a user's agents."""
    return await _pause(user_id, agent_id, False, session)


@router.post(
    MINE + "/retire",
    dependencies=[require(Scope.OPERATE)],
    response_model=AgentOut,
    tags=["agents"],
    responses=error_responses(404, 409),
)
async def retire_agent(
    user_id: int,
    agent_id: int,
    session: AsyncSession = Depends(get_db_session),
    principal: Principal = Depends(get_principal),
) -> AgentOut:
    """Delete the tasks of one of a user's agents and disable it (kept for the audit trail)."""
    agent = await my_agents.retire(session, user_id, agent_id, actor=current_actor())
    await session.commit()
    return _agent_out(agent, principal)


@router.post(
    MINE + "/run",
    dependencies=[require(Scope.ADMIN)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["agents"],
    responses=error_responses(404, 409),
)
async def run_agent(
    user_id: int, agent_id: int, session: AsyncSession = Depends(get_db_session)
) -> Job:
    """Run every task of one of a user's agents now, whatever its schedule. Returns the job."""
    ids = await my_agents.task_ids(session, user_id, agent_id)
    if not ids:
        raise InvalidInputError(f"Agent {agent_id} has no scheduled task to run")

    async def run(job: Job) -> dict:
        results = []
        for index, task_id in enumerate(ids):
            job.update(index / len(ids), f"Running task {task_id}")
            results.append(await tasks.execute(task_id, trigger="manual"))
        return {"tasks": results}

    return registry.start("agent-run", run)
