"""Scheduled tasks: every user's tasks, the global pause switch, and "run now".
A task's prompt is text a user wrote, so every route needs the admin scope, like the logs;
each change is an admin event (without the prompt). A user manages their own tasks with the
`/task` command (app/channels/dispatch.py), which calls the same functions (app/tasks.py).
"""

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from sqlalchemy.ext.asyncio import AsyncSession

from app import schedule_phrase
from app import tasks as service
from app.admin.jobs import Job, JobError, registry
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.routes import LIMIT, OFFSET, _page
from app.api.schemas import JobOut
from app.api.scopes import Scope, require
from app.db.models import ChannelIdentity, ScheduledTask, User

router = APIRouter()


class TaskIn(BaseModel):
    """`kind` is cron, every or daily; `expr` its schedule ("0 8 * * 1-5", "2h", "08:30"),
    read in the user's timezone. Without `channel_identity_id`, the user's only Telegram or
    email identity; without `agent_id`, the agent that identity talks to."""

    model_config = ConfigDict(extra="forbid")
    user_id: int
    prompt: str = Field(min_length=1, max_length=service.MAX_PROMPT)
    kind: str = Field(max_length=16)
    expr: str = Field(min_length=1, max_length=service.MAX_EXPR)
    agent_id: int | None = None
    channel_identity_id: int | None = None
    enabled: StrictBool = True
    standing_tools: list[str] = Field(
        default_factory=list,
        max_length=service.MAX_STANDING_TOOLS,
        description="MCP tools of the agent its scheduled runs may use without asking",
    )


class TaskUpdate(BaseModel):
    """A field left out keeps its value."""

    model_config = ConfigDict(extra="forbid")
    prompt: str | None = Field(default=None, min_length=1, max_length=service.MAX_PROMPT)
    kind: str | None = Field(default=None, max_length=16)
    expr: str | None = Field(default=None, min_length=1, max_length=service.MAX_EXPR)
    agent_id: int | None = None
    channel_identity_id: int | None = None
    enabled: StrictBool | None = None
    standing_tools: list[str] | None = Field(default=None, max_length=service.MAX_STANDING_TOOLS)


class TaskOut(BaseModel):
    id: int
    user_id: int
    agent_id: int
    channel_identity_id: int
    channel: str | None
    prompt: str
    kind: str
    expr: str
    timezone: str
    enabled: bool
    next_run_at: datetime | None
    last_run_at: datetime | None
    last_status: str | None
    last_error: str | None
    run_count: int
    created_at: datetime
    # Standing approvals that cover the tool's approved definition now, and those given
    # for a definition an administrator has since replaced (refused until approved again).
    standing_tools: list[str] = []
    standing_stale: list[str] = []


class SchedulePhraseIn(BaseModel):
    """A schedule in words, read in `timezone` (an IANA name; UTC when left out)."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2000)
    timezone: str | None = Field(default=None, max_length=64)


class SchedulePhraseOut(BaseModel):
    kind: str
    expr: str
    prompt: str | None
    timezone: str
    next_runs: list[datetime]


class PauseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paused: StrictBool


class PauseOut(BaseModel):
    paused: bool


async def _out(session: AsyncSession, task: ScheduledTask) -> TaskOut:
    user = await session.get(User, task.user_id)
    identity = await session.get(ChannelIdentity, task.channel_identity_id)
    approved = await service.approved_hashes(session, list(task.standing_tools or {}))
    standing = service.standing_state(task, approved)
    return TaskOut(
        id=task.id,
        user_id=task.user_id,
        agent_id=task.agent_id,
        channel_identity_id=task.channel_identity_id,
        channel=identity.channel.value if identity is not None else None,
        prompt=task.prompt,
        kind=task.kind,
        expr=task.expr,
        timezone=(user.timezone if user is not None else None) or "UTC",
        enabled=task.enabled,
        next_run_at=service._as_utc(task.next_run_at),
        last_run_at=service._as_utc(task.last_run_at),
        last_status=task.last_status,
        last_error=task.last_error,
        run_count=task.run_count,
        created_at=service._as_utc(task.created_at),
        standing_tools=standing["current"],
        standing_stale=standing["stale"],
    )


@router.post(
    "/schedules/parse",
    dependencies=[require(Scope.OPERATE)],
    response_model=SchedulePhraseOut,
    tags=["tasks"],
    responses=error_responses(409),
)
async def parse_schedule(body: SchedulePhraseIn) -> SchedulePhraseOut:
    """Read a schedule written in words ("every Thursday at 9") as cron, every or daily.

    The model proposes and the task parser decides, the same code as `/task add`: 422
    with the reason when the words hold no schedule, or one that never falls due. Nothing is
    written; the next 3 run times are returned in the timezone."""
    timezone = service.check_timezone(body.timezone) if body.timezone else None
    parsed = await schedule_phrase.parse(body.text, timezone)
    return SchedulePhraseOut(
        kind=parsed["kind"], expr=parsed["expr"], prompt=parsed["prompt"],
        timezone=timezone or "UTC", next_runs=parsed["next_runs"],
    )  # fmt: skip


# Declared before /tasks/{task_id}: "pause" is not a task id.
@router.get(
    "/tasks/pause",
    dependencies=[require(Scope.ADMIN)],
    response_model=PauseOut,
    tags=["tasks"],
    responses=error_responses(),
)
async def get_pause(session: AsyncSession = Depends(get_db_session)) -> PauseOut:
    """Whether the scheduled tasks are paused (then none runs)."""
    row = await service.get_config(session)
    await session.commit()
    return PauseOut(paused=row.paused)


@router.post(
    "/tasks/pause",
    dependencies=[require(Scope.ADMIN)],
    response_model=PauseOut,
    tags=["tasks"],
    responses=error_responses(409),
)
async def set_pause(body: PauseIn, session: AsyncSession = Depends(get_db_session)) -> PauseOut:
    """Pause every scheduled task (`paused: true`), or resume them. While paused, a task
    that falls due is moved to its next time, not run late."""
    row = await service.set_paused(session, body.paused, actor=current_actor())
    await session.commit()
    return PauseOut(paused=row.paused)


@router.get(
    "/tasks",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[TaskOut],
    tags=["tasks"],
    responses=error_responses(),
)
async def list_tasks(
    response: Response,
    user_id: int | None = Query(default=None, description="Only this user's tasks."),
    limit: int = LIMIT,
    offset: int = OFFSET,
    session: AsyncSession = Depends(get_db_session),
) -> list[TaskOut]:
    """Every user's scheduled tasks, or one user's, oldest first."""
    rows = await service.list_tasks(session, user_id)
    return [await _out(session, t) for t in _page(response, rows, limit, offset)]


@router.post(
    "/tasks",
    dependencies=[require(Scope.ADMIN)],
    response_model=TaskOut,
    status_code=status.HTTP_201_CREATED,
    tags=["tasks"],
    responses=error_responses(404, 409),
)
async def create_task(body: TaskIn, session: AsyncSession = Depends(get_db_session)) -> TaskOut:
    """Create a scheduled task for a user. A schedule that does not parse, or never falls
    due, is refused (422)."""
    task = await service.create_task(
        session,
        user_id=body.user_id,
        prompt=body.prompt,
        kind=body.kind,
        expr=body.expr,
        agent_id=body.agent_id,
        channel_identity_id=body.channel_identity_id,
        enabled=body.enabled,
        standing_tools=body.standing_tools,
        actor=current_actor(),
    )
    await session.commit()
    return await _out(session, task)


@router.get(
    "/tasks/{task_id}",
    dependencies=[require(Scope.ADMIN)],
    response_model=TaskOut,
    tags=["tasks"],
    responses=error_responses(404),
)
async def get_task(task_id: int, session: AsyncSession = Depends(get_db_session)) -> TaskOut:
    """One scheduled task."""
    return await _out(session, await service.get_task(session, task_id))


@router.patch(
    "/tasks/{task_id}",
    dependencies=[require(Scope.ADMIN)],
    response_model=TaskOut,
    tags=["tasks"],
    responses=error_responses(404, 409),
)
async def update_task(
    task_id: int, body: TaskUpdate, session: AsyncSession = Depends(get_db_session)
) -> TaskOut:
    """Change a task: its prompt, schedule, agent, delivery identity, or stop it
    (`enabled: false`) and start it again."""
    task = await service.update_task(
        session, task_id, body.model_dump(exclude_unset=True), actor=current_actor()
    )
    await session.commit()
    return await _out(session, task)


@router.delete(
    "/tasks/{task_id}",
    dependencies=[require(Scope.ADMIN)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["tasks"],
    responses=error_responses(404, 409),
)
async def delete_task(task_id: int, session: AsyncSession = Depends(get_db_session)) -> None:
    """Delete a task and its conversation."""
    await service.delete_task(session, task_id, actor=current_actor())
    await session.commit()


@router.post(
    "/tasks/{task_id}/run",
    dependencies=[require(Scope.ADMIN)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["tasks"],
    responses=error_responses(404, 409),
)
async def run_task(task_id: int, session: AsyncSession = Depends(get_db_session)) -> Job:
    """Run a task now, whatever its schedule (its next time does not move), even while
    paused. Returns the job; the outcome is in its result and in the task."""
    task = await service.get_task(session, task_id)
    actor = current_actor()
    await service.record_admin_event(
        session, actor=actor, action="task.run", target_type="task", target_id=task.id,
        details={"user_id": task.user_id},
    )  # fmt: skip
    await session.commit()

    async def run(job: Job) -> dict:
        job.update(0.1, "Running the task's turn")
        result = await service.execute(task_id, trigger="manual")
        if result["status"] == "refused":
            raise JobError("The user, the agent or the identity's permission is not active")
        return result

    return registry.start("task-run", run)
