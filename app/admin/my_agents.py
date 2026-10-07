"""A user's own agents: the overview, pause, resume, retire. One implementation for the
channel commands (`/myagents`, app/channels/dispatch.py) and the Admin API
(app/api/agent_spec_routes.py). Every operation takes the user's id and refuses another user's
agent with the same words as a missing one ("No agent N"); it goes through the task operations
of app/tasks.py, which carry the same ownership check.

- pause and resume stop and start the agent's scheduled tasks; the agent still answers in chat;
- retire deletes its tasks and disables the agent (an agent is never deleted: the audit trail
  refers to it; an administrator purges a user to delete everything).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import tasks
from app.admin.agent_spec import own_agent
from app.admin.service import list_agents, record_admin_event
from app.db.models import Agent, ScheduledTask


async def _tasks_of(session: AsyncSession, agent_id: int) -> list[ScheduledTask]:
    stmt = (
        select(ScheduledTask).where(ScheduledTask.agent_id == agent_id).order_by(ScheduledTask.id)
    )
    return list((await session.execute(stmt)).scalars())


async def overview(session: AsyncSession, user_id: int) -> list[dict]:
    """Each of the user's agents with its purpose and its tasks' schedule, next and last run."""
    out = []
    for agent in await list_agents(session, user_id):
        out.append(
            {
                "agent_id": agent.id,
                "name": agent.name,
                "is_active": agent.is_active,
                "purpose": agent.purpose,
                "tasks": [
                    {
                        "task_id": t.id,
                        "kind": t.kind,
                        "expr": t.expr,
                        "enabled": t.enabled,
                        "next_run_at": tasks._as_utc(t.next_run_at),
                        "last_run_at": tasks._as_utc(t.last_run_at),
                        "last_status": t.last_status,
                    }
                    for t in await _tasks_of(session, agent.id)
                ],
            }
        )
    return out


async def set_paused(
    session: AsyncSession, user_id: int, agent_id: int, paused: bool, *, actor: str
) -> list[int]:
    """Stop (or start again) every scheduled task of the user's agent; returns the task ids
    whose state changed."""
    agent = await own_agent(session, user_id, agent_id)
    changed = []
    for task in await _tasks_of(session, agent.id):
        if task.enabled == paused:
            await tasks.update_task(session, task.id, {"enabled": not paused}, actor=actor,
                                    user_id=user_id)  # fmt: skip
            changed.append(task.id)
    return changed


async def retire(session: AsyncSession, user_id: int, agent_id: int, *, actor: str) -> Agent:
    """Delete the agent's tasks and disable it (never deleted: the audit trail refers to it)."""
    agent = await own_agent(session, user_id, agent_id)
    removed = []
    for task in await _tasks_of(session, agent.id):
        await tasks.delete_task(session, task.id, actor=actor, user_id=user_id)
        removed.append(task.id)
    agent.is_active = False
    await session.flush()
    await record_admin_event(
        session, actor=actor, action="agent.retire", target_type="agent", target_id=agent.id,
        details={"owner_user_id": user_id, "tasks_deleted": removed},
    )  # fmt: skip
    return agent


async def task_ids(session: AsyncSession, user_id: int, agent_id: int) -> list[int]:
    """The ids of the user's agent's tasks (to run them now)."""
    agent = await own_agent(session, user_id, agent_id)
    return [t.id for t in await _tasks_of(session, agent.id)]
