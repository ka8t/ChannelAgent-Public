"""Agent templates: what the agent builder starts from, as data an administrator writes.

Reading needs the read scope, like the skills; every change needs the admin scope and is an
admin event, and every change of a template is a new version. A template never widens a right:
its tools are attached only when the user may attach them (app/builder.py).
"""

from datetime import datetime

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import agent_templates as service
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.scopes import Scope, require
from app.db.models import AgentTemplate

router = APIRouter()
NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,39}$"


class TemplateFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str | None = Field(
        default=None, min_length=1, max_length=service.MAX_DESCRIPTION,
        description="what it is for: the builder picks by it",
    )  # fmt: skip
    guidance: str | None = Field(
        default=None, max_length=service.MAX_GUIDANCE,
        description="instructions for the builder's model",
    )  # fmt: skip
    agent_instructions: str | None = Field(
        default=None, max_length=service.MAX_INSTRUCTIONS,
        description="added at the end of the agent's own instructions",
    )  # fmt: skip
    memory_mode: str | None = Field(
        default=None, description="fixed memory mode (off, ondemand, always, search) or null"
    )
    tools: list[str] | None = Field(
        default=None, max_length=service.MAX_TOOLS,
        description="tool names after mcp__<server>__, attached when the user may attach them",
    )  # fmt: skip
    needs_schedule: StrictBool | None = None
    questions: dict[str, str] | None = Field(
        default=None, description="the only questions asked, by purpose, schedule, task_prompt"
    )
    enabled: StrictBool | None = None


class TemplateIn(TemplateFields):
    name: str = Field(pattern=NAME_PATTERN, description="lowercase letters, digits and -")
    description: str = Field(min_length=1, max_length=service.MAX_DESCRIPTION)


class TemplateOut(BaseModel):
    name: str
    description: str
    guidance: str
    agent_instructions: str
    memory_mode: str | None
    tools: list[str]
    needs_schedule: bool
    questions: dict[str, str]
    enabled: bool
    version: int
    source: str
    updated_at: datetime


class TemplateVersionOut(BaseModel):
    version: int
    data: dict
    created_at: datetime


class StartersOut(BaseModel):
    created: list[str]
    kept: list[str]


def _out(template: AgentTemplate) -> TemplateOut:
    return TemplateOut(
        **service.snapshot(template), version=template.version, source=template.source,
        updated_at=template.updated_at,
    )  # fmt: skip


@router.get(
    "/agent-templates",
    dependencies=[require(Scope.READ)],
    response_model=list[TemplateOut],
    tags=["agent-templates"],
    responses=error_responses(),
)
async def list_agent_templates(
    session: AsyncSession = Depends(get_db_session),
) -> list[TemplateOut]:
    """Every agent template, by name."""
    return [_out(t) for t in await service.list_templates(session)]


@router.post(
    "/agent-templates",
    dependencies=[require(Scope.ADMIN)],
    response_model=TemplateOut,
    status_code=status.HTTP_201_CREATED,
    tags=["agent-templates"],
    responses=error_responses(409),
)
async def create_agent_template(
    body: TemplateIn, session: AsyncSession = Depends(get_db_session)
) -> TemplateOut:
    """Create an agent template (version 1). 409 when the name is taken."""
    fields = body.model_dump(exclude_unset=True)
    template = await service.create_template(session, fields, actor=current_actor())
    await session.commit()
    return _out(template)


@router.post(
    "/agent-templates/starters",
    dependencies=[require(Scope.ADMIN)],
    response_model=StartersOut,
    tags=["agent-templates"],
    responses=error_responses(409),
)
async def install_starter_templates(
    session: AsyncSession = Depends(get_db_session),
) -> StartersOut:
    """Create the starter templates that are missing (news digest, web page watch, reminder,
    daily summary); one already present is left as it is."""
    result = await service.install_starters(session, actor=current_actor())
    await session.commit()
    return StartersOut(**result)


@router.get(
    "/agent-templates/{name}",
    dependencies=[require(Scope.READ)],
    response_model=TemplateOut,
    tags=["agent-templates"],
    responses=error_responses(404),
)
async def get_agent_template(
    name: str, session: AsyncSession = Depends(get_db_session)
) -> TemplateOut:
    """One agent template."""
    return _out(await service.get_template(session, name))


@router.get(
    "/agent-templates/{name}/versions",
    dependencies=[require(Scope.READ)],
    response_model=list[TemplateVersionOut],
    tags=["agent-templates"],
    responses=error_responses(404),
)
async def list_agent_template_versions(
    name: str, session: AsyncSession = Depends(get_db_session)
) -> list[TemplateVersionOut]:
    """Every version of an agent template, oldest first, each the whole template."""
    return [
        TemplateVersionOut(version=v.version, data=v.data, created_at=v.created_at)
        for v in await service.versions(session, name)
    ]


@router.patch(
    "/agent-templates/{name}",
    dependencies=[require(Scope.ADMIN)],
    response_model=TemplateOut,
    tags=["agent-templates"],
    responses=error_responses(404, 409),
)
async def update_agent_template(
    name: str, body: TemplateFields, session: AsyncSession = Depends(get_db_session)
) -> TemplateOut:
    """Change an agent template; a real change is a new version, the old one is kept."""
    fields = body.model_dump(exclude_unset=True)
    template = await service.update_template(session, name, fields, actor=current_actor())
    await session.commit()
    return _out(template)


@router.delete(
    "/agent-templates/{name}",
    dependencies=[require(Scope.ADMIN)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["agent-templates"],
    responses=error_responses(404, 409),
)
async def delete_agent_template(name: str, session: AsyncSession = Depends(get_db_session)) -> None:
    """Delete an agent template and its versions. Agents made from it are not changed."""
    await service.delete_template(session, name, actor=current_actor())
    await session.commit()
