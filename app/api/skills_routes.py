"""Local skills (decision M3): create, read, change, version, import and grant to agents.

A skill is instructions an administrator writes; reading one needs the read scope, like the MCP
server list, and every change needs the admin scope and is an admin event. The body enters an
agent's conversation only when the agent calls `load_skill` for a skill granted to it.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import skills as service
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.scopes import Scope, require
from app.config import get_settings
from app.db.models import Skill

router = APIRouter()
NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,39}$"


class SkillIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=NAME_PATTERN, description="lowercase letters, digits and -")
    description: str = Field(min_length=1, max_length=service.MAX_DESCRIPTION)
    body: str = Field(min_length=1, max_length=service.MAX_BODY, description="the instructions")
    tools: list[str] = Field(default_factory=list, max_length=service.MAX_TOOLS)
    self_service: StrictBool = Field(
        default=False, description="a user may attach it to an agent they create"
    )


class SkillUpdate(BaseModel):
    """A field left out keeps its value; a new description or body is a new version."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = Field(default=None, min_length=1, max_length=service.MAX_DESCRIPTION)
    body: str | None = Field(default=None, min_length=1, max_length=service.MAX_BODY)
    tools: list[str] | None = Field(default=None, max_length=service.MAX_TOOLS)
    self_service: StrictBool | None = None


class SkillImportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    folder: str = Field(pattern=NAME_PATTERN, description="a folder of SKILLS_DIR with a SKILL.md")


class AgentSkillsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skills: list[str] = Field(max_length=500, description="the names of the skills it may load")


class SkillSummary(BaseModel):
    name: str
    description: str
    tools: list[str]
    version: int
    source: str
    self_service: bool
    updated_at: datetime


class SkillOut(SkillSummary):
    body: str


class SkillVersionOut(BaseModel):
    version: int
    description: str
    body: str
    created_at: datetime


class AgentSkillsOut(BaseModel):
    agent_id: int
    skills: list[str]


def _summary(skill: Skill) -> dict:
    return {
        "name": skill.name,
        "description": skill.description,
        "tools": list(skill.tools or []),
        "version": skill.version,
        "source": skill.source,
        "self_service": skill.self_service,
        "updated_at": skill.updated_at,
    }


def _out(skill: Skill) -> SkillOut:
    return SkillOut(**_summary(skill), body=skill.body)


@router.get(
    "/skills",
    dependencies=[require(Scope.READ)],
    response_model=list[SkillSummary],
    tags=["skills"],
    responses=error_responses(),
)
async def list_skills(session: AsyncSession = Depends(get_db_session)) -> list[SkillSummary]:
    """Every skill, by name, without its body."""
    return [SkillSummary(**_summary(s)) for s in await service.list_skills(session)]


@router.post(
    "/skills",
    dependencies=[require(Scope.ADMIN)],
    response_model=SkillOut,
    status_code=status.HTTP_201_CREATED,
    tags=["skills"],
    responses=error_responses(409),
)
async def create_skill(body: SkillIn, session: AsyncSession = Depends(get_db_session)) -> SkillOut:
    """Create a skill (version 1). 409 when the name is taken."""
    skill = await service.create_skill(session, **body.model_dump(), actor=current_actor())
    await session.commit()
    return _out(skill)


@router.post(
    "/skills/import",
    dependencies=[require(Scope.ADMIN)],
    response_model=SkillOut,
    tags=["skills"],
    responses=error_responses(404, 409),
)
async def import_skill(
    body: SkillImportIn, session: AsyncSession = Depends(get_db_session)
) -> SkillOut:
    """Create or update (a new version) the skill of `SKILLS_DIR/<folder>/SKILL.md`. Its front
    matter's name must be the folder's; a symbolic link or a path is refused."""
    skill = await service.import_folder(
        session, get_settings().skills_dir, body.folder, actor=current_actor()
    )
    await session.commit()
    return _out(skill)


@router.get(
    "/skills/{name}",
    dependencies=[require(Scope.READ)],
    response_model=SkillOut,
    tags=["skills"],
    responses=error_responses(404),
)
async def get_skill(name: str, session: AsyncSession = Depends(get_db_session)) -> SkillOut:
    """One skill, with its body."""
    return _out(await service.get_skill(session, name))


@router.get(
    "/skills/{name}/versions",
    dependencies=[require(Scope.READ)],
    response_model=list[SkillVersionOut],
    tags=["skills"],
    responses=error_responses(404),
)
async def list_skill_versions(
    name: str, session: AsyncSession = Depends(get_db_session)
) -> list[SkillVersionOut]:
    """Every version of a skill's description and body, oldest first."""
    return [
        SkillVersionOut(
            version=v.version, description=v.description, body=v.body, created_at=v.created_at
        )
        for v in await service.versions(session, name)
    ]


@router.patch(
    "/skills/{name}",
    dependencies=[require(Scope.ADMIN)],
    response_model=SkillOut,
    tags=["skills"],
    responses=error_responses(404, 409),
)
async def update_skill(
    name: str, body: SkillUpdate, session: AsyncSession = Depends(get_db_session)
) -> SkillOut:
    """Change a skill; a new description or body is a new version, the old one is kept."""
    skill = await service.update_skill(
        session, name, actor=current_actor(), **body.model_dump(exclude_unset=True)
    )
    await session.commit()
    return _out(skill)


@router.delete(
    "/skills/{name}",
    dependencies=[require(Scope.ADMIN)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["skills"],
    responses=error_responses(404, 409),
)
async def delete_skill(name: str, session: AsyncSession = Depends(get_db_session)) -> None:
    """Delete a skill, its versions, and its grant to every agent."""
    await service.delete_skill(session, name, actor=current_actor())
    await session.commit()


@router.put(
    "/agents/{agent_id}/skills",
    dependencies=[require(Scope.ADMIN)],
    response_model=AgentSkillsOut,
    tags=["skills"],
    responses=error_responses(404, 409),
)
async def set_agent_skills(
    agent_id: int, body: AgentSkillsIn, session: AsyncSession = Depends(get_db_session)
) -> AgentSkillsOut:
    """The skills this agent may load (replaces the list; [] grants none, the default). Only
    their names and descriptions enter its prompt."""
    names = await service.set_agent_skills(session, agent_id, body.skills, actor=current_actor())
    await session.commit()
    return AgentSkillsOut(agent_id=agent_id, skills=names)
