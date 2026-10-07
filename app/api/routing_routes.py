"""Which model answers a turn when its agent has none of its own: an
ordered list of rules and a default, read by app.graph.run_turn. No classifier
(D7 — rules on what is known about the message
before any model call, then the default.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import routing as service
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.schemas import RoutingIn, RoutingOut, RoutingRuleOut
from app.api.scopes import Scope, require

router = APIRouter()


def _out(state: dict) -> RoutingOut:
    return RoutingOut(
        default_model=state["default_model"],
        rules=[RoutingRuleOut(**rule) for rule in state["rules"]],
        model_ctx_sizes=state["model_ctx_sizes"],
        max_tools=state["max_tools"],
        tool_rules=state["tool_rules"],
        tool_model=state["tool_model"],
    )


@router.get(
    "/routing",
    dependencies=[require(Scope.READ)],
    response_model=RoutingOut,
    tags=["routing"],
    responses=error_responses(),
)
async def get_routing(session: AsyncSession = Depends(get_db_session)) -> RoutingOut:
    """The rules, in the order they are tried, and the default model a turn falls
    back to when none matches (or its agent has none of its own).
    """
    return _out(await service.get_routing(session))


@router.put(
    "/routing",
    dependencies=[require(Scope.ADMIN)],
    response_model=RoutingOut,
    tags=["routing"],
    responses=error_responses(409),
)
async def set_routing(
    body: RoutingIn, session: AsyncSession = Depends(get_db_session)
) -> RoutingOut:
    """Replace the whole routing table: the rules, in order, the default model,
    each model's own context size (narrows LLAMA_CTX_SIZE, never widens it), and the tool
    budget: at most `max_tools` MCP tools per turn (0 = no cap; 5 when left out), the
    tool rules that choose them first, and the model of tool turns.
    """
    result = await service.set_routing(
        session,
        default_model=body.default_model,
        rules=[rule.model_dump() for rule in body.rules],
        model_ctx_sizes=body.model_ctx_sizes,
        max_tools=body.max_tools,
        tool_rules=[rule.model_dump() for rule in body.tool_rules],
        tool_model=body.tool_model,
        actor=current_actor(),
    )
    await session.commit()
    return _out(result)
