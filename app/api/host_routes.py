"""The host scope of the Admin API: what only the host can do, forwarded to the host
helper after this API's own authorization (owner scope for anything that changes the host).

Each change is an admin event in this database and a message to the administrators
before the helper runs it; the helper keeps its own audit, two lines per call, in
`logs/host-helper-audit.jsonl` (a restore replaces this database, that file stays). Stop,
start, restart, restore and rekey are jobs of the helper, named `host-...`: `GET /jobs/{id}`
follows them through the helper, also while the application restarts.
"""

from fastapi import APIRouter, Depends, Path, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import service
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.schemas import (
    BackupRestoreIn,
    HostAuditOut,
    HostRekeyIn,
    HostStatusOut,
    JobOut,
    SidecarIn,
    SidecarOut,
)
from app.api.scopes import Scope, require
from app.channels.notify import notify_admins
from app.host import client as helper

router = APIRouter()


async def _forward_change(
    session: AsyncSession,
    *,
    method: str,
    path: str,
    action: str,
    notice: str,
    body: dict | None = None,
    details: dict | None = None,
) -> dict:
    """Forward one owner-scope change, then record it and tell the administrators."""
    actor = current_actor()
    job = await helper.call(method, path, scope="owner", actor=actor, body=body)
    await service.record_admin_event(
        session,
        actor=actor,
        action=action,
        target_type="host",
        details={**(details or {}), "job": job.get("id")},
    )
    await session.commit()
    await notify_admins(session, f"Host operation by {actor}: {notice} (job {job.get('id')}).")
    return job


@router.get(
    "/host/status",
    dependencies=[require(Scope.ADMIN)],
    response_model=HostStatusOut,
    tags=["host"],
    responses=error_responses(409),
)
async def host_status() -> dict:
    """Whether the application runs on the host, in which mode, as the host helper sees it."""
    return await helper.call("GET", "/status", scope="admin", actor=current_actor())


@router.post(
    "/host/stop",
    dependencies=[require(Scope.OWNER)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["host"],
    responses=error_responses(409),
)
async def host_stop(session: AsyncSession = Depends(get_db_session)) -> dict:
    """Stop the application (native process or container). Returns the helper's job; the API
    stops answering once it is done."""
    return await _forward_change(
        session, method="POST", path="/app/stop", action="host.stop", notice="stop"
    )


@router.post(
    "/host/start",
    dependencies=[require(Scope.OWNER)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["host"],
    responses=error_responses(409),
)
async def host_start(session: AsyncSession = Depends(get_db_session)) -> dict:
    """Start the application in the mode start.sh last used (409 when it already runs). From
    a stopped application, call it with `./start.sh --admin host-start` (in process)."""
    return await _forward_change(
        session, method="POST", path="/app/start", action="host.start", notice="start"
    )


@router.post(
    "/host/restart",
    dependencies=[require(Scope.OWNER)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["host"],
    responses=error_responses(409),
)
async def host_restart(session: AsyncSession = Depends(get_db_session)) -> dict:
    """Stop and start the application again, for example after a configuration change."""
    return await _forward_change(
        session, method="POST", path="/app/restart", action="host.restart", notice="restart"
    )


@router.post(
    "/backups/{name}/restore",
    dependencies=[require(Scope.OWNER)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["backups"],
    responses=error_responses(404, 409),
)
async def restore_backup(
    body: BackupRestoreIn,
    name: str = Path(pattern=r"^[A-Za-z0-9._-]{1,200}$", description="a name from GET /backups"),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Restore a listed backup: the helper stops the application, runs the checked restore
    (the current database is kept as a before-restore copy) and starts it again."""
    return await _forward_change(
        session,
        method="POST",
        path=f"/backups/{name}/restore",
        action="host.restore",
        notice=f"restore of {name}",
        body=body.model_dump(),
        details={"backup": name, "allow_unreadable": body.allow_unreadable},
    )


@router.post(
    "/host/rekey",
    dependencies=[require(Scope.OWNER)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["host"],
    responses=error_responses(409),
)
async def host_rekey(body: HostRekeyIn, session: AsyncSession = Depends(get_db_session)) -> dict:
    """Rotate ENCRYPTION_KEY with the application stopped (the guided sequence of
    ./start.sh --rekey), then start it again. A dry run stops it too: the check needs that."""
    return await _forward_change(
        session,
        method="POST",
        path="/rekey",
        action="host.rekey",
        notice="dry run of the key rotation" if body.dry_run else "key rotation",
        body=body.model_dump(),
        details=body.model_dump(),
    )


@router.get(
    "/host/audit",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[HostAuditOut],
    tags=["host"],
    responses=error_responses(409),
)
async def host_audit(limit: int = Query(default=50, ge=1, le=500)) -> list:
    """The host helper's audit lines, newest first: before and after each call, and refusals."""
    return await helper.call(
        "GET", "/audit", scope="admin", actor=current_actor(), params={"limit": limit}
    )


# --- sidecar containers for third-party MCP servers ---


@router.get(
    "/mcp/sidecars",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[SidecarOut],
    tags=["mcp"],
    responses=error_responses(409),
)
async def list_sidecars() -> list:
    """The sidecar containers of third-party MCP servers, with their state and port."""
    return await helper.call("GET", "/sidecars", scope="admin", actor=current_actor())


@router.post(
    "/mcp/sidecars",
    dependencies=[require(Scope.OWNER)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["mcp"],
    responses=error_responses(409),
)
async def deploy_sidecar(body: SidecarIn, session: AsyncSession = Depends(get_db_session)) -> dict:
    """Run a third-party MCP server in its own container (an image pinned in
    MCP_SIDECAR_IMAGES): internal network, read-only, no volume. The job's result gives the
    URL to register it with create-server."""
    return await _forward_change(
        session,
        method="POST",
        path="/sidecars",
        action="mcp_sidecar.deploy",
        notice=f"sidecar {body.name} ({body.image}, egress {body.egress})",
        body=body.model_dump(),
        details={"name": body.name, "image": body.image, "egress": body.egress},
    )


@router.delete(
    "/mcp/sidecars/{name}",
    dependencies=[require(Scope.OWNER)],
    tags=["mcp"],
    responses=error_responses(404, 409),
)
async def remove_sidecar(
    name: str = Path(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$"),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Stop and remove a sidecar, its forwarder and its network."""
    actor = current_actor()
    result = await helper.call("DELETE", f"/sidecars/{name}", scope="owner", actor=actor)
    await service.record_admin_event(
        session, actor=actor, action="mcp_sidecar.remove", target_type="mcp_sidecar",
        details={"name": name},
    )  # fmt: skip
    await session.commit()
    return result
