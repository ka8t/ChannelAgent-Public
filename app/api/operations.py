"""Jobs and backups. Every long operation is a job: the route returns `202` and
the job, and `/jobs/{id}` says how it is going.
"""

import asyncio
import re

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import backup_schedule, export, retention, service
from app.admin.jobs import Job, JobError, registry
from app.admin.restore import list_backups
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.schemas import (
    BackupOut,
    BackupScheduleIn,
    BackupScheduleOut,
    ExportIn,
    ImportIn,
    JobOut,
    RetentionIn,
    RetentionOut,
    RetentionRunIn,
)
from app.api.scopes import Scope, require
from app.config import get_settings
from app.db.backup import BackupError, make_backup
from app.db.session import session_scope, sqlite_file_path
from app.host import client as helper

router = APIRouter()


def _database_file():
    path = sqlite_file_path(get_settings().database_url)
    if path is None:
        raise service.ConflictError("Backups are only available for a SQLite database")
    return path


@router.get(
    "/jobs",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[JobOut],
    tags=["jobs"],
    responses=error_responses(),
)
async def list_jobs() -> list[Job]:
    """The recent jobs, newest first."""
    return registry.list()


@router.get(
    "/jobs/{job_id}",
    dependencies=[require(Scope.ADMIN)],
    response_model=JobOut,
    tags=["jobs"],
    responses=error_responses(404),
)
async def get_job(job_id: str) -> Job | dict:
    """One job: status, progress, result or error. A `host-...` job is the host helper's."""
    if job_id.startswith(helper.JOB_PREFIX):
        return await helper.call("GET", f"/jobs/{job_id}", scope="admin", actor=current_actor())
    return registry.get(job_id)


@router.post(
    "/jobs/{job_id}/cancel",
    dependencies=[require(Scope.ADMIN)],
    response_model=JobOut,
    tags=["jobs"],
    responses=error_responses(404, 409),
)
async def cancel_job(job_id: str) -> Job | dict:
    """Ask a running job to stop. A finished job cannot be cancelled (409), nor a stop,
    restore or rekey of the host helper."""
    if job_id.startswith(helper.JOB_PREFIX):
        return await helper.call(
            "POST", f"/jobs/{job_id}/cancel", scope="admin", actor=current_actor()
        )
    job = registry.cancel(job_id)
    await asyncio.sleep(0)  # let the cancellation reach the task before answering
    return job


@router.get(
    "/backups",
    dependencies=[require(Scope.ADMIN)],
    response_model=list[BackupOut],
    tags=["backups"],
    responses=error_responses(409),
)
async def list_database_backups() -> list[BackupOut]:
    """The backups of the database in `backups/`, newest first."""
    return [
        BackupOut(name=b.path.name, kind=b.kind, size_bytes=b.size, created_at=b.stamp)
        for b in list_backups(_database_file())
    ]


@router.post(
    "/backups",
    dependencies=[require(Scope.ADMIN)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["backups"],
    responses=error_responses(409),
)
async def create_database_backup() -> Job:
    """Copy and verify the database into `backups/`. Returns the job doing it."""
    path = _database_file()
    actor = current_actor()

    async def run(job: Job) -> dict:
        job.update(0.1, "Copying the database")
        try:
            target = await asyncio.to_thread(make_backup, path, "manual")
        except BackupError as exc:
            raise JobError(str(exc)) from exc
        async with session_scope() as session:
            await service.record_admin_event(
                session,
                actor=actor,
                action="backup.create",
                target_type="backup",
                details={"name": target.name},
            )
            await session.commit()
        return {"name": target.name, "size_bytes": target.stat().st_size}

    return registry.start("backup", run)


# --- password-protected export and import ---

EXPORT_NAME = re.compile(r"channelagent-export-\d{8}T\d{12}Z\.caexport")


def _export_paths():
    from app.checkpoints import checkpoint_db_path

    database = _database_file()
    return database, checkpoint_db_path(), database.parent / "backups"


@router.post(
    "/backups/export",
    dependencies=[require(Scope.OWNER)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["backups"],
    responses=error_responses(409),
)
async def export_backup(body: ExportIn) -> Job:
    """Write the database and the conversations into one file of `backups/`, encrypted with a
    key derived from `password` (scrypt, AES-256-GCM), for storage outside the machine. Returns
    the job; its result names the file. Owner scope: the file holds everything."""
    try:
        export.check_password(body.password)
    except export.ExportError as exc:
        raise service.InvalidInputError(str(exc)) from None
    database, checkpoints, directory = _export_paths()
    actor, password = current_actor(), body.password

    async def run(job: Job) -> dict:
        job.update(0.1, "Copying and encrypting the data")
        try:
            result = await asyncio.to_thread(
                export.export, database, checkpoints, password, directory
            )
        except export.ExportError as exc:
            raise JobError(str(exc)) from exc
        async with session_scope() as session:
            await service.record_admin_event(
                session, actor=actor, action="backup.export", target_type="backup",
                details={"name": result["name"], "size_bytes": result["size_bytes"]},
            )  # fmt: skip
            await session.commit()
        return result

    return registry.start("export", run)


@router.post(
    "/backups/import",
    dependencies=[require(Scope.OWNER)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["backups"],
    responses=error_responses(404, 409),
)
async def import_backup(body: ImportIn) -> Job:
    """Decrypt an export of `backups/` into a new directory `data/imports/import-<stamp>/`,
    checked (integrity, row counts of its manifest); the live data is not touched. A wrong
    password fails the job. Putting the data in place is a restore, application stopped."""
    database, _checkpoints, directory = _export_paths()
    if not EXPORT_NAME.fullmatch(body.name):
        raise service.InvalidInputError(
            "name is an export file: channelagent-export-<stamp>.caexport"
        )
    source = directory / body.name
    if not source.is_file():
        raise service.NotFoundError(f"No export {body.name}")
    actor, password = current_actor(), body.password

    async def run(job: Job) -> dict:
        job.update(0.1, "Decrypting and checking the export")
        try:
            result = await asyncio.to_thread(
                export.import_export, source, password, database.parent
            )
        except export.ExportError as exc:
            raise JobError(str(exc)) from exc
        async with session_scope() as session:
            await service.record_admin_event(
                session, actor=actor, action="backup.import", target_type="backup",
                details={"name": body.name, "directory": result["directory"]},
            )  # fmt: skip
            await session.commit()
        return result

    return registry.start("import", run)


def _schedule_out(row) -> BackupScheduleOut:
    return BackupScheduleOut(
        enabled=row.enabled,
        interval_minutes=row.interval_minutes,
        keep=row.keep,
        last_run_at=backup_schedule._as_utc(row.last_run_at),
        last_success_at=backup_schedule._as_utc(row.last_success_at),
        last_error=row.last_error,
        last_files=list(row.last_files or []),
        failing=row.failing,
        next_run_at=backup_schedule.next_run_at(row),
    )


@router.get(
    "/backups/schedule",
    dependencies=[require(Scope.READ)],
    response_model=BackupScheduleOut,
    tags=["backups"],
    responses=error_responses(409),
)
async def get_backup_schedule(
    session: AsyncSession = Depends(get_db_session),
) -> BackupScheduleOut:
    """The scheduled backup: enabled, interval, how many are kept, the last run and
    the next one."""
    _database_file()
    row = await backup_schedule.get_schedule(session)
    await session.commit()
    return _schedule_out(row)


@router.put(
    "/backups/schedule",
    dependencies=[require(Scope.ADMIN)],
    response_model=BackupScheduleOut,
    tags=["backups"],
    responses=error_responses(409),
)
async def set_backup_schedule(
    body: BackupScheduleIn, session: AsyncSession = Depends(get_db_session)
) -> BackupScheduleOut:
    """Change the scheduled backup: enabled, `interval_minutes` (1 to 10080), `keep` (1
    to 365). Applies within a minute, without a restart."""
    _database_file()
    row = await backup_schedule.set_schedule(
        session, body.model_dump(exclude_none=True), actor=current_actor()
    )
    await session.commit()
    return _schedule_out(row)


@router.post(
    "/backups/schedule/run",
    dependencies=[require(Scope.ADMIN)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["backups"],
    responses=error_responses(409),
)
async def run_backup_schedule_now() -> Job:
    """Run the scheduled backup now (database and checkpoints, verified, rotated). Returns
    the job; a failure is in its error and in `GET /backups/schedule`."""
    _database_file()
    actor = current_actor()

    async def run(job: Job) -> dict:
        job.update(0.1, "Copying the database and the checkpoints")
        row = await backup_schedule.run_once(actor=actor)
        if row.failing:
            raise JobError(row.last_error or "the backup failed")
        return {"files": list(row.last_files)}

    return registry.start("scheduled-backup", run)


# --- retention ---


def _retention_out(row) -> RetentionOut:
    return RetentionOut(
        messages_days=row.messages_days,
        tool_calls_days=row.tool_calls_days,
        api_calls_days=row.api_calls_days,
        conversations_days=row.conversations_days,
        last_run_at=backup_schedule._as_utc(row.last_run_at),
        last_run=row.last_run,
    )


@router.get(
    "/retention",
    dependencies=[require(Scope.ADMIN)],
    response_model=RetentionOut,
    tags=["retention"],
    responses=error_responses(),
)
async def get_retention(session: AsyncSession = Depends(get_db_session)) -> RetentionOut:
    """The retention periods (days; null = kept for ever, the default) and the last run."""
    row = await retention.get_config(session)
    await session.commit()
    return _retention_out(row)


@router.put(
    "/retention",
    dependencies=[require(Scope.OWNER)],
    response_model=RetentionOut,
    tags=["retention"],
    responses=error_responses(409),
)
async def set_retention(
    body: RetentionIn, session: AsyncSession = Depends(get_db_session)
) -> RetentionOut:
    """Set the retention periods: 1 to 3650 days, or null to keep for ever. A period left out
    keeps its value. Nothing is deleted until a run."""
    row = await retention.set_config(
        session, body.model_dump(exclude_unset=True), actor=current_actor()
    )
    await session.commit()
    return _retention_out(row)


@router.post(
    "/retention/run",
    dependencies=[require(Scope.OWNER)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["retention"],
    responses=error_responses(409),
)
async def run_retention(body: RetentionRunIn) -> Job:
    """Apply the retention: `dry_run` (the default) counts what would go per table; a real run
    backs up the database and the checkpoints first, then deletes. Returns the job."""
    _database_file()
    actor = current_actor()

    async def run(job: Job) -> dict:
        job.update(0.1, "Counting" if body.dry_run else "Backing up, then deleting")
        async with session_scope() as session:
            result = await retention.run(session, dry_run=body.dry_run, actor=actor)
            await session.commit()
        return result

    return registry.start("retention", run, cancellable=False)
