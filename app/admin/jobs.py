"""Long operations as jobs: a job has an id, a status, a progress and can be
cancelled. The Admin API returns `202` and a job for anything slow; the script waits
for it and the UI shows its progress. Held in memory (one process serves the API), the
last `keep` jobs are remembered.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.admin.service import ConflictError, InvalidInputError, NotFoundError
from app.logging_setup import scrub

logger = logging.getLogger("channelagent")

FINAL = frozenset({"done", "failed", "cancelled"})


class JobError(Exception):
    """A failure whose message is safe to show the administrator."""


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"
    progress: float | None = None
    message: str | None = None
    result: dict | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    task: asyncio.Task | None = field(default=None, repr=False, compare=False)
    # A job that drives a child process to the end (the host helper's stop, restore, rekey)
    # cannot be cut half-way: cancelling it is refused.
    cancellable: bool = field(default=True, repr=False, compare=False)

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        if progress is not None:
            self.progress = round(min(max(progress, 0.0), 1.0), 3)
        if message is not None:
            self.message = message
        self.updated_at = datetime.now(UTC)


Runner = Callable[[Job], Awaitable[dict | None]]


class JobRegistry:
    def __init__(self, keep: int = 100, prefix: str = "") -> None:
        self._jobs: dict[str, Job] = {}
        self._keep = keep
        # The host helper's jobs are named `host-...`: the API knows where to ask.
        self._prefix = prefix

    def start(self, kind: str, runner: Runner, *, cancellable: bool = True) -> Job:
        job = Job(id=self._prefix + uuid.uuid4().hex[:12], kind=kind, cancellable=cancellable)
        self._jobs[job.id] = job
        job.task = asyncio.get_running_loop().create_task(self._run(job, runner))
        self._prune()
        return job

    async def _run(self, job: Job, runner: Runner) -> None:
        try:
            job.result = await runner(job)
            job.status = "done"
            job.progress = 1.0
        except asyncio.CancelledError:
            job.status = "cancelled"
        except (JobError, ConflictError, InvalidInputError, NotFoundError) as exc:
            # written to be shown to the administrator, unlike any other exception
            job.status, job.error = "failed", scrub(str(exc))[:300]
        except Exception:  # noqa: BLE001 - the detail goes to the log, not to the caller
            error_id = uuid.uuid4().hex[:12]
            logger.exception("Job %s (%s) failed, error id %s", job.id, job.kind, error_id)
            job.status, job.error = "failed", f"Internal error, see the log (error id {error_id})"
        finally:
            job.updated_at = datetime.now(UTC)

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise NotFoundError(f"No job {job_id}") from None

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status in FINAL:
            raise ConflictError(f"Job {job_id} is already {job.status}")
        if not job.cancellable:
            raise ConflictError(f"Job {job_id} ({job.kind}) cannot be stopped half-way")
        if job.task is not None:
            job.task.cancel()
        return job

    def _prune(self) -> None:
        finished = [j for j in self._jobs.values() if j.status in FINAL]
        surplus = max(0, len(self._jobs) - self._keep)
        for job in sorted(finished, key=lambda j: j.created_at)[:surplus]:
            del self._jobs[job.id]

    def clear(self) -> None:
        self._jobs.clear()


registry = JobRegistry()
