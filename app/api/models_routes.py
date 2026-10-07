"""Models on disk: list, delete, import and pull as jobs. Where this process has no
models directory (the application in its container) and the host helper is enabled, the
host's `models/` is managed through it."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin import memory_estimate, models, service
from app.admin.jobs import Job, JobError, registry
from app.api.actor import current_actor
from app.api.deps import get_db_session
from app.api.errors import error_responses
from app.api.schemas import JobOut, ModelImportIn, ModelOut, ModelPullIn
from app.api.scopes import Scope, require
from app.api.status import _engine as engine_status
from app.config import get_settings
from app.db.session import session_scope
from app.host import client as helper

router = APIRouter()


def _on_host() -> bool:
    return not Path(get_settings().models_dir).is_dir() and helper.enabled()


async def _record(action: str, details: dict) -> None:
    async with session_scope() as session:
        await service.record_admin_event(
            session, actor=current_actor(), action=action, target_type="model", details=details
        )
        await session.commit()


@router.get(
    "/models",
    dependencies=[require(Scope.READ)],
    response_model=list[ModelOut],
    tags=["models"],
    responses=error_responses(409),
)
async def list_models() -> list[ModelOut]:
    """The models in the models directory, and which one the running engine has loaded."""
    if _on_host():
        return await helper.call("GET", "/models", scope="read", actor=current_actor())
    installed = models.list_installed()
    loaded = (await engine_status()).model
    configured = get_settings().model_file
    return [
        ModelOut(**m, loaded=m["name"] == loaded, configured=m["name"] == configured)
        for m in installed
    ]


@router.delete(
    "/models/{name}",
    dependencies=[require(Scope.ADMIN)],
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["models"],
    responses=error_responses(404, 409),
)
async def delete_model(name: str, session: AsyncSession = Depends(get_db_session)) -> None:
    """Delete a model file and its recorded hash; the loaded or configured model is refused."""
    if _on_host():
        result = await helper.call(
            "DELETE", f"/models/{models.valid_name(name)}", scope="admin", actor=current_actor()
        )
    else:
        result = models.delete_model(name, (await engine_status()).model)
    await service.record_admin_event(
        session,
        actor=current_actor(),
        action="model.delete",
        target_type="model",
        details=result,
    )
    await session.commit()


@router.post(
    "/models/import",
    dependencies=[require(Scope.ADMIN)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["models"],
    responses=error_responses(409),
)
async def import_model(body: ModelImportIn) -> Job:
    """Copy a local GGUF file into the models directory. Returns the job doing it."""
    if _on_host():
        job = await helper.call(
            "POST", "/models/import", scope="admin", actor=current_actor(), body=body.model_dump()
        )
        await _record("model.import", {"job": job["id"], "via": "host-helper"})
        return job
    plan = models.prepare_import(body.path, body.name, body.force)
    models.claim(plan["name"])
    actor = current_actor()

    async def run(job: Job) -> dict:
        try:
            result = await models.run_import(job, plan)
        except OSError as exc:
            raise JobError(f"The copy failed: {exc.strerror or 'input/output error'}") from exc
        async with session_scope() as session:
            await service.record_admin_event(
                session,
                actor=actor,
                action="model.import",
                target_type="model",
                details=result,
            )
            await session.commit()
        return result

    job = registry.start("model-import", run)
    # Whatever ends the job, done, failed or cancelled, the name is free again.
    job.task.add_done_callback(lambda _task, name=plan["name"]: models.release(name))
    return job


@router.post(
    "/models/pull",
    dependencies=[require(Scope.ADMIN)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["models"],
    responses=error_responses(409),
)
async def pull_model(body: ModelPullIn) -> Job:
    """Download a model from the hub (<repo>[:quant]) or an https URL. Returns the job."""
    if _on_host():
        job = await helper.call(
            "POST", "/models/pull", scope="admin", actor=current_actor(), body=body.model_dump()
        )
        await _record("model.pull", {"job": job["id"], "via": "host-helper"})
        return job
    plan = await models.prepare_pull(
        body.spec, body.name, body.sha256.lower() if body.sha256 else None, body.force
    )
    actor = current_actor()

    async def run(job: Job) -> dict:
        result = await models.run_pull(job, plan)
        async with session_scope() as session:
            await service.record_admin_event(
                session,
                actor=actor,
                action="model.pull",
                target_type="model",
                details=result,
            )
            await session.commit()
        return result

    return registry.start("model-pull", run)


# --- memory estimate and measured check ---


@router.get(
    "/models/{name}/estimate",
    dependencies=[require(Scope.READ)],
    tags=["models"],
    responses=error_responses(404, 409),
)
async def estimate_model(
    name: str,
    ctx: int | None = Query(default=None, description="context tokens (default LLAMA_CTX_SIZE)"),
    slots: int = Query(default=4, description="parallel slots of the engine"),
    cache_type: str = Query(default="q8_0", description="the context cache type"),
    router_models: int = Query(default=1, description="models loaded at once (router mode)"),
    projector: str | None = Query(default=None, description="a projector file loaded beside"),
) -> dict:
    """The memory this model needs with this context, line by line with what each line is,
    the GPU memory of the machine the engine runs on, and the verdict: fits, tight or not."""
    query = {"ctx": ctx, "slots": slots, "cache_type": cache_type,
             "router_models": router_models, "projector": projector}  # fmt: skip
    if _on_host():
        return await helper.call(
            "GET", f"/models/{models.valid_name(name)}/estimate", scope="read",
            actor=current_actor(), params={k: v for k, v in query.items() if v is not None},
        )  # fmt: skip
    return await asyncio.to_thread(memory_estimate.for_model, name, **query)


@router.post(
    "/models/{name}/benchmark",
    dependencies=[require(Scope.ADMIN)],
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["models"],
    responses=error_responses(404, 409),
)
async def benchmark_model(
    name: str,
    ctx: int | None = Query(default=None, description="context tokens (default LLAMA_CTX_SIZE)"),
    slots: int = Query(default=4, description="parallel slots of the engine"),
) -> Job:
    """Run the model in a temporary engine, ask one question and report its real speed
    (tokens per second) and whether the machine swapped: does it run smoothly? Returns the
    job. Refused when the estimate says it does not fit."""
    query = {"ctx": ctx, "slots": slots}
    if _on_host():
        job = await helper.call(
            "POST", f"/models/{models.valid_name(name)}/benchmark", scope="admin",
            actor=current_actor(), params={k: v for k, v in query.items() if v is not None},
        )  # fmt: skip
        await _record("model.benchmark", {"name": name, "job": job["id"], "via": "host-helper"})
        return job
    await asyncio.to_thread(memory_estimate.for_model, name, ctx, slots=slots)  # 404, 422 first
    await _record("model.benchmark", {"name": name, "ctx": ctx, "slots": slots})
    return registry.start(
        "model-benchmark", lambda job: memory_estimate.benchmark(job, name, ctx, slots)
    )
