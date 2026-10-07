"""The host helper (serves the host scope of the Admin API.

    python -m app.host.helper          (started by start.sh when HOST_HELPER_ENABLED=true)

A fixed table of operations, typed bodies, no shell. It listens on 127.0.0.1 only, or on a
Unix socket (HOST_HELPER_SOCKET, mode 660, in a directory of mode 750 when created here), and
every call must be signed by the application (`app/host/signing.py`): an unsigned, badly
signed, expired or replayed call gets 401, an operation outside the table 403 and a scope
below the one the operation needs 403. Every call is written to `logs/host-helper-audit.jsonl`
before it runs and again with its result; a job's second line is written when the job ends.

The operations: the application's state, stop, start and restart, restore of a listed
backup and rotation of the key (both with the application stopped and started again), the
configuration in `.env`, and the files under `models/`. They exist for a client that cannot
do them itself: the application in its container, or a script on another machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import socket
import stat
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from app import settings_rules as rules
from app.admin.jobs import Job, JobError, JobRegistry
from app.admin.service import ConflictError, InvalidInputError, NotFoundError
from app.host import ops
from app.host.signing import SignatureError, Verifier
from app.logging_setup import scrub

logger = logging.getLogger("channelagent.host")

SCOPES = {"read": 1, "operate": 2, "admin": 3, "owner": 4}
MAX_BODY_BYTES = 64 * 1024
AUDIT_FILE = "logs/host-helper-audit.jsonl"
AUDIT_LIMIT_MAX = 500

jobs = JobRegistry(prefix="host-")


# --- bodies ---


class RestoreIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_unreadable: bool = False


class RekeyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_unreadable: bool = False
    dry_run: bool = False


class ConfigSetIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(max_length=100)
    value: str = Field(max_length=4096)


class ModelImportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(max_length=4096)
    name: str | None = Field(default=None, max_length=200)
    force: bool = False


class SidecarIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=40)
    image: str = Field(max_length=300)
    port: int = Field(default=8000, ge=1, le=65535)
    egress: str = Field(default="local", max_length=10)
    memory_mb: int = Field(default=256, ge=32, le=4096)
    cpus: float = Field(default=0.5, ge=0.1, le=4.0)


class ModelPullIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: str = Field(max_length=2048)
    name: str | None = Field(default=None, max_length=200)
    sha256: str | None = Field(default=None, max_length=64)
    force: bool = False


# --- the table ---


@dataclass(frozen=True)
class Operation:
    name: str
    method: str
    pattern: re.Pattern
    scope: str
    handler: Callable[..., Awaitable[object]]
    body: type[BaseModel] | None = None


def _job_out(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


async def _status(_params, _body):
    return {**await asyncio.to_thread(ops.app_state), "helper_started_at": STARTED_AT}


def _job(kind: str, runner, *, cancellable: bool = False) -> Job:
    return jobs.start(kind, runner, cancellable=cancellable)


async def _stop(_params, _body):
    return _job("host-stop", ops.stop)


async def _start(_params, _body):
    await asyncio.to_thread(ops.check_not_running)
    return _job("host-start", ops.start)


async def _restart(_params, _body):
    return _job("host-restart", ops.restart)


async def _restore(params, body: RestoreIn):
    name = params["name"]
    await asyncio.to_thread(ops.backup_file, name)  # refused now, not inside the job
    return _job("host-restore", lambda job: ops.restore(job, name, body.allow_unreadable))


async def _rekey(_params, body: RekeyIn):
    return _job("host-rekey", lambda job: ops.rekey(job, body.allow_unreadable, body.dry_run))


def _env_files() -> tuple[str, str]:
    return str(ops.PROJECT_DIR / ".env"), str(ops.PROJECT_DIR / ".env.example")


async def _config_get(_params, _body):
    env, example = _env_files()
    return [
        {
            "key": e["key"],
            "value": None if e["secret"] else (e["value"] or None),
            "is_set": e["is_set"],
            "secret": e["secret"],
        }
        for e in rules.config_entries(env, example)
    ]


async def _config_set(_params, body: ConfigSetIn):
    env, example = _env_files()
    try:
        result = rules.set_config(env, example, body.key, body.value, allow_initial_key=False)
    except rules.ConfigError as exc:
        if exc.kind in ("key_protected", "backup_failed"):
            raise ConflictError(str(exc)) from None
        raise InvalidInputError(str(exc)) from None
    return {"key": body.key, "changed": True, "backup": result["backup"]}


async def _loaded_model() -> str | None:
    from app.api.status import _engine

    return (await _engine()).model


async def _models_list(_params, _body):
    from app.admin import models
    from app.config import get_settings

    loaded, configured = await _loaded_model(), get_settings().model_file
    return [
        {**m, "loaded": m["name"] == loaded, "configured": m["name"] == configured}
        for m in models.list_installed()
    ]


def _int(params: dict, key: str, default=None):
    from app.admin.service import InvalidInputError

    value = params.get(key)
    if value in (None, ""):
        return default
    if not str(value).isdigit():
        raise InvalidInputError(f"{key} is a whole number")
    return int(value)


async def _models_estimate(params, _body):
    """The estimate where the models and the engine are."""
    import asyncio

    from app.admin import memory_estimate

    return await asyncio.to_thread(
        memory_estimate.for_model, params["name"], _int(params, "ctx"),
        slots=_int(params, "slots", 4), cache_type=params.get("cache_type") or "q8_0",
        router_models=_int(params, "router_models", 1), projector=params.get("projector") or None,
    )  # fmt: skip


async def _models_benchmark(params, _body):
    """The measured check, as a job of the helper."""
    import asyncio

    from app.admin import memory_estimate

    name, ctx, slots = params["name"], _int(params, "ctx"), _int(params, "slots", 4)
    await asyncio.to_thread(memory_estimate.for_model, name, ctx, slots=slots)
    return _job("model-benchmark", lambda job: memory_estimate.benchmark(job, name, ctx, slots))


async def _models_delete(params, _body):
    from app.admin import models

    return models.delete_model(params["name"], await _loaded_model())


async def _models_import(_params, body: ModelImportIn):
    from app.admin import models

    plan = models.prepare_import(body.path, body.name, body.force)
    models.claim(plan["name"])

    async def run(job: Job) -> dict:
        try:
            return await models.run_import(job, plan)
        except OSError as exc:
            raise JobError(f"The copy failed: {exc.strerror or 'input/output error'}") from exc

    job = _job("model-import", run, cancellable=True)
    job.task.add_done_callback(lambda _task, name=plan["name"]: models.release(name))
    return job


async def _models_pull(_params, body: ModelPullIn):
    from app.admin import models

    plan = await models.prepare_pull(
        body.spec, body.name, body.sha256.lower() if body.sha256 else None, body.force
    )
    return _job("model-pull", lambda job: models.run_pull(job, plan), cancellable=True)


async def _sidecars_list(_params, _body):
    from app.host import sidecars

    return await sidecars.running()


async def _sidecar_deploy(_params, body: SidecarIn):
    from app.host import sidecars

    values = ops.env_values()
    sidecars.check_request(
        body.name, body.image, sidecars.allowed_images(values.get("MCP_SIDECAR_IMAGES", "")),
        body.egress,
    )  # fmt: skip
    forwarder = values.get("MCP_SIDECAR_FORWARDER_IMAGE") or "python:3.12-slim"
    return _job(
        "sidecar-deploy",
        lambda job: sidecars.deploy(
            job,
            name=body.name,
            image=body.image,
            port=body.port,
            egress=body.egress,
            memory_mb=body.memory_mb,
            cpus=body.cpus,
            forwarder_image=forwarder,
        ),  # fmt: skip
    )


async def _sidecar_remove(params, _body):
    from app.host import sidecars

    return await sidecars.remove_checked(params["name"])


async def _job_get(params, _body):
    return jobs.get(params["job_id"])


async def _job_cancel(params, _body):
    job = jobs.cancel(params["job_id"])
    await asyncio.sleep(0)
    return job


async def _audit(params, _body):
    limit = params.get("limit", "50")
    if not limit.isdigit() or not 1 <= int(limit) <= AUDIT_LIMIT_MAX:
        raise InvalidInputError(f"limit must be a whole number from 1 to {AUDIT_LIMIT_MAX}")
    return await asyncio.to_thread(read_audit, int(limit))


_NAME = r"(?P<name>[A-Za-z0-9._-]{1,200})"
_JOB = r"(?P<job_id>host-[0-9a-f]{12})"


def _op(name, method, path, scope, handler, body=None) -> Operation:
    return Operation(name, method, re.compile(path), scope, handler, body)


TABLE: tuple[Operation, ...] = (
    _op("status", "GET", r"/status", "admin", _status),
    _op("app.stop", "POST", r"/app/stop", "owner", _stop),
    _op("app.start", "POST", r"/app/start", "owner", _start),
    _op("app.restart", "POST", r"/app/restart", "owner", _restart),
    _op("backup.restore", "POST", rf"/backups/{_NAME}/restore", "owner", _restore, RestoreIn),
    _op("rekey", "POST", r"/rekey", "owner", _rekey, RekeyIn),
    _op("config.get", "GET", r"/config", "admin", _config_get),
    _op("config.set", "PATCH", r"/config", "owner", _config_set, ConfigSetIn),
    _op("models.list", "GET", r"/models", "read", _models_list),
    _op("models.delete", "DELETE", rf"/models/{_NAME}", "admin", _models_delete),
    _op("models.estimate", "GET", rf"/models/{_NAME}/estimate", "read", _models_estimate),
    _op("models.benchmark", "POST", rf"/models/{_NAME}/benchmark", "admin", _models_benchmark),
    _op("models.import", "POST", r"/models/import", "admin", _models_import, ModelImportIn),
    _op("models.pull", "POST", r"/models/pull", "admin", _models_pull, ModelPullIn),
    _op("sidecars.list", "GET", r"/sidecars", "admin", _sidecars_list),
    _op("sidecars.deploy", "POST", r"/sidecars", "owner", _sidecar_deploy, SidecarIn),
    _op(
        "sidecars.remove",
        "DELETE",
        r"/sidecars/(?P<name>[a-z0-9-]{1,40})",
        "owner",
        _sidecar_remove,
    ),  # fmt: skip
    _op("job.get", "GET", rf"/jobs/{_JOB}", "admin", _job_get),
    _op("job.cancel", "POST", rf"/jobs/{_JOB}/cancel", "admin", _job_cancel),
    _op("audit", "GET", r"/audit", "admin", _audit),
)


def find(method: str, path: str) -> tuple[Operation, dict] | None:
    for op in TABLE:
        match = op.pattern.fullmatch(path)
        if match and op.method == method:
            return op, match.groupdict()
    return None


# --- audit ---


def _audit_path() -> Path:
    return ops.PROJECT_DIR / AUDIT_FILE


def audit(record: dict) -> None:
    """Append one line; the file is created readable by its owner only."""
    path = _audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"at": datetime.now(UTC).isoformat(), **record}, ensure_ascii=False)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (scrub(line) + "\n").encode())
    finally:
        os.close(fd)


def read_audit(limit: int) -> list[dict]:
    try:
        lines = _audit_path().read_text().splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines[-limit:] if line.strip()][::-1]


def _safe_args(params: dict, body: BaseModel | None) -> dict:
    """What the audit keeps of a call: the path parameters and the body, a value of a secret
    setting removed."""
    args = dict(params)
    if body is not None:
        data = body.model_dump()
        if isinstance(body, ConfigSetIn) and data["key"] in rules.SENSITIVE:
            data["value"] = "(secret)"
        args.update(data)
    return args


# --- the server ---


def _reply(status_code: int, detail: str, **extra) -> JSONResponse:
    return JSONResponse({"detail": detail, **extra}, status_code=status_code)


class Helper:
    def __init__(self, secret: str) -> None:
        self.verifier = Verifier(secret)

    async def dispatch(self, request: Request) -> Response:
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY_BYTES:
                return _reply(413, "Request body too large")
        path = request.url.path
        signed_path = path + (f"?{request.url.query}" if request.url.query else "")
        call_id = uuid.uuid4().hex[:12]
        headers = {k.lower(): v for k, v in request.headers.items()}
        try:
            scope, actor = self.verifier.verify(request.method, signed_path, body, headers)
        except SignatureError as exc:
            audit(
                {
                    "call": call_id,
                    "phase": "refused",
                    "method": request.method,
                    "path": path[:200],
                    "result": str(exc),
                }
            )
            return _reply(401, f"Refused: {exc}")
        found = find(request.method, path)
        if found is None:
            audit(
                {
                    "call": call_id,
                    "phase": "refused",
                    "actor": actor,
                    "method": request.method,
                    "path": path[:200],
                    "result": "operation not in the table",
                }
            )
            return _reply(403, "Operation not in the helper's table")
        op, params = found
        if SCOPES.get(scope, 0) < SCOPES[op.scope]:
            audit(
                {
                    "call": call_id,
                    "phase": "refused",
                    "actor": actor,
                    "op": op.name,
                    "result": f"scope {scope} below {op.scope}",
                }
            )
            return _reply(403, "Insufficient scope")
        params.update(dict(request.query_params))
        parsed = None
        if op.body is not None:
            try:
                parsed = op.body.model_validate_json(body or b"{}")
            except ValidationError as exc:
                return _reply(422, "Invalid body", errors=json.loads(exc.json(include_url=False)))
        base = {"call": call_id, "actor": actor, "scope": scope, "op": op.name}
        audit({**base, "phase": "before", "args": _safe_args(params, parsed)})
        try:
            result = await op.handler(params, parsed)
        except (NotFoundError, ConflictError, InvalidInputError, JobError) as exc:
            code = {NotFoundError: 404, ConflictError: 409}.get(type(exc), 422)
            audit({**base, "phase": "after", "result": "refused", "detail": str(exc)[:300]})
            return _reply(code, str(exc))
        except Exception:  # noqa: BLE001 - the detail goes to the log, not to the caller
            error_id = uuid.uuid4().hex[:12]
            logger.exception("Host helper call %s failed, error id %s", call_id, error_id)
            audit({**base, "phase": "after", "result": "error", "detail": error_id})
            return _reply(500, "Internal error", error_id=error_id)
        if isinstance(result, Job):
            if op.name in ("job.get", "job.cancel"):
                audit({**base, "phase": "after", "result": "ok"})
                return JSONResponse(_job_out(result))
            self._audit_when_done(result, base)
            return JSONResponse(_job_out(result), status_code=202)
        audit({**base, "phase": "after", "result": "ok"})
        # Dates (a model's modified_at) and other non-JSON values, the way FastAPI encodes them.
        return JSONResponse(jsonable_encoder(result))

    @staticmethod
    def _audit_when_done(job: Job, base: dict) -> None:
        def done(_task) -> None:
            audit(
                {
                    **base,
                    "phase": "after",
                    "job": job.id,
                    "result": job.status,
                    "detail": (job.error or "")[:300] or None,
                }
            )

        job.task.add_done_callback(done)


STARTED_AT = datetime.now(UTC).isoformat()


def build_app(secret: str) -> Starlette:
    helper = Helper(secret)
    methods = sorted({op.method for op in TABLE} | {"PUT"})
    return Starlette(routes=[Route("/{path:path}", helper.dispatch, methods=methods)])


def _socket_ready(path: Path) -> None:
    """A directory created here is for the owner and the group only; a stale socket is
    removed, anything else at that path is left alone and refused."""
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True)
        os.chmod(path.parent, 0o750)  # explicit: mkdir's mode is cut by the umask (077 here)
    if path.exists() or path.is_symlink():
        if not stat.S_ISSOCK(path.lstat().st_mode):
            raise SystemExit(f"{path} exists and is not a socket: refusing to replace it")
        path.unlink()


def bind_socket(path: Path) -> socket.socket:
    """The listening Unix socket, bound here and not by uvicorn, which would make it 666:
    mode 660 from the start (umask during the bind), in a directory of mode 750."""
    _socket_ready(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o117)
    try:
        sock.bind(str(path))
    finally:
        os.umask(old)
    os.chmod(path, 0o660)
    return sock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The ChannelAgent host helper.")
    parser.add_argument("--check", action="store_true", help="check the settings and exit")
    args = parser.parse_args(argv)
    values = ops.env_values()
    secret = values.get("HOST_HELPER_SECRET", "")
    if values.get("HOST_HELPER_ENABLED") != "true":
        print("HOST_HELPER_ENABLED is not true: the helper does not start.", file=sys.stderr)
        return 2
    if not rules.api_key_is_acceptable(secret):
        print(
            "HOST_HELPER_SECRET is missing or too weak: the helper does not start.", file=sys.stderr
        )
        return 2
    socket_path = values.get("HOST_HELPER_SOCKET", "")
    port = int(values.get("HOST_HELPER_PORT") or 8701)
    # The helper reaches this machine's engine the way --native does, for the models.
    engine = values.get("LLAMA_SERVER_URL") or "http://host.docker.internal:8080"
    if rules.engine_is_local(engine):
        engine = f"http://127.0.0.1:{values.get('LLAMA_PORT') or 8080}"
    os.environ["LLAMA_SERVER_URL"] = engine
    if args.check:
        where = f"unix:{socket_path}" if socket_path else f"127.0.0.1:{port}"
        print(f"ok: would listen on {where}")
        return 0
    # Only a real start changes the process: --check leaves the directory and logging alone.
    os.chdir(ops.PROJECT_DIR)
    from app.logging_setup import configure_logging, install_redaction

    configure_logging()
    install_redaction()
    import uvicorn

    app = build_app(secret)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    sockets = [bind_socket(Path(socket_path))] if socket_path else None
    logger.info("Host helper listening on %s", socket_path or f"127.0.0.1:{port}")
    server = uvicorn.Server(config)
    try:
        asyncio.run(server.serve(sockets=sockets))
    finally:
        if socket_path and Path(socket_path).is_socket():
            Path(socket_path).unlink()
    return 0 if server.started else 1


if __name__ == "__main__":
    sys.exit(main())
