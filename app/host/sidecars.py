"""Sidecar containers for third-party MCP servers (decision M2), run by the host helper.

A third-party server runs in its own container, never inside the application:

- its own internal Docker network (`--internal`): no way out, and no way to the other sidecars;
  a server whose egress label is `lan` or `internet` is also joined to Docker's default bridge,
  so it can go out (Docker cannot tell a LAN from the internet: both labels mean "may go out");
- read-only root filesystem, a small `/tmp` in memory, user `nobody` (65534), every capability
  dropped, no new privileges, memory, CPU and process limits, no volume: it sees none of
  `data/`, `.env` or `models/`;
- an image pinned by digest, from the allow-list MCP_SIDECAR_IMAGES in `.env` (`repo@sha256:…`
  or a local image id `sha256:…`); anything else is refused.

The application reaches it through a forwarder container (python:3.12-slim, the project's own
base image, pinned by digest, standard library only) joined to the sidecar's network and to the
default bridge, published on 127.0.0.1 only. Measured on Docker Desktop (2026-09-27): a container
on an internal network cannot be published, and turning the bridge's masquerade off does not stop
its egress there; the forwarder keeps both properties.

Only the helper runs Docker: the application never gets the Docker socket.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket

from app.admin.jobs import Job, JobError
from app.admin.service import ConflictError, InvalidInputError, NotFoundError

NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
DIGEST_REF = re.compile(r"([a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}|sha256:[0-9a-f]{64})")
EGRESS = ("local", "lan", "internet")
LABEL = "channelagent.sidecar"
DOCKER_TIMEOUT_SECONDS = 600
READY_TIMEOUT_SECONDS = 60

FORWARDER = """
import asyncio, sys
TARGET, PORT = sys.argv[1], int(sys.argv[2])
async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()
async def handle(client_reader, client_writer):
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(TARGET, PORT)
    except OSError:
        client_writer.close()
        return
    await asyncio.gather(pipe(client_reader, upstream_writer),
                         pipe(upstream_reader, client_writer), return_exceptions=True)
async def main():
    server = await asyncio.start_server(handle, "0.0.0.0", PORT)
    async with server:
        await server.serve_forever()
asyncio.run(main())
"""


def allowed_images(value: str) -> list[str]:
    """The pinned images of MCP_SIDECAR_IMAGES (comma separated); a malformed entry is ignored
    by the helper and refused by `./start.sh --config KEY=VALUE`."""
    return [item.strip() for item in value.split(",") if DIGEST_REF.fullmatch(item.strip())]


def names(name: str) -> dict[str, str]:
    return {"server": f"ca-mcp-{name}", "forwarder": f"ca-mcp-{name}-fwd",
            "network": f"ca-mcp-{name}"}  # fmt: skip


def check_request(name: str, image: str, allowed: list[str], egress: str) -> None:
    if not NAME.fullmatch(name or ""):
        raise InvalidInputError("name is lowercase letters, digits and hyphens, up to 40")
    if egress not in EGRESS:
        raise InvalidInputError(f"egress is one of: {', '.join(EGRESS)}")
    if image not in allowed:
        raise InvalidInputError(
            "image is not in MCP_SIDECAR_IMAGES: only images pinned by digest there are run"
        )


def server_argv(name: str, image: str, port: int, memory_mb: int, cpus: float) -> list[str]:
    n = names(name)
    return [
        "docker", "run", "-d", "--name", n["server"], "--label", f"{LABEL}={name}",
        "--network", n["network"], "--network-alias", "mcp",
        "--read-only", "--tmpfs", "/tmp:rw,size=64m,mode=1777",
        "--user", "65534:65534", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--memory", f"{memory_mb}m", "--cpus", str(cpus), "--pids-limit", "128",
        "--env", f"PORT={port}",
        image,
    ]  # fmt: skip


def forwarder_argv(name: str, image: str, port: int, host_port: int) -> list[str]:
    n = names(name)
    return [
        "docker", "run", "-d", "--name", n["forwarder"], "--label", f"{LABEL}={name}",
        "--publish", f"127.0.0.1:{host_port}:{port}",
        "--read-only", "--user", "65534:65534", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--memory", "64m", "--cpus", "0.2",
        image, "python", "-c", FORWARDER, "mcp", str(port),
    ]  # fmt: skip


async def docker(*args: str, timeout: float = DOCKER_TIMEOUT_SECONDS) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )  # fmt: skip
    try:
        out, _ = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise JobError(f"docker {args[0]} did not finish within {int(timeout)} s") from None
    return process.returncode, out.decode(errors="replace").strip()


async def _must(*args: str) -> str:
    code, out = await docker(*args)
    if code != 0:
        raise JobError(f"docker {args[0]} failed: {out.splitlines()[-1][:200] if out else code}")
    return out


def free_loopback_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def wait_ready(host_port: int, timeout: float = READY_TIMEOUT_SECONDS) -> None:
    """Wait until the server answers HTTP through the forwarder. The forwarder accepts a
    connection at once, even before the server listens, so a TCP connect proves nothing: a
    first call right after `docker run` was cut (measured, 2026-09-27)."""
    import httpx

    deadline = asyncio.get_running_loop().time() + timeout
    async with httpx.AsyncClient(timeout=3) as client:
        while True:
            try:
                await client.get(f"http://127.0.0.1:{host_port}/mcp")
                return  # any HTTP answer: the server is up
            except httpx.HTTPError:
                if asyncio.get_running_loop().time() > deadline:
                    raise JobError(f"The server did not answer within {int(timeout)} s") from None
                await asyncio.sleep(0.5)


async def pinned(image: str) -> str:
    """`image` pinned by its digest (pulled first when this machine does not have it)."""
    code, out = await docker("image", "inspect", "--format", "{{json .RepoDigests}}", image)
    if code != 0:
        await _must("pull", image)
        out = await _must("image", "inspect", "--format", "{{json .RepoDigests}}", image)
    digests = json.loads(out or "[]")
    if not digests:
        raise JobError(f"{image} has no registry digest to pin")
    return digests[0]


async def running() -> list[dict]:
    out = await _must(
        "ps", "-a", "--filter", f"label={LABEL}",
        "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Label \"" + LABEL + "\"}}",
    )  # fmt: skip
    rows = []
    for line in out.splitlines():
        container, status, ports, name = (line.split("\t") + ["", "", "", ""])[:4]
        rows.append({"name": name, "container": container, "status": status, "ports": ports})
    return sorted(rows, key=lambda r: r["container"])


async def deploy(
    job: Job, *, name: str, image: str, port: int, egress: str, memory_mb: int, cpus: float,
    forwarder_image: str,
) -> dict:  # fmt: skip
    n = names(name)
    if any(row["container"] in (n["server"], n["forwarder"]) for row in await running()):
        raise ConflictError(f"A sidecar named {name} exists: remove it first")
    job.update(0.1, "Pinning the forwarder image")
    forwarder = await pinned(forwarder_image)
    job.update(0.2, f"Creating the internal network {n['network']}")
    await _must("network", "create", "--internal", "--label", f"{LABEL}={name}", n["network"])
    try:
        job.update(0.4, f"Starting {image}")
        await _must(*server_argv(name, image, port, memory_mb, cpus)[1:])
        if egress != "local":
            await _must("network", "connect", "bridge", n["server"])
        host_port = free_loopback_port()
        job.update(0.7, "Starting the forwarder")
        await _must(*forwarder_argv(name, forwarder, port, host_port)[1:])
        await _must("network", "connect", n["network"], n["forwarder"])
        job.update(0.9, "Waiting for the server to answer")
        await wait_ready(host_port)
    except JobError:
        await remove(name)
        raise
    return {
        "name": name,
        "image": image,
        "egress": egress,
        "forwarder": forwarder,
        "host_port": host_port,
        # The application registers the server with the URL of its own mode.
        "url_native": f"http://127.0.0.1:{host_port}/mcp",
        "url_container": f"http://host.docker.internal:{host_port}/mcp",
    }


async def remove(name: str) -> dict:
    n = names(name)
    removed = []
    for container in (n["forwarder"], n["server"]):
        code, _ = await docker("rm", "-f", container)
        if code == 0:
            removed.append(container)
    await docker("network", "rm", n["network"])
    return {"name": name, "removed": removed}


async def remove_checked(name: str) -> dict:
    if not NAME.fullmatch(name or ""):
        raise InvalidInputError("name is lowercase letters, digits and hyphens, up to 40")
    if not any(row["name"] == name for row in await running()):
        raise NotFoundError(f"No sidecar named {name}")
    return await remove(name)
