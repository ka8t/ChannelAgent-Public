"""Models on disk: names, paths, listing and deletion. The import and the pull, which
are jobs, live in the same module.

A model is a `.gguf` file in MODELS_DIR, with an optional `<name>.sha256` next to it that says
what the file hashed to when it was imported or pulled. A name is an allow-list, never a
path: it cannot leave the directory, and it cannot carry anything a shell would read.
"""

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from app.admin.service import ConflictError, InvalidInputError, NotFoundError
from app.config import get_settings

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf")
GGUF_MAGIC = b"GGUF"
CHUNK = 1024 * 1024


def valid_name(name: str) -> str:
    if not NAME_RE.fullmatch(name or "") or ".." in name or len(name) > 200:
        raise InvalidInputError(
            "A model name is letters, digits, dots, hyphens and underscores, ends in .gguf, "
            "and is a file name, not a path"
        )
    return name


def models_dir() -> Path:
    directory = Path(get_settings().models_dir)
    if not directory.is_dir():
        raise ConflictError(
            "There is no models directory here: models are managed on the host "
            "(./start.sh --admin list-models)"
        )
    return directory.resolve()


def model_path(name: str, directory: Path | None = None) -> Path:
    directory = directory or models_dir()
    path = directory / valid_name(name)
    # A link inside the directory that points elsewhere would make the operation touch the
    # target: the file is judged where it is, never through a link.
    if path.is_symlink() or path.resolve().parent != directory:
        raise InvalidInputError("A model is a regular file of the models directory")
    return path


def sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def read_sha256(path: Path) -> str | None:
    try:
        text = sidecar(path).read_text().split()[0]
    except (OSError, IndexError):
        return None
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else None


def write_sha256(path: Path, digest: str) -> None:
    sidecar(path).write_text(f"{digest}  {path.name}\n")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def list_installed() -> list[dict]:
    """Every `.gguf` in the directory: name, size, recorded SHA256 (None when none was
    recorded: hashing several GB on every listing would be the price of knowing), mtime."""
    result = []
    for path in sorted(models_dir().glob("*.gguf")):
        if path.is_symlink() or not path.is_file() or not NAME_RE.fullmatch(path.name):
            continue
        stat = path.stat()
        result.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "sha256": read_sha256(path),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC),
            }
        )
    return result


def delete_model(name: str, loaded: str | None) -> dict:
    """Remove a model and its recorded hash. The model the engine has loaded and the one
    named by MODEL_FILE are refused: the first is in use, the second is what the next start
    of the engine loads."""
    path = model_path(name)
    if not path.is_file():
        raise NotFoundError(f"No model named {name}")
    if loaded == name:
        raise ConflictError(f"{name} is loaded by the engine")
    if get_settings().model_file == name:
        raise ConflictError(f"{name} is the model the engine starts with (MODEL_FILE)")
    size = path.stat().st_size
    path.unlink()
    sidecar(path).unlink(missing_ok=True)
    return {"name": name, "size_bytes": size}


def free_bytes(directory: Path) -> int:
    stat = os.statvfs(directory)
    return stat.f_bavail * stat.f_frsize


# --- import from a local file ---

_active: set[str] = set()  # names being imported or pulled: one job per name
FREE_SPACE_MARGIN = 100 * 1024 * 1024
COPY_CHUNK = 8 * CHUNK


def claim(name: str) -> None:
    if name in _active:
        raise ConflictError(f"{name} is already being imported or pulled")
    _active.add(name)


def release(name: str) -> None:
    _active.discard(name)


def part_path(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def prepare_import(source: str, name: str | None, force: bool) -> dict:
    """Everything that can be refused before a job starts: the source is a regular GGUF file,
    the name is valid and free (or `force`), and the disk has room. Returns the plan."""
    src = Path(source).expanduser()
    if not src.is_file():
        raise InvalidInputError("The source is not a file")
    with open(src, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise InvalidInputError("The source is not a GGUF model file")
    chosen = name or src.name
    if name is None and not NAME_RE.fullmatch(chosen):
        raise InvalidInputError("The source name is not a model name: give the model a name")
    directory = models_dir()
    dest = model_path(chosen, directory)
    if (dest.exists() or dest.is_symlink()) and not force:
        raise ConflictError(f"{chosen} already exists (use force to replace it)")
    size = src.stat().st_size
    if free_bytes(directory) < size + FREE_SPACE_MARGIN:
        raise ConflictError("There is not enough free disk space for this model")
    return {"src": src, "name": chosen, "dest": dest, "size": size}


def _copy_chunk(src, dst, digest) -> int:
    chunk = src.read(COPY_CHUNK)
    if chunk:
        digest.update(chunk)
        dst.write(chunk)
    return len(chunk)


async def run_import(job, plan: dict) -> dict:
    """Copy the file in chunks with a running SHA256 into `<name>.part`, then rename it into
    place. A cancelled or failed copy leaves no `.part`."""
    import asyncio

    dest, part, size = plan["dest"], part_path(plan["dest"]), plan["size"]
    digest, copied, done = hashlib.sha256(), 0, False
    try:
        job.update(0.0, "Copying")
        with open(plan["src"], "rb") as src, open(part, "wb") as dst:
            while True:
                n = await asyncio.to_thread(_copy_chunk, src, dst, digest)
                if n == 0:
                    break
                copied += n
                job.update(copied / size if size else 1.0)
            dst.flush()
            os.fsync(dst.fileno())
        if copied != size:
            raise OSError("the source changed size while it was copied")
        os.replace(part, dest)
        done = True
    finally:
        if not done:
            part.unlink(missing_ok=True)
    write_sha256(dest, digest.hexdigest())
    return {"name": plan["name"], "size_bytes": copied, "sha256": digest.hexdigest()}


# --- pull from a hub or a URL ---

import asyncio  # noqa: E402
from urllib.parse import quote, urljoin, urlsplit  # noqa: E402

import httpx  # noqa: E402

from app.admin.jobs import JobError  # noqa: E402
from app.security.outbound import Guard, OutboundError, guard_from_settings  # noqa: E402

HUB_SPEC = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+(?::[\w.-]+)?")
SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)
MAX_REDIRECTS = 5
_REDIRECTS = {301, 302, 303, 307, 308}


def build_guard() -> Guard:
    return guard_from_settings()


def build_client() -> httpx.AsyncClient:
    """One client for a pull. Redirects are followed by hand, one guarded hop at a time."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, read=120.0), follow_redirects=False, trust_env=False
    )


def parse_spec(spec: str) -> dict:
    spec = (spec or "").strip()
    if spec.startswith(("https://", "http://")):
        return {"kind": "url", "url": spec}
    if HUB_SPEC.fullmatch(spec):
        repo, _, quant = spec.partition(":")
        return {"kind": "hub", "repo": repo, "quant": quant or None}
    raise InvalidInputError("A model to pull is <repo>[:quant] or an https URL")


async def prepare_pull(spec: str, name: str | None, sha256: str | None, force: bool) -> dict:
    """What can be refused before a job starts: the spec, the name, an existing file, and for
    a URL the guard (a host that is not allowed never becomes a job)."""
    plan = parse_spec(spec)
    plan.update(sha256=sha256, force=force, name=name)
    directory = models_dir()
    if plan["kind"] == "url":
        try:
            await asyncio.to_thread(build_guard().prepare, plan["url"])
        except OutboundError as exc:
            raise InvalidInputError(str(exc)) from None
        plan["name"] = name or Path(urlsplit(plan["url"]).path).name
        if not NAME_RE.fullmatch(plan["name"]):
            raise InvalidInputError("The URL does not end in a model name: give the model a name")
    if plan["name"]:
        dest = model_path(plan["name"], directory)
        if (dest.exists() or dest.is_symlink()) and not force:
            raise ConflictError(f"{plan['name']} already exists (use force to replace it)")
    return plan


def _expected_hash(*candidates: str | None) -> str | None:
    for value in candidates:
        if value:
            value = value.strip().strip('"').lower().removeprefix("w/").strip('"')
            if re.fullmatch(r"[0-9a-f]{64}", value):
                return value
    return None


async def _fetch(client, guard, url, *, token, hub_host, start=0):
    """GET `url` as a stream, following redirects one hop at a time through the guard. The
    token goes to the hub host only, never on to another host. Returns (response, headers
    seen on any hop): the hub states the file's SHA256 on its first answer."""
    seen: dict = {}
    for _ in range(MAX_REDIRECTS + 1):
        try:
            target = await asyncio.to_thread(guard.prepare, url)
        except OutboundError as exc:
            raise JobError(str(exc)) from None
        host = (urlsplit(url).hostname or "").lower()
        headers = {"Host": target.host_header, "User-Agent": "channelagent-model-pull"}
        if start:
            headers["Range"] = f"bytes={start}-"
        if token and host == hub_host:
            headers["Authorization"] = f"Bearer {token}"
        extensions = {"sni_hostname": target.sni} if target.sni else {}
        request = client.build_request("GET", target.url, headers=headers, extensions=extensions)
        response = await client.send(request, stream=True)
        for key in ("x-linked-etag", "x-linked-size"):
            if response.headers.get(key):
                seen[key] = response.headers[key]
        if response.status_code in _REDIRECTS:
            location = response.headers.get("location")
            await response.aclose()
            if not location:
                raise JobError("The server redirected without saying where")
            url = urljoin(url, location)
            continue
        return response, seen
    raise JobError("Too many redirects")


async def _hub_file(client, guard, hub_url, repo, quant, token) -> str:
    """The one `.gguf` of `repo` that matches `quant`, asked of the hub's API."""
    hub_host = (urlsplit(hub_url).hostname or "").lower()
    api = f"{hub_url.rstrip('/')}/api/models/{quote(repo, safe='/')}"
    response, _ = await _fetch(client, guard, api, token=token, hub_host=hub_host)
    try:
        if response.status_code == 404:
            raise JobError(f"The hub has no repository {repo}")
        if response.status_code != 200:
            raise JobError(f"The hub answered {response.status_code} for {repo}")
        body = await response.aread()
    finally:
        await response.aclose()
    try:
        files = [s["rfilename"] for s in json.loads(body).get("siblings", [])]
    except (ValueError, KeyError, AttributeError):
        raise JobError("The hub's answer could not be read") from None
    ggufs = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    if quant:
        token_re = re.compile(rf"(?<![a-z0-9]){re.escape(quant.lower())}(?![a-z0-9])")
        ggufs = [f for f in ggufs if token_re.search(f.lower())]
    if not ggufs:
        raise JobError(f"{repo} has no .gguf file" + (f" for {quant}" if quant else ""))
    if len(ggufs) > 1:
        shown = ", ".join(sorted(ggufs)[:10])
        raise JobError(f"{repo} has several matching files, give a quantization: {shown}")
    if SHARD.search(ggufs[0]):
        raise JobError(f"{ggufs[0]} is a sharded model, which is not supported")
    return f"{hub_url.rstrip('/')}/{quote(repo, safe='/')}/resolve/main/{quote(ggufs[0])}"


async def run_pull(job, plan: dict) -> dict:
    """Download into `<name>.part` (continued when one is there), check the SHA256, rename
    it into place. Refusals and limits leave nothing; an interrupted download keeps its
    `.part` so the same pull resumes."""
    settings = get_settings()
    guard, hub_url = build_guard(), settings.model_hub_url
    hub_host = (urlsplit(hub_url).hostname or "").lower()
    token = settings.hf_token or None
    try:
        async with asyncio.timeout(settings.model_pull_timeout_seconds):
            async with build_client() as client:
                job.update(0.0, "Resolving")
                if plan["kind"] == "hub":
                    url = await _hub_file(
                        client, guard, hub_url, plan["repo"], plan["quant"], token
                    )
                else:
                    url = plan["url"]
                name = plan["name"] or Path(urlsplit(url).path).name
                if not NAME_RE.fullmatch(name):
                    raise JobError("The file has no usable model name: give the model a name")
                directory = models_dir()
                dest = model_path(name, directory)
                if (dest.exists() or dest.is_symlink()) and not plan["force"]:
                    raise JobError(f"{name} already exists (use force to replace it)")
                claim(name)
                try:
                    return await _download(
                        job, client, guard, plan, url, name, dest, token, hub_host
                    )
                finally:
                    release(name)
    except TimeoutError:
        raise JobError(
            "The pull passed its time limit (MODEL_PULL_TIMEOUT_SECONDS); run it again to resume"
        ) from None


async def _download(job, client, guard, plan, url, name, dest, token, hub_host) -> dict:
    settings = get_settings()
    part = part_path(dest)
    offset = part.stat().st_size if part.is_file() else 0
    job.update(0.02, f"Downloading {name}")
    response, seen = await _fetch(client, guard, url, token=token, hub_host=hub_host, start=offset)
    try:
        status_code = response.status_code
        if status_code == 200 and offset:
            offset = 0  # the server ignored the range: start again
        elif status_code == 206:
            content_range = response.headers.get("content-range", "")
            match = re.fullmatch(r"bytes (\d+)-\d+/(\d+)", content_range)
            if not match or int(match.group(1)) != offset:
                raise JobError("The server answered a range that does not continue the file")
        elif status_code != 200:
            raise JobError(f"The server answered {status_code}")
        if status_code == 206:
            total = int(match.group(2))
        else:
            length = response.headers.get("content-length")
            linked = seen.get("x-linked-size", "0")
            total = int(length) if length and length.isdigit() else int(linked or 0)
        if total and total > settings.model_pull_max_bytes:
            raise JobError(
                f"The file is {total} bytes, over the limit MODEL_PULL_MAX_BYTES "
                f"({settings.model_pull_max_bytes})"
            )
        directory = dest.parent
        if total and free_bytes(directory) < total - offset + FREE_SPACE_MARGIN:
            raise JobError("There is not enough free disk space for this model")
        received, over = offset, False
        try:
            with open(part, "ab" if offset else "wb") as out:
                async for chunk in response.aiter_bytes(CHUNK):
                    received += len(chunk)
                    if received > settings.model_pull_max_bytes:
                        over = True
                        break
                    out.write(chunk)
                    if total:
                        job.update(received / total)
                out.flush()
                os.fsync(out.fileno())
        except httpx.HTTPError:
            raise JobError(
                f"The download was interrupted at {received} bytes; run the same pull to resume"
            ) from None
        if over:  # a file over the limit is not kept, however far it got
            part.unlink(missing_ok=True)
            raise JobError("The download went over MODEL_PULL_MAX_BYTES; file removed")
    finally:
        await response.aclose()
    if total and received != total:
        raise JobError(
            f"The download stopped at {received} of {total} bytes; run it again to resume"
        )
    job.update(0.98, "Checking")
    digest = await asyncio.to_thread(sha256_of, part)
    expected = _expected_hash(plan.get("sha256"), seen.get("x-linked-etag"))
    if expected and digest != expected:
        part.unlink(missing_ok=True)
        raise JobError("The SHA256 of the download does not match the expected value; file removed")
    os.replace(part, dest)
    write_sha256(dest, digest)
    return {
        "name": name,
        "size_bytes": received,
        "sha256": digest,
        "verified": expected is not None,
        "source": urlsplit(url).hostname,
    }
