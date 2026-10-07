"""A conversation's engine cache saved on disk, so a turn after an engine restart, or
after another conversation took its slot, does not read the whole conversation again.

llama-server keeps what it has read of a prompt in a slot (one per parallel request) and can
save a slot to a file (`--slot-save-path`, set by start.sh) and restore it. The owner's model
is hybrid (recurrent layers): a restored state cannot be rolled back, not even by one token, so
it is reused only when the next prompt *extends* it (measured on the real
engine: 1,683 of 1,695 tokens from the cache, 0.87 s instead
of 26.1 s). What is saved is therefore never the reply the engine generated, which the history
renders differently, but the stable prefix of the turn: the longest common token prefix of
this turn's prompt and of the same conversation without this turn's memory hits,
followed by an assistant reply and a new user message (a trailing assistant message would be
rendered as a reply to continue, generation prompt included). The engine renders and
tokenizes both (`/apply-template`, `/tokenize`), so no chat template is written here.

Each conversation turn is pinned to one slot (`id_slot`) by an in-process allocator, least
recently used first, one turn per slot at a time. Before the turn's main completion:
- the thread's file is restored when this process does not know the thread to be in its slot,
  and the file still is a prefix of this turn (its length and digest are kept next to it);
- the stable prefix is read (`n_predict` 0) and the slot saved when it grew by `SAVE_STEP`
  tokens or more since the last save, or when there is no usable file.

Files: `<slot dir>/<model key>-<sha256(thread id)>.bin` and its `.json` sidecar. The model key
comes from the engine (`/v1/models`): another model ignores and deletes them. They hold the
engine state of a conversation, not encrypted (stated on): mode 600 (start.sh starts the
engine with umask 077), `SLOT_CACHE_MAX_MB` caps them (oldest deleted first), deleting a
thread deletes its files (`forget`), backups copy only `*.db`. Router mode (several models) or
an engine without the slot API: off for this process, one log line, the turn goes on.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from app.config import get_settings

logger = logging.getLogger("channelagent")

SAVE_STEP = 1024  # tokens the stable prefix grows by before it is saved again
MIN_TOKENS = 512  # shorter conversations are read again in seconds: not saved

_state: dict = {}  # "engine": (model key, slot count) or None (off); see _engine()
_locks: list[asyncio.Lock] = []
_owner: dict[int, str] = {}  # slot -> thread id whose prefix this process put there
_used: dict[int, float] = {}  # slot -> last time a turn took it


def slot_dir() -> Path:
    """`slots/` next to the main SQLite database (`data/slots`), where start.sh points the
    engine's `--slot-save-path`."""
    from app.db.session import sqlite_file_path

    main = sqlite_file_path(get_settings().database_url)
    return (main.parent if main is not None else Path("data")) / "slots"


def _thread_key(thread_id: str) -> str:
    return hashlib.sha256(thread_id.encode()).hexdigest()[:32]


def _digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def reset() -> None:
    """Forget the engine and the slots (tests, and an engine that went away)."""
    _state.clear()
    _locks.clear()
    _owner.clear()
    _used.clear()


async def _engine(client: httpx.AsyncClient) -> tuple[str, int] | None:
    """(model key, slot count), or None when the cache is off for this process."""
    if "engine" in _state:
        return _state["engine"]
    engine = None
    if get_settings().slot_cache_max_mb > 0:
        try:
            models = (await client.get("/v1/models")).json().get("data") or []
            props = (await client.get("/props")).json()
            slots = int(props.get("total_slots") or 0)
            if len(models) != 1 or slots < 1:
                logger.info("Slot cache off: %d models on the engine (router mode)", len(models))
            else:
                meta = models[0].get("meta") or {}
                identity = f"{models[0].get('id')}|{meta.get('size')}|{meta.get('n_params')}"
                key = hashlib.sha256(identity.encode()).hexdigest()[:12]
                engine = (key, slots)
                _discard_other_models(key)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            logger.info("Slot cache off: the engine did not describe its model and slots")
    _state["engine"] = engine
    if engine is not None:
        _locks[:] = [asyncio.Lock() for _ in range(engine[1])]
    return engine


def _discard_other_models(key: str) -> None:
    folder = slot_dir()
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.suffix in (".bin", ".json") and not path.name.startswith(f"{key}-"):
            path.unlink(missing_ok=True)


def _name(key: str, thread_id: str) -> str:
    return f"{key}-{_thread_key(thread_id)}.bin"


def _read_meta(name: str) -> dict | None:
    try:
        meta = json.loads((slot_dir() / name).with_suffix(".json").read_text())
    except (OSError, ValueError):
        return None
    if not (slot_dir() / name).is_file():
        return None
    return meta if isinstance(meta, dict) and isinstance(meta.get("n"), int) else None


def _write_meta(name: str, tokens: list[int]) -> None:
    path = (slot_dir() / name).with_suffix(".json")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"n": len(tokens), "digest": _digest(tokens)}, f)


def _cap_disk(keep: str) -> None:
    """Delete the oldest files above SLOT_CACHE_MAX_MB, never the one just saved."""
    limit = get_settings().slot_cache_max_mb * 2**20
    files = sorted(slot_dir().glob("*.bin"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    for path in files:
        if total <= limit:
            break
        if path.name == keep:
            continue
        total -= path.stat().st_size
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)


async def _tokens(client: httpx.AsyncClient, body: dict) -> list[int]:
    rendered = await client.post("/apply-template", json=body)
    rendered.raise_for_status()
    response = await client.post(
        "/tokenize", json={"content": rendered.json()["prompt"], "add_special": True}
    )
    response.raise_for_status()
    return response.json()["tokens"]


def _common(a: list[int], b: list[int]) -> list[int]:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return a[:n]


async def _acquire(thread_id: str) -> int:
    """The slot this thread's turn runs in, locked for the turn."""
    mine = [s for s, t in _owner.items() if t == thread_id]
    if mine:
        slot = mine[0]
    else:
        free = [s for s in range(len(_locks)) if not _locks[s].locked()]
        candidates = free or list(range(len(_locks)))
        slot = min(candidates, key=lambda s: _used.get(s, 0.0))
    await _locks[slot].acquire()
    _used[slot] = time.monotonic()
    return slot


async def _prepare(
    client: httpx.AsyncClient, key: str, slot: int, thread_id: str, body: dict, stable: dict
) -> None:
    name = _name(key, thread_id)
    prefix = _common(await _tokens(client, body), await _tokens(client, stable))
    meta = _read_meta(name)
    usable = (
        meta is not None
        and len(prefix) >= meta["n"]
        and _digest(prefix[: meta["n"]]) == meta.get("digest")
    )
    if usable and _owner.get(slot) != thread_id:
        response = await client.post(
            f"/slots/{slot}", params={"action": "restore"}, json={"filename": name}
        )
        response.raise_for_status()
    _owner[slot] = thread_id
    grown = len(prefix) - (meta["n"] if usable else 0)
    if len(prefix) < MIN_TOKENS or (usable and grown < SAVE_STEP):
        return
    read = await client.post(
        "/completion",
        json={"prompt": prefix, "n_predict": 0, "id_slot": slot, "cache_prompt": True},
    )
    read.raise_for_status()
    saved = await client.post(
        f"/slots/{slot}", params={"action": "save"}, json={"filename": name}
    )
    saved.raise_for_status()
    _write_meta(name, prefix)
    _cap_disk(keep=name)
    logger.info("Slot cache: %d tokens of the conversation saved", len(prefix))


@asynccontextmanager
async def pinned(client: httpx.AsyncClient, thread_id: str | None, body: dict, stable: dict):
    """Yields the request fields that pin this turn to its slot (`{"id_slot": n}`), or `{}`
    when the cache is off. `body` is the turn's request, `stable` the same conversation
    without this turn's memory hits, then an assistant reply and a new user message."""
    engine = await _engine(client) if thread_id else None
    if engine is None:
        yield {}
        return
    slot = await _acquire(thread_id)
    try:
        try:
            await _prepare(client, engine[0], slot, thread_id, body, stable)
        except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
            # The turn goes on unpinned from the cache's point of view: what the slot holds
            # is unknown, so it is not recorded as this thread's.
            _owner.pop(slot, None)
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 501):
                logger.warning("Slot cache off: the engine refused the slot API (%s)", exc)
                _state["engine"] = None
            else:
                logger.warning("Slot cache skipped for this turn: %s", type(exc).__name__)
        yield {"id_slot": slot}
    finally:
        _locks[slot].release()


async def forget(client: httpx.AsyncClient | None, thread_ids: Iterable[str]) -> int:
    """Delete these threads' files (every model) and erase their slots in the engine.
    Returns how many files were deleted."""
    keys = {_thread_key(t): t for t in thread_ids}
    deleted = 0
    folder = slot_dir()
    if folder.is_dir():
        for path in folder.iterdir():
            stem = path.stem.split("-", 1)[-1]
            if stem in keys and path.suffix in (".bin", ".json"):
                path.unlink(missing_ok=True)
                deleted += path.suffix == ".bin"
    for slot, thread_id in list(_owner.items()):
        if thread_id in keys.values():
            _owner.pop(slot, None)
            if client is not None:
                try:
                    await client.post(f"/slots/{slot}", params={"action": "erase"})
                except httpx.HTTPError:
                    logger.warning("Slot cache: the engine did not erase slot %d", slot)
    return deleted
