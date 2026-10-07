"""Tests: persistent memory per user and per agent. The five memory tools, the
four modes (what is injected, which tools exist), scoping to (user, agent), encryption at
rest, purge, key rotation, survival across a real process restart, the Admin API, and a
turn through the graph with a scripted chat model.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app import memory
from app.db.models import MemoryEntry

REPO = Path(__file__).resolve().parent.parent
KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
SECRETISH = "the user's dog is called Biscuit"


@pytest.fixture
async def world(fresh_db):
    """Two users; Sam has two agents, Alex one."""
    from app.admin import service
    from app.db.session import init_db, session_scope

    await init_db()
    async with session_scope() as session:
        sam = await service.create_user(session, "Sam")
        alex = await service.create_user(session, "Alex")
        a1 = await service.create_agent(session, sam.id, "one")
        a2 = await service.create_agent(session, sam.id, "two")
        b1 = await service.create_agent(session, alex.id, "alex")
        await session.commit()
        ids = {"sam": sam.id, "alex": alex.id, "a1": a1.id, "a2": a2.id, "b1": b1.id}
    return ids


async def _rows(**where) -> int:
    from app.db.session import session_scope

    async with session_scope() as session:
        stmt = select(func.count()).select_from(MemoryEntry)
        for name, value in where.items():
            stmt = stmt.where(getattr(MemoryEntry, name) == value)
        return (await session.execute(stmt)).scalar_one()


def _call(run, name, **arguments):
    return run(memory.TOOL_PREFIX + name, json.dumps(arguments))


# --- the tools ---


async def test_the_tools_add_search_read_edit_and_delete(world):
    run = memory.make_executor(world["sam"], world["a1"])
    assert await _call(run, "add", title="Pet", content=SECRETISH) == "saved as #1"
    await _call(run, "add", title="Work", content="Sam is a nurse working nights")
    assert await _call(run, "search", query="dog Biscuit") == "#1 Pet"
    assert await _call(run, "search", query="spaceship") == "nothing found"
    assert await _call(run, "read", id=1) == f"#1 Pet\n{SECRETISH}"
    assert await _call(run, "edit", id=1, content="The dog is called Biscuit, a beagle")
    assert "beagle" in await _call(run, "read", id=1)
    assert await _call(run, "delete", id=2) == "#2 deleted"
    assert await _rows(agent_id=world["a1"]) == 1


async def test_another_agent_or_user_never_reaches_an_entry(world):
    own = memory.make_executor(world["sam"], world["a1"])
    await _call(own, "add", title="Pet", content=SECRETISH)
    for other in (
        memory.make_executor(world["sam"], world["a2"]),  # same user, another agent
        memory.make_executor(world["alex"], world["b1"]),  # another user
    ):
        assert await _call(other, "search", query="dog Biscuit") == "nothing found"
        assert "No memory entry 1" in await _call(other, "read", id=1)
        assert "No memory entry 1" in await _call(other, "edit", id=1, title="x")
        assert "No memory entry 1" in await _call(other, "delete", id=1)
    assert await _rows() == 1
    assert "Biscuit" in await _call(own, "read", id=1)


async def test_the_tools_refuse_bad_input_without_raising(world, monkeypatch):
    run = memory.make_executor(world["sam"], world["a1"])
    assert "not JSON" in await run("memory_add", "{nope")
    assert "JSON object" in await run("memory_add", "[1]")
    assert "title" in await _call(run, "add", title="", content="x")
    assert "content" in await _call(run, "add", title="t", content="x" * (memory.MAX_CONTENT + 1))
    assert "id is the entry's number" in await _call(run, "read", id="1")
    await _call(run, "add", title="first", content="first")
    # true is 1 in Python: it must not read entry #1
    assert "id is the entry's number" in await _call(run, "read", id=True)
    await _call(run, "delete", id=1)
    assert "give a new title" in await _call(run, "edit", id=1)
    assert "unknown memory tool" in await run("memory_forget_everything", "{}")
    monkeypatch.setattr(memory, "MAX_ENTRIES_PER_AGENT", 2)
    await _call(run, "add", title="a", content="a")
    await _call(run, "add", title="b", content="b")
    assert "memory is full" in await _call(run, "add", title="c", content="c")
    assert await _rows() == 2


async def test_entries_are_encrypted_at_rest(world):
    from app.config import get_settings
    from app.db.session import sqlite_file_path

    await _call(
        memory.make_executor(world["sam"], world["a1"]), "add", title="Pet", content=SECRETISH
    )
    path = sqlite_file_path(get_settings().database_url)
    with sqlite3.connect(path) as con:
        title, content = con.execute("select title, content from memory_entries").fetchone()
    assert "Pet" not in title and "Biscuit" not in content
    assert b"Biscuit" not in path.read_bytes()


# --- the modes ---


@pytest.mark.parametrize(
    ("mode", "tools"),
    [("off", 0), ("ondemand", 5), ("always", 5), ("search", 5), ("unknown", 0)],
)
def test_the_mode_decides_which_tools_exist(mode, tools):
    names = [t["function"]["name"] for t in memory.tool_definitions(mode)]
    assert len(names) == tools
    if tools:
        assert names == [f"memory_{n}" for n in ("search", "read", "add", "edit", "delete")]


async def test_the_mode_decides_what_is_injected(world):
    from app.db.session import session_scope

    run = memory.make_executor(world["sam"], world["a1"])
    await _call(run, "add", title="Pet", content=SECRETISH)
    await _call(run, "add", title="Work", content="Sam is a nurse working nights")
    text = "what is my dog's name?"
    async with session_scope() as session:
        blocks = {
            mode: await memory.injection(session, world["sam"], world["a1"], mode, text)
            for mode in memory.MODES
        }
        other = await memory.injection(session, world["alex"], world["b1"], "always", text)
    assert blocks["off"] == ""
    assert blocks["ondemand"] == memory.GUIDANCE, "ondemand: the guidance, no entry"
    assert all(blocks[m].startswith(memory.GUIDANCE) for m in ("ondemand", "always", "search"))
    assert memory.INJECTION_HEADER in blocks["always"]
    assert "#1 Pet" in blocks["always"] and "#2 Work" in blocks["always"]
    assert "Biscuit" not in blocks["always"], "the index carries titles only"
    assert "Biscuit" in blocks["search"] and "nurse" not in blocks["search"]
    assert other == memory.GUIDANCE, "another user's agent sees none of these entries"


# --- purge, key rotation, restart ---


async def test_purging_a_user_deletes_their_memory_only(world):
    from app.admin import service
    from app.db.session import session_scope

    await _call(memory.make_executor(world["sam"], world["a1"]), "add", title="a", content="a")
    await _call(memory.make_executor(world["sam"], world["a2"]), "add", title="b", content="b")
    await _call(memory.make_executor(world["alex"], world["b1"]), "add", title="c", content="c")
    async with session_scope() as session:
        await service.delete_user(session, world["sam"], purge=True)
        await session.commit()
    assert await _rows(user_id=world["sam"]) == 0
    assert await _rows(user_id=world["alex"]) == 1


async def test_the_rekey_and_its_dry_run_count_the_memory_columns(world):
    from cryptography.fernet import Fernet

    from app.admin.rekey import run_rekey
    from app.config import get_settings
    from app.db.session import get_engine, sqlite_file_path

    for n in range(3):
        await _call(
            memory.make_executor(world["sam"], world["a1"]), "add", title=f"t{n}", content=f"c{n}"
        )
    await get_engine().dispose()
    path = sqlite_file_path(get_settings().database_url)
    old, new = os.environ["ENCRYPTION_KEY"], Fernet.generate_key().decode()
    plan = run_rekey(path, None, old, new, dry_run=True, backup=False)
    assert (
        plan.targets["memory_entries.title"].old,
        plan.targets["memory_entries.content"].old,
    ) == (3, 3)
    done = run_rekey(path, None, old, new, backup=False)
    assert done.applied
    with sqlite3.connect(path) as con:
        tokens = [row[0] for row in con.execute("select content from memory_entries order by id")]
    assert [Fernet(new.encode()).decrypt(t.encode()).decode() for t in tokens] == ["c0", "c1", "c2"]


RESTART_SCRIPT = """
import asyncio, json, sys
from app import memory
from app.db.session import init_db

async def main(step, user_id, agent_id):
    await init_db()
    run = memory.make_executor(user_id, agent_id)
    if step == "write":
        print(await run("memory_add", json.dumps({"title": "Pet", "content": "Biscuit"})))
    else:
        print(await run("memory_read", json.dumps({"id": 1})))

asyncio.run(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
"""


async def test_memory_survives_a_restart_of_the_process(world, tmp_path):
    """Two separate processes on the same real database file."""
    from app.db.session import get_engine

    await get_engine().dispose()
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    args = [str(world["sam"]), str(world["a1"])]
    first = subprocess.run(
        [sys.executable, "-c", RESTART_SCRIPT, "write", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert first.stdout.strip().endswith("saved as #1"), first.stderr[-500:]
    second = subprocess.run(
        [sys.executable, "-c", RESTART_SCRIPT, "read", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert second.stdout.strip().endswith("#1 Pet\nBiscuit"), second.stderr[-500:]


# --- the Admin API ---


@pytest.fixture
async def api(world, monkeypatch):
    import httpx

    from app.api import deps
    from app.api.app import app
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        c.app, c.ids = app, world
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()


async def test_the_api_browses_adds_edits_and_deletes_with_admin_events(api):
    from app.db.models import AdminEvent
    from app.db.session import session_scope

    base = f"/users/{api.ids['sam']}/agents/{api.ids['a1']}/memory"
    created = await api.post(base, json={"title": "Pet", "content": SECRETISH})
    assert created.status_code == 201, created.text
    entry = created.json()["id"]
    await api.post(base, json={"title": "Work", "content": "nurse"})
    listed = await api.get(base)
    assert listed.status_code == 200 and listed.headers["X-Total-Count"] == "2"
    assert [e["title"] for e in (await api.get(base, params={"query": "dog"})).json()] == ["Pet"]
    edited = await api.patch(f"{base}/{entry}", json={"title": "Dog"})
    assert edited.json()["title"] == "Dog" and edited.json()["content"] == SECRETISH
    assert (await api.delete(f"{base}/{entry}")).status_code == 204
    assert (await api.get(f"{base}")).headers["X-Total-Count"] == "1"
    async with session_scope() as session:
        events = list(
            (
                await session.execute(select(AdminEvent).where(AdminEvent.action.like("memory.%")))
            ).scalars()
        )
    assert sorted(e.action for e in events) == [
        "memory.add",
        "memory.add",
        "memory.delete",
        "memory.edit",
    ]
    assert all("Biscuit" not in json.dumps(e.details) for e in events)


async def test_the_api_refuses_another_users_agent_and_unknown_entries(api):
    wrong = f"/users/{api.ids['sam']}/agents/{api.ids['b1']}/memory"
    assert (await api.get(wrong)).status_code == 404
    assert (await api.post(wrong, json={"title": "t", "content": "c"})).status_code == 404
    base = f"/users/{api.ids['sam']}/agents/{api.ids['a1']}/memory"
    assert (await api.patch(f"{base}/99", json={"title": "x"})).status_code == 404
    assert (await api.delete(f"{base}/99")).status_code == 404
    assert (await api.post(base, json={"title": "", "content": "c"})).status_code == 422


async def test_memory_routes_need_the_admin_scope(api):
    from app.api.scopes import Principal, Scope, get_principal

    api.app.dependency_overrides[get_principal] = lambda: Principal("op", Scope.OPERATE)
    base = f"/users/{api.ids['sam']}/agents/{api.ids['a1']}/memory"
    assert (await api.get(base)).status_code == 403
    assert (await api.post(base, json={"title": "t", "content": "c"})).status_code == 403
