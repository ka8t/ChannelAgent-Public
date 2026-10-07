"""Tests: importing a local GGUF file into the models directory, as a job."""

import asyncio
import hashlib
import time

import httpx
import pytest

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}


def _model(path, size=3 * 1024 * 1024, seed=1):
    data = b"GGUF" + bytes((seed * i) % 251 for i in range(4096)) * (size // 4096)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
async def client(fresh_db, monkeypatch, tmp_path):
    from app.admin import models
    from app.admin.jobs import registry
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    directory = tmp_path / "models"
    directory.mkdir()
    monkeypatch.setenv("MODELS_DIR", str(directory))
    monkeypatch.setenv("MODEL_FILE", "")
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    registry.clear()
    models._active.clear()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        c.dir, c.src = directory, tmp_path
        yield c
    registry.clear()
    models._active.clear()
    deps.reset_failure_state()


async def _wait(client, job_id, tries=200):
    for _ in range(tries):
        job = (await client.get(f"/jobs/{job_id}")).json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.03)
    raise AssertionError(job)


async def _import(client, **body):
    return await client.post("/models/import", json=body)


async def test_an_import_copies_the_file_and_records_size_and_sha256(client):
    digest = _model(client.src / "tiny.gguf")
    started = await _import(client, path=str(client.src / "tiny.gguf"))
    assert started.status_code == 202
    job = await _wait(client, started.json()["id"])
    assert job["status"] == "done" and job["progress"] == 1.0
    size = (client.src / "tiny.gguf").stat().st_size
    assert job["result"] == {"name": "tiny.gguf", "size_bytes": size, "sha256": digest}
    copy = client.dir / "tiny.gguf"
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == digest
    assert (client.dir / "tiny.gguf.sha256").read_text().split()[0] == digest
    assert sorted(p.name for p in client.dir.iterdir()) == ["tiny.gguf", "tiny.gguf.sha256"]
    listed = (await client.get("/models")).json()
    assert listed[0]["sha256"] == digest
    events = (await client.get("/admin-events", params={"action": "model.import"})).json()
    assert len(events) == 1 and digest in str(events[0]["details"])


async def test_a_file_that_is_not_gguf_is_refused_and_nothing_is_added(client):
    (client.src / "secrets.env").write_text("ENCRYPTION_KEY=abc\n")
    response = await _import(client, path=str(client.src / "secrets.env"), name="x.gguf")
    assert response.status_code == 422 and "not a GGUF" in response.json()["detail"]
    assert list(client.dir.iterdir()) == []


async def test_a_directory_or_a_missing_file_is_refused(client):
    assert (await _import(client, path=str(client.src))).status_code == 422
    assert (await _import(client, path=str(client.src / "missing.gguf"))).status_code == 422


async def test_a_blob_without_a_model_name_needs_one_and_a_bad_name_is_refused(client):
    _model(client.src / "sha256-0123abcd")  # an Ollama blob has no extension
    path = str(client.src / "sha256-0123abcd")
    unnamed = await _import(client, path=path)
    assert unnamed.status_code == 422 and "give the model a name" in unnamed.json()["detail"]
    assert (await _import(client, path=path, name="a;b.gguf")).status_code == 422
    assert (await _import(client, path=path, name="../x.gguf")).status_code == 422
    ok = await _import(client, path=path, name="blob-model.gguf")
    assert (await _wait(client, ok.json()["id"]))["status"] == "done"


async def test_an_existing_name_is_refused_unless_forced(client):
    _model(client.src / "m.gguf", seed=1)
    first = _model(client.src / "m.gguf", seed=1)
    await _wait(client, (await _import(client, path=str(client.src / "m.gguf"))).json()["id"])
    second = _model(client.src / "m.gguf", seed=7)
    assert second != first
    assert (await _import(client, path=str(client.src / "m.gguf"))).status_code == 409
    forced = await _import(client, path=str(client.src / "m.gguf"), force=True)
    job = await _wait(client, forced.json()["id"])
    assert job["status"] == "done" and job["result"]["sha256"] == second


async def test_not_enough_free_space_is_refused_up_front(client, monkeypatch):
    from app.admin import models

    _model(client.src / "big.gguf")
    monkeypatch.setattr(models, "free_bytes", lambda _d: 1024)
    response = await _import(client, path=str(client.src / "big.gguf"))
    assert response.status_code == 409 and "free disk space" in response.json()["detail"]
    assert list(client.dir.iterdir()) == []


async def test_a_copy_cancelled_half_way_leaves_no_partial_file(client, monkeypatch):
    from app.admin import models

    _model(client.src / "slow.gguf", size=4 * 1024 * 1024)
    real = models._copy_chunk
    monkeypatch.setattr(models, "COPY_CHUNK", 512 * 1024)

    def slow(src, dst, digest):
        time.sleep(0.05)
        return real(src, dst, digest)

    monkeypatch.setattr(models, "_copy_chunk", slow)
    started = (await _import(client, path=str(client.src / "slow.gguf"))).json()
    seen = 0.0
    for _ in range(100):
        seen = (await client.get(f"/jobs/{started['id']}")).json()["progress"] or 0.0
        if seen >= 0.3:
            break
        await asyncio.sleep(0.03)
    assert 0.3 <= seen < 1.0
    # while it copies, the model exists only as a partial file: never under its final name
    assert not (client.dir / "slow.gguf").exists() and (client.dir / "slow.gguf.part").exists()
    assert (await client.post(f"/jobs/{started['id']}/cancel")).status_code == 200
    assert (await _wait(client, started["id"]))["status"] == "cancelled"
    leftovers = [p.name for p in client.dir.iterdir()]
    print(f"after cancel at {seen:.2f}: {len(leftovers)} files in the models directory")
    assert leftovers == []
    # and the name is free again
    monkeypatch.setattr(models, "_copy_chunk", real)
    again = await _import(client, path=str(client.src / "slow.gguf"))
    assert (await _wait(client, again.json()["id"]))["status"] == "done"


async def test_a_copy_that_fails_leaves_no_partial_file_and_a_safe_error(client, monkeypatch):
    from app.admin import models

    _model(client.src / "bad.gguf")

    def broken(*_a):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(models, "_copy_chunk", broken)
    job = await _wait(
        client, (await _import(client, path=str(client.src / "bad.gguf"))).json()["id"]
    )
    assert job["status"] == "failed" and "No space left" in job["error"]
    assert list(client.dir.iterdir()) == []


async def test_two_imports_of_one_name_at_once_are_refused(client, monkeypatch):
    from app.admin import models

    _model(client.src / "twice.gguf")
    real = models._copy_chunk
    monkeypatch.setattr(models, "_copy_chunk", lambda s, d, h: (time.sleep(0.2), real(s, d, h))[1])
    first = await _import(client, path=str(client.src / "twice.gguf"))
    second = await _import(client, path=str(client.src / "twice.gguf"), force=True)
    assert first.status_code == 202 and second.status_code == 409
    await _wait(client, first.json()["id"])


async def test_without_a_models_directory_the_answer_is_409(client, monkeypatch):
    from app.config import get_settings

    _model(client.src / "x.gguf")
    monkeypatch.setenv("MODELS_DIR", str(client.src / "missing"))
    get_settings.cache_clear()
    assert (await _import(client, path=str(client.src / "x.gguf"))).status_code == 409


async def test_a_source_that_changes_size_during_the_copy_is_not_kept(tmp_path, monkeypatch):
    from app.admin import models

    src = tmp_path / "grow.gguf"
    _model(src)
    directory = tmp_path / "models"
    directory.mkdir()

    class Job:
        def update(self, *_a):
            pass

    plan = {
        "src": src,
        "name": "grow.gguf",
        "dest": directory / "grow.gguf",
        "size": src.stat().st_size + 5,
    }
    with pytest.raises(OSError, match="changed size"):
        await models.run_import(Job(), plan)
    assert list(directory.iterdir()) == []
