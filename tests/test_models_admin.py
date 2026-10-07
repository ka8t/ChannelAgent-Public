"""Tests: listing and deleting the models in the models directory."""

import httpx
import pytest

from app.api.schemas import EngineStatus

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}
DIGEST = "a" * 64


def _gguf(path, size=64):
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))


@pytest.fixture
async def client(fresh_db, monkeypatch, tmp_path):
    from app.api import deps, models_routes
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    directory = tmp_path / "models"
    directory.mkdir()
    _gguf(directory / "loaded.gguf", 100)
    _gguf(directory / "configured.gguf", 200)
    _gguf(directory / "other.gguf", 300)
    (directory / "other.gguf.sha256").write_text(f"{DIGEST}  other.gguf\n")
    (directory / "notes.txt").write_text("not a model")
    (directory / "half.gguf.part").write_bytes(b"GGUF")
    monkeypatch.setenv("MODELS_DIR", str(directory))
    monkeypatch.setenv("MODEL_FILE", "configured.gguf")
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    state = {"engine": EngineStatus(reachable=True, model="loaded.gguf", n_ctx=4096)}

    async def engine():
        return state["engine"]

    monkeypatch.setattr(models_routes, "engine_status", engine)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        c.dir, c.state = directory, state
        yield c
    deps.reset_failure_state()


async def test_the_list_shows_size_hash_loaded_and_configured(client):
    response = await client.get("/models")
    assert response.status_code == 200
    by_name = {m["name"]: m for m in response.json()}
    assert sorted(by_name) == ["configured.gguf", "loaded.gguf", "other.gguf"]  # no .txt, no .part
    assert by_name["loaded.gguf"]["size_bytes"] == 100 and by_name["loaded.gguf"]["loaded"] is True
    assert by_name["configured.gguf"]["configured"] is True
    assert by_name["configured.gguf"]["loaded"] is False
    assert by_name["other.gguf"]["sha256"] == DIGEST
    assert by_name["loaded.gguf"]["sha256"] is None
    assert not by_name["other.gguf"]["loaded"] and not by_name["other.gguf"]["configured"]


async def test_the_list_with_the_engine_down_says_nothing_is_loaded(client):
    client.state["engine"] = EngineStatus(reachable=False)
    assert not any(m["loaded"] for m in (await client.get("/models")).json())


@pytest.mark.parametrize(
    "name",
    [
        "x.txt",
        "a;b.gguf",
        "a b.gguf",
        ".hidden.gguf",
        "a..b.gguf",
        "x" * 300 + ".gguf",
    ],
)
async def test_a_bad_name_is_refused_with_422(client, name):
    assert (await client.delete(f"/models/{name}")).status_code == 422


async def test_a_path_never_reaches_the_file_system(client, tmp_path):
    outside = tmp_path / "outside.gguf"
    _gguf(outside)
    for name in ("..%2Foutside.gguf", "%2e%2e%2fx.gguf", "%2Fetc%2Fpasswd", "..%5Coutside.gguf"):
        assert (await client.delete(f"/models/{name}")).status_code in (404, 422)
    assert outside.exists()


async def test_the_loaded_and_the_configured_model_cannot_be_deleted(client):
    for name in ("loaded.gguf", "configured.gguf"):
        response = await client.delete(f"/models/{name}")
        assert response.status_code == 409, name
        assert (client.dir / name).exists()


async def test_another_model_is_deleted_with_its_hash_file_and_the_event_is_recorded(client):
    before = sorted(p.name for p in client.dir.iterdir())
    response = await client.delete("/models/other.gguf")
    assert response.status_code == 204
    after = sorted(p.name for p in client.dir.iterdir())
    print(f"files before {len(before)}, after {len(after)}")
    assert set(before) - set(after) == {"other.gguf", "other.gguf.sha256"}
    events = (await client.get("/admin-events", params={"action": "model.delete"})).json()
    assert len(events) == 1 and "other.gguf" in str(events[0]["details"])
    assert (await client.delete("/models/other.gguf")).status_code == 404


async def test_without_a_models_directory_the_answer_is_409(client, monkeypatch, tmp_path):
    from app.config import get_settings

    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "missing"))
    get_settings.cache_clear()
    assert (await client.get("/models")).status_code == 409
    assert (await client.delete("/models/x.gguf")).status_code == 409


async def test_a_link_out_of_the_directory_is_neither_listed_nor_followed(client, tmp_path):
    outside = tmp_path / "precious.txt"
    outside.write_text("do not touch")
    (client.dir / "link.gguf").symlink_to(outside)
    assert "link.gguf" not in [m["name"] for m in (await client.get("/models")).json()]
    assert (await client.delete("/models/link.gguf")).status_code == 422
    assert outside.read_text() == "do not touch"
    assert (client.dir / "link.gguf").is_symlink()
