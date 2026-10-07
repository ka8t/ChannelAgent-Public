"""Tests: configuration through the API. A secret is never returned or echoed,
ENCRYPTION_KEY is never set through it, and it applies the same checks as `--set`.
"""

import re
import shutil
from pathlib import Path

import httpx
import pytest
from cryptography.fernet import Fernet

REPO = Path(__file__).resolve().parent.parent
KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}
SECRETS = {
    "API_SERVER_KEY": KEY,
    "TELEGRAM_BOT_TOKEN": "123456789:AAExampleTokenValueForTestsOnly_12345",
    "EMAIL_PASSWORD": "pa55-w0rd-for-tests-only",
}


@pytest.fixture
async def client(fresh_db, monkeypatch, tmp_path):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    env, example = tmp_path / ".env", tmp_path / ".env.example"
    shutil.copy(REPO / ".env.example", example)
    lines = example.read_text().splitlines()
    out = []
    for ln in lines:
        name = ln.split("=", 1)[0]
        out.append(f"{name}={SECRETS[name]}" if name in SECRETS else ln)
    env.write_text("\n".join(out) + "\n")
    env.chmod(0o600)
    monkeypatch.setenv("ENV_FILE", str(env))
    monkeypatch.setenv("ENV_EXAMPLE_FILE", str(example))
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        c.env = env
        yield c
    deps.reset_failure_state()


async def test_reading_the_configuration_never_returns_a_secret(client):
    response = await client.get("/config")
    assert response.status_code == 200
    entries = {e["key"]: e for e in response.json()}
    secrets = [e for e in entries.values() if e["secret"]]
    assert {"API_SERVER_KEY", "TELEGRAM_BOT_TOKEN", "EMAIL_PASSWORD", "ENCRYPTION_KEY"} <= {
        e["key"] for e in secrets
    }
    assert all(e["value"] is None for e in secrets)
    assert entries["API_SERVER_KEY"]["is_set"] is True
    for value in SECRETS.values():
        assert value not in response.text
    print(f"config: {len(entries)} variables, {len(secrets)} secret, 0 clear secret values")
    assert entries["LLAMA_PORT"]["value"] == "8080"


async def test_the_encryption_key_can_never_be_set_through_the_api(client):
    key = Fernet.generate_key().decode()
    before = client.env.read_bytes()
    response = await client.patch("/config", json={"key": "ENCRYPTION_KEY", "value": key})
    assert response.status_code == 409
    assert key not in response.text
    assert client.env.read_bytes() == before
    assert not list(client.env.parent.glob(".env.bak*"))


async def test_an_empty_encryption_key_is_refused_too(client):
    text = client.env.read_text().splitlines()
    client.env.write_text(
        "\n".join("ENCRYPTION_KEY=" if ln.startswith("ENCRYPTION_KEY=") else ln for ln in text)
        + "\n"
    )
    response = await client.patch(
        "/config", json={"key": "ENCRYPTION_KEY", "value": Fernet.generate_key().decode()}
    )
    assert response.status_code == 409


async def test_a_change_applies_the_same_rules_as_the_script(client):
    ok = await client.patch("/config", json={"key": "LLAMA_PORT", "value": "9001"})
    assert ok.status_code == 200
    body = ok.json()
    assert re.fullmatch(r"\.env\.bak-\d{8}-\d{6}", body.pop("backup")), ok.json()
    assert body == {
        "key": "LLAMA_PORT",
        "changed": True,
        "applies": "at the next start of the application",
    }
    assert "LLAMA_PORT=9001" in client.env.read_text()
    backup = client.env.parent / ok.json()["backup"]
    assert "LLAMA_PORT=8080" in backup.read_text()
    assert oct(backup.stat().st_mode & 0o777) == "0o600"
    bad = await client.patch("/config", json={"key": "LLAMA_PORT", "value": "70000"})
    assert bad.status_code == 422 and "NOT changed" in bad.json()["detail"]
    unknown = await client.patch("/config", json={"key": "LLAMA_PROT", "value": "1"})
    assert unknown.status_code == 422 and "LLAMA_PORT" in unknown.json()["detail"]
    assert "LLAMA_PORT=9001" in client.env.read_text()


async def test_a_secret_write_is_recorded_without_its_value_and_never_echoed(client):
    new_token = "987654321:AAAnotherTokenValueForTestsOnly_67890"
    response = await client.patch("/config", json={"key": "TELEGRAM_BOT_TOKEN", "value": new_token})
    assert response.status_code == 200 and new_token not in response.text
    events = (await client.get("/admin-events", params={"action": "config.set"})).json()
    assert len(events) == 1 and events[0]["actor"] == "api"
    assert new_token not in str(events) and "TELEGRAM_BOT_TOKEN" in str(events)
    assert new_token not in (await client.get("/config")).text


async def test_without_an_env_file_the_answer_is_409(client, monkeypatch, tmp_path):
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "missing.env"))
    assert (await client.get("/config")).status_code == 409
    patch = await client.patch("/config", json={"key": "LLAMA_PORT", "value": "1"})
    assert patch.status_code == 409


async def test_unknown_fields_are_refused(client):
    response = await client.patch("/config", json={"key": "LLAMA_PORT", "value": "1", "force": 1})
    assert response.status_code == 422


async def test_the_web_fetch_allow_list_is_read_and_set_through_the_api(client):
    """The hosts a page may come from are managed through /config (UI and script)."""
    listed = (await client.get("/config")).json()
    names = {entry["key"] for entry in listed} if isinstance(listed, list) else set(listed)
    assert {"WEB_FETCH_ALLOWED_HOSTS", "WEB_FETCH_MAX_BYTES"} <= names
    ok = await client.patch("/config", json={"key": "WEB_FETCH_ALLOWED_HOSTS", "value": "*"})
    assert ok.status_code == 200
    line = next(ln for ln in client.env.read_text().splitlines()
                if ln.startswith("WEB_FETCH_ALLOWED_HOSTS="))  # fmt: skip
    assert line == "WEB_FETCH_ALLOWED_HOSTS='*'", "quoted: the shell must not expand it"
    from dotenv import dotenv_values

    assert dotenv_values(client.env)["WEB_FETCH_ALLOWED_HOSTS"] == "*"
    bad = await client.patch("/config", json={"key": "WEB_FETCH_MAX_BYTES", "value": "0"})
    assert bad.status_code == 422


async def test_a_real_browser_name_is_written_quoted_and_read_back_exactly(client):
    """A User-Agent has spaces, parentheses and semicolons; the file keeps it for both the
    application (dotenv) and the shell (start.sh sources .env)."""
    import subprocess

    from dotenv import dotenv_values

    agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140.0"
    ok = await client.patch("/config", json={"key": "WEB_FETCH_USER_AGENT", "value": agent})
    assert ok.status_code == 200, ok.text
    assert dotenv_values(client.env)["WEB_FETCH_USER_AGENT"] == agent
    shell = subprocess.run(
        ["bash", "-c", f'set -a; source "{client.env}"; printf %s "$WEB_FETCH_USER_AGENT"'],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert shell.stdout == agent


async def test_a_failed_backup_is_refused_and_writes_nothing(client, monkeypatch):
    from app import settings_rules

    def fail(path):
        raise settings_rules.ConfigError("the .env could not be backed up", "backup_failed")

    monkeypatch.setattr(settings_rules, "backup_env", fail)
    before = client.env.read_bytes()
    response = await client.patch("/config", json={"key": "LLAMA_PORT", "value": "9001"})
    assert response.status_code == 409 and "backed up" in response.json()["detail"]
    assert client.env.read_bytes() == before
