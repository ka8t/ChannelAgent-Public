"""Tests: the Admin API must not be reachable from the network by
default. It serves decrypted conversations behind one static key over
plain HTTP.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


async def _stop_main(task) -> None:
    """Cancel app.main.main() and wait until its own cleanup has ended. A cancelled task
    left pending was finished by pytest-asyncio while closing the event loop, and when its
    cleanup waited for ever the whole suite hung there (measured: 2 hangs in 7 full runs).
    Waiting here, with a limit, turns such a hang into a failure that names every task left."""
    import asyncio

    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=15)
    if not done:
        stacks = []
        for other in asyncio.all_tasks():
            if other is not asyncio.current_task():
                frames = other.get_stack()
                where = " <- ".join(f"{f.f_code.co_name}:{f.f_lineno}" for f in frames)
                stacks.append(f"{other.get_name()}: {where}")
        raise AssertionError("app.main.main() did not stop within 15 s:\n" + "\n".join(stacks))


def test_api_binds_to_loopback_by_default(monkeypatch):
    monkeypatch.delenv("API_SERVER_HOST", raising=False)
    from app.config import Settings

    assert Settings(_env_file=None).api_server_host == "127.0.0.1"


def test_api_host_can_be_widened_explicitly(monkeypatch):
    monkeypatch.setenv("API_SERVER_HOST", "0.0.0.0")
    from app.config import Settings

    assert Settings(_env_file=None).api_server_host == "0.0.0.0"


def test_compose_publishes_the_api_on_loopback_by_default():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    ports = compose["services"]["channelagent"]["ports"]
    assert ports == [
        "${API_BIND_ADDRESS:-127.0.0.1}:${API_SERVER_PORT:-8700}:${API_SERVER_PORT:-8700}"
    ]


def test_compose_makes_the_api_listen_on_every_interface_inside_the_container():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    env = compose["services"]["channelagent"]["environment"]
    assert env["API_SERVER_HOST"] == "0.0.0.0"
    assert "ENV API_SERVER_HOST=0.0.0.0" in (REPO_ROOT / "Dockerfile").read_text()


def test_env_example_documents_both_addresses_with_loopback_defaults():
    lines = (REPO_ROOT / ".env.example").read_text().splitlines()
    assert "API_SERVER_HOST=127.0.0.1" in lines
    assert "API_BIND_ADDRESS=127.0.0.1" in lines


@pytest.mark.parametrize(
    ("key", "acceptable"),
    [(None, False), ("", False), ("short", False), ("x" * 15, False),
     ("x" * 16, False), ("a" * 64, False), ("Zq8vT3mK9xW2pL7n", True)],
)
def test_short_or_missing_api_key_is_not_acceptable(key, acceptable):
    from app.api.deps import api_key_is_acceptable

    assert api_key_is_acceptable(key) is acceptable


async def test_main_does_not_start_the_api_with_a_short_key(fresh_db, monkeypatch, caplog):
    import asyncio
    import logging

    from app import main as app_main
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", "short")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("EMAIL_IMAP_HOST", "")
    get_settings.cache_clear()

    started = []
    monkeypatch.setattr("uvicorn.Server.serve", lambda self: started.append(self))

    caplog.set_level(logging.INFO, logger="channelagent")
    task = asyncio.create_task(app_main.main())
    await asyncio.sleep(0)
    for _ in range(50):
        if any("Admin API NOT started" in r.message for r in caplog.records):
            break
        await asyncio.sleep(0.05)
    await _stop_main(task)
    assert any("Admin API NOT started" in r.message for r in caplog.records)
    assert started == [], "uvicorn must not be started with a weak key"


@pytest.mark.parametrize(("env_host", "expected"), [(None, "127.0.0.1"), ("0.0.0.0", "0.0.0.0")])
async def test_main_starts_uvicorn_on_the_configured_host(
    fresh_db, monkeypatch, env_host, expected
):
    import asyncio

    import uvicorn

    from app import main as app_main
    from app.config import get_settings

    monkeypatch.setenv("API_SERVER_KEY", "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("EMAIL_IMAP_HOST", "")
    # A non-loopback host outside a container needs a declared TLS proxy.
    monkeypatch.setenv("API_REMOTE", "tls-proxy")
    if env_host is None:
        monkeypatch.delenv("API_SERVER_HOST", raising=False)
    else:
        monkeypatch.setenv("API_SERVER_HOST", env_host)
    get_settings.cache_clear()

    seen: dict = {}

    class SpyConfig:
        def __init__(self, app, **kwargs):
            seen.update(kwargs)

    class SpyServer:
        def __init__(self, config):
            pass

        async def serve(self):
            await asyncio.sleep(3600)

    monkeypatch.setattr(uvicorn, "Config", SpyConfig)
    monkeypatch.setattr(uvicorn, "Server", SpyServer)

    task = asyncio.create_task(app_main.main())
    for _ in range(100):
        if seen:
            break
        await asyncio.sleep(0.05)
    await _stop_main(task)
    assert seen["host"] == expected
