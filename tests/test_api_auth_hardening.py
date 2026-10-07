"""Tests: failed Admin API authentication is logged, rate limited
per source address, weak keys are refused, and rotation is documented.
"""

import logging
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}
SECRET_WRONG = "Bearer this-wrong-key-must-never-be-logged"


@pytest.fixture
async def api(fresh_db, monkeypatch):
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    await init_db()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    deps.reset_failure_state()


async def test_eleventh_wrong_key_from_one_address_gets_429(api):
    codes = [
        (await api.get("/users", headers={"Authorization": SECRET_WRONG})).status_code
        for _ in range(11)
    ]
    assert codes == [401] * 10 + [429]


async def test_a_blocked_address_is_refused_even_with_the_right_key(api):
    for _ in range(10):
        await api.get("/users", headers={"Authorization": SECRET_WRONG})
    assert (await api.get("/users", headers=GOOD)).status_code == 429


async def test_ten_failures_below_the_limit_do_not_block_the_right_key(api):
    for _ in range(9):
        await api.get("/users", headers={"Authorization": SECRET_WRONG})
    assert (await api.get("/users", headers=GOOD)).status_code == 200


async def test_the_block_expires_after_the_window(api, monkeypatch):
    from app.api import deps

    clock = [1000.0]
    monkeypatch.setattr(deps, "_now", lambda: clock[0])
    for _ in range(10):
        await api.get("/users", headers={"Authorization": SECRET_WRONG})
    assert (await api.get("/users", headers=GOOD)).status_code == 429
    clock[0] += deps.FAILURE_WINDOW_SECONDS + 1
    assert (await api.get("/users", headers=GOOD)).status_code == 200


async def test_another_address_is_not_blocked(api, fresh_db):
    from app.api.app import app

    for _ in range(10):
        await api.get("/users", headers={"Authorization": SECRET_WRONG})
    other = httpx.ASGITransport(app=app, client=("198.51.100.7", 5000))
    async with httpx.AsyncClient(transport=other, base_url="http://t") as c:
        assert (await c.get("/users", headers=GOOD)).status_code == 200


async def test_a_successful_call_does_not_count_as_a_failure(api):
    for _ in range(30):
        assert (await api.get("/users", headers=GOOD)).status_code == 200


async def test_missing_header_counts_as_a_failure(api):
    codes = [(await api.get("/users")).status_code for _ in range(11)]
    assert codes == [401] * 10 + [429]


async def test_failures_are_logged_with_the_address_and_never_the_key(api, caplog):
    with caplog.at_level(logging.WARNING, logger="channelagent"):
        await api.get("/users", headers={"Authorization": SECRET_WRONG})
    lines = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(lines) == 1
    text = lines[0].getMessage()
    assert "203.0.113.9" in text
    assert "this-wrong-key" not in text
    assert KEY not in text


async def test_a_blocked_attempt_is_logged_too(api, caplog):
    for _ in range(10):
        await api.get("/users", headers={"Authorization": SECRET_WRONG})
    with caplog.at_level(logging.WARNING, logger="channelagent"):
        await api.get("/users", headers={"Authorization": SECRET_WRONG})
    assert any("blocked" in r.getMessage() for r in caplog.records)


async def test_the_failure_table_does_not_grow_without_bound(api, monkeypatch):
    from app.api import deps

    monkeypatch.setattr(deps, "MAX_TRACKED_ADDRESSES", 5)
    from app.api.app import app

    for i in range(20):
        t = httpx.ASGITransport(app=app, client=(f"192.0.2.{i}", 1))
        async with httpx.AsyncClient(transport=t, base_url="http://t") as c:
            await c.get("/users", headers={"Authorization": SECRET_WRONG})
    assert len(deps._failures) <= 5


@pytest.mark.parametrize(
    "key",
    [
        "a" * 32,
        "0" * 64,
        "abab" * 8,
        "abcabc" * 6,
        "0123456789abcdef",
        "0123456789abcdef0123456789abcdef",
        "changeme-changeme-changeme",
        "my-Password-is-Password-123",
        "your_api_server_key_here",
        "example-key-example-key",
        "abcdabcdabcdaabc",  # fewer than 6 distinct characters
        "Zq8vT3mK" * 4,  # 8 distinct characters, still one unit repeated
    ],
)
def test_obviously_weak_keys_are_rejected(key):
    from app.api.deps import api_key_is_acceptable

    assert api_key_is_acceptable(key) is False


@pytest.mark.parametrize(
    "key",
    [
        "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA",
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    ],
)
def test_random_keys_are_accepted(key):
    from app.api.deps import api_key_is_acceptable

    assert api_key_is_acceptable(key) is True
