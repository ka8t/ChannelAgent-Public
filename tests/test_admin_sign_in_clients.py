"""The two clients of the API sign in as named administrators: the admin UI keeps the
token in its server-side session and revokes it at sign-out; the script saves it per API
address in a file of mode 600 and uses it for every later call. The same credential choice
for the script's curl lines (`app/admin/credentials.py`).
"""

import io
import os
import re
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from app.admin import client as cli
from app.admin import credentials
from app.security import totp

KEY = "Uq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
PASSWORD = "a long enough passphrase"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
async def base(fresh_db, monkeypatch, tmp_path):
    from app.admin.jobs import registry
    from app.api import deps
    from app.config import get_settings
    from app.db.session import init_db
    from app.security import passwords
    from app.ui import security

    monkeypatch.setattr(passwords, "LOG2_N", 10)
    passwords.dummy_hash.cache_clear()
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    get_settings.cache_clear()
    deps.reset_failure_state()
    security.sessions.clear()
    security.login_limiter.clear()
    registry.clear()
    await init_db()
    yield
    security.sessions.clear()
    security.login_limiter.clear()
    deps.reset_failure_state()
    passwords.dummy_hash.cache_clear()
    get_settings.cache_clear()


async def _create_owner(name="alice") -> str:
    """The first owner, made with the static key as an installation would. Its TOTP secret."""
    from app.api.app import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        created = await c.post(
            "/admins",
            json={"name": name, "scope": "owner", "password": PASSWORD},
            headers={"Authorization": f"Bearer {KEY}"},
        )
    assert created.status_code == 201
    return created.json()["totp_secret"]


def _code(secret: str, offset: int = 0) -> str:
    return totp.code_at(secret, totp.current_step() + offset)


def _db_rows(query: str) -> list:
    from app.config import get_settings

    path = get_settings().database_url.split("///", 1)[1]
    return sqlite3.connect(path).execute(query).fetchall()


# --- the admin UI ---


@pytest.fixture
async def ui(base):
    from app.server import root

    transport = httpx.ASGITransport(app=root, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="https://t") as client:
        yield client


def _csrf(html: str) -> str:
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


async def _ui_sign_in(ui, **fields) -> httpx.Response:
    page = await ui.get("/ui/login")
    return await ui.post("/ui/login", data={"csrf": _csrf(page.text), **fields})


async def test_the_key_form_is_offered_until_a_named_owner_exists(ui):
    assert 'name="key"' in (await ui.get("/ui/login")).text
    await _create_owner()
    page = (await ui.get("/ui/login")).text
    assert 'name="key"' not in page and 'name="name"' in page
    assert (await _ui_sign_in(ui, key=KEY)).status_code == 401


async def test_a_named_owner_signs_in_to_the_ui_and_acts_as_themselves(ui):
    secret = await _create_owner()
    signed = await _ui_sign_in(ui, name="alice", password=PASSWORD, code=_code(secret))
    assert signed.status_code == 303
    assert "adm:alice" in (await ui.get("/ui/")).text  # whoami through the API
    assert ("adm:alice", "ui.login") in _db_rows("select actor, action from admin_events")
    tokens = _db_rows("select label, scope, revoked_at from api_tokens")
    assert tokens == [("ui:web", "owner", None)]


async def test_the_token_never_reaches_the_browser(ui, monkeypatch):
    from app.admin import accounts

    issued = []
    real = accounts.issue_token

    async def spy(*a, **k):
        row, raw = await real(*a, **k)
        issued.append(raw)
        return row, raw

    monkeypatch.setattr(accounts, "issue_token", spy)
    secret = await _create_owner()
    signed = await _ui_sign_in(ui, name="alice", password=PASSWORD, code=_code(secret))
    pages = [signed, await ui.get("/ui/"), await ui.get("/ui/status")]
    assert issued and all(issued[0] not in (r.text + str(r.headers)) for r in pages)


async def test_a_wrong_code_is_refused_and_counts_for_the_limiter(ui):
    secret = await _create_owner()
    wrong = "000000" if _code(secret) != "000000" else "111111"
    codes = [
        (await _ui_sign_in(ui, name="alice", password=PASSWORD, code=wrong)).status_code
        for _ in range(6)
    ]
    assert codes == [401] * 5 + [429]


async def test_signing_out_revokes_the_token(ui):
    secret = await _create_owner()
    await _ui_sign_in(ui, name="alice", password=PASSWORD, code=_code(secret))
    csrf = _csrf((await ui.get("/ui/")).text)
    assert (await ui.post("/ui/logout", data={"csrf": csrf})).status_code == 303
    assert _db_rows("select revoked_at is not null from api_tokens") == [(1,)]
    assert ("adm:alice", "ui.logout") in _db_rows("select actor, action from admin_events")


# --- the script ---


async def run(*argv: str, stdin: str = ""):
    out, err = io.StringIO(), io.StringIO()
    old = sys.stdin
    sys.stdin = io.StringIO(stdin)
    try:
        code = await cli.amain(list(argv), out, err)
    finally:
        sys.stdin = old
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def server(base, monkeypatch):
    from tests.test_admin_client import _ThreadedServer

    with _ThreadedServer() as running:
        monkeypatch.setenv("API_SERVER_PORT", str(running.port))
        monkeypatch.setenv("API_SERVER_HOST", "127.0.0.1")
        monkeypatch.delenv("API_URL", raising=False)
        yield f"http://127.0.0.1:{running.port}"


async def test_the_script_signs_in_saves_the_token_and_uses_it(server):
    secret = await _create_owner()
    code, out, err = await run(
        "sign-in", "--name", "alice", "--transport", "http", stdin=f"{PASSWORD}\n{_code(secret)}\n"
    )
    assert code == 0, err
    assert "ca_" not in out  # the token is saved, not printed
    path = credentials.token_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    saved = credentials.saved_token(server)
    assert saved["token"].startswith("ca_") and saved["account"] == "alice"
    code, out, _ = await run("whoami", "--transport", "http", "--json")
    assert code == 0 and '"adm:alice"' in out


async def test_without_a_saved_token_the_disabled_key_is_refused_over_http(server):
    await _create_owner()
    code, _, err = await run("list-users", "--transport", "http")
    assert code == 1 and "named owner" in err


async def test_in_process_the_key_still_works_for_recovery(base, monkeypatch):
    await _create_owner()
    code, out, err = await run("list-admins", "--transport", "inprocess", "--json")
    assert code == 0, err
    assert '"alice"' in out


async def test_signing_out_revokes_and_forgets_the_token(server):
    secret = await _create_owner()
    await run(
        "sign-in", "--name", "alice", "--transport", "http", stdin=f"{PASSWORD}\n{_code(secret)}\n"
    )
    code, out, _ = await run("sign-out")
    assert code == 0 and "signed out" in out
    assert credentials.saved_token(server) is None
    assert _db_rows("select revoked_at is not null from api_tokens") == [(1,)]


async def test_a_refused_saved_token_says_to_sign_in_again(server):
    secret = await _create_owner()
    await run(
        "sign-in", "--name", "alice", "--transport", "http", stdin=f"{PASSWORD}\n{_code(secret)}\n"
    )
    saved = credentials.saved_token(server)
    credentials.save_token(server, {**saved, "token": "ca_" + "x" * 43})
    code, _, err = await run("list-users", "--transport", "http")
    assert code == 1 and "sign-in" in err


async def test_a_wrong_password_saves_nothing(server):
    await _create_owner()
    code, _, _ = await run(
        "sign-in", "--name", "alice", "--transport", "http", stdin="wrong password here\n000000\n"
    )
    assert code == 1 and credentials.saved_token(server) is None


def test_the_curl_header_is_the_saved_token_else_the_key(tmp_path):
    env = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path), "API_SERVER_KEY": KEY}
    script = [sys.executable, str(REPO / "app/admin/credentials.py"), "--header", "http://h:1"]

    def header(environment):
        result = subprocess.run(script, env=environment, capture_output=True, text=True)
        return result.returncode, result.stdout.strip()

    assert header(env) == (0, f"Authorization: Bearer {KEY}")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    try:
        credentials.save_token("http://h:1", {"token": "ca_saved", "id": 1})
    finally:
        del os.environ["XDG_CONFIG_HOME"]
    assert header(env) == (0, "Authorization: Bearer ca_saved")
    no_key = {k: v for k, v in env.items() if k != "API_SERVER_KEY"}
    no_key["XDG_CONFIG_HOME"] = str(tmp_path / "empty")
    assert header(no_key)[0] == 1


def test_start_sh_asks_the_shared_module_for_its_curl_credential():
    text = (REPO / "start.sh").read_text()
    assert 'python3 app/admin/credentials.py --header "$1"' in text
    assert "Bearer %s" not in text  # no second way of choosing the credential
