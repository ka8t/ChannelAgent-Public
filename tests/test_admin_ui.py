"""Tests: the admin UI (D8: server-rendered, vendored, no CDN, no Node).

- No URL of another site in the templates and static files; no inline script, no event
  handler attribute, no inline style.
- Sign-in with the admin key into a server-side session: a `__Host-` cookie (Secure,
  HttpOnly, SameSite=Strict), the id replaced at sign-in, idle and absolute expiry, sign-out;
  the 6th wrong key in a minute is 429.
- CSRF: a POST without the token, or from another site, is 403.
- CSP with a nonce, nosniff, no-referrer and no-store on every authenticated response.
- Channel text (an access request's first message, a logged message) is escaped on every screen.
- Every API operation has a screen linked from the operations page (contract: N of N, 0
  unreachable).
- The UI acts through the API with the actor `ui:web`.
"""

import re
from pathlib import Path

import httpx
import pytest

KEY = "Uq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
REPO = Path(__file__).resolve().parents[1]
UI_FILES = sorted(
    p for p in (REPO / "app" / "ui").rglob("*") if p.suffix in (".html", ".js", ".css")
)
PAYLOAD = '<script>alert("x")</script><img src=x onerror=alert(1)>'


@pytest.fixture
async def ui(fresh_db, monkeypatch):
    from app.admin.jobs import registry
    from app.api import deps
    from app.config import get_settings
    from app.db.session import init_db
    from app.server import root
    from app.ui import security

    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    security.sessions.clear()
    security.login_limiter.clear()
    registry.clear()
    await init_db()
    transport = httpx.ASGITransport(app=root, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="https://t") as client:
        yield client
    security.sessions.clear()
    security.login_limiter.clear()
    registry.clear()
    get_settings.cache_clear()


def _csrf(html: str) -> str:
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


async def _sign_in(client, key=KEY) -> httpx.Response:
    page = await client.get("/ui/login")
    return await client.post("/ui/login", data={"csrf": _csrf(page.text), "key": key})


async def _signed_in(client) -> str:
    response = await _sign_in(client)
    assert response.status_code == 303, response.text
    return _csrf((await client.get("/ui/")).text)


async def _seed_channel_text() -> None:
    from app.admin import service
    from app.db.models import Channel, Direction
    from app.db.session import session_scope

    async with session_scope() as session:
        user = await service.create_user(session, display_name=PAYLOAD)
        agents = await service.list_agents(session, user.id)
        agent = agents[0] if agents else await service.create_agent(session, user.id, "default")
        await service.request_access(session, Channel.TELEGRAM, "999", PAYLOAD)
        await service.record_action(
            session,
            user_id=user.id,
            agent_id=agent.id,
            channel=Channel.TELEGRAM,
            direction=Direction.INBOUND,
            text=PAYLOAD,
        )
        await session.commit()


# --- the files ---


def test_no_url_of_another_site_in_the_templates_and_static_files():
    assert UI_FILES, "the UI files are found"
    hits = [
        f"{p.name}: {m.group(0)}"
        for p in UI_FILES
        for m in re.finditer(r"(https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}", p.read_text())
    ]
    assert hits == []


def test_no_inline_script_handler_or_style_in_the_templates():
    templates = [p for p in UI_FILES if p.suffix == ".html"]
    text = "\n".join(p.read_text() for p in templates)
    assert re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", text) == []
    assert re.findall(r"\son[a-z]+\s*=", text) == []
    assert re.findall(r"\sstyle\s*=", text) == []
    assert "<style" not in text


# --- sign-in ---


async def test_sign_in_sets_a_host_cookie_and_replaces_the_session_id(ui):
    page = await ui.get("/ui/login")
    before = page.headers["set-cookie"]
    assert before.startswith("__Host-ca_session=")
    for attribute in ("Path=/", "Secure", "HttpOnly", "SameSite=Strict"):
        assert attribute in before
    response = await ui.post("/ui/login", data={"csrf": _csrf(page.text), "key": KEY})
    assert response.status_code == 303 and response.headers["location"] == "/ui/"
    after = response.headers["set-cookie"]
    assert after.split(";")[0] != before.split(";")[0]
    assert (await ui.get("/ui/")).status_code == 200
    ui.cookies.clear()
    ui.cookies.set(
        "__Host-ca_session", before.split(";")[0].split("=", 1)[1], domain="t.local"
    )  # the cookie jar's name for host "t"
    assert (await ui.get("/ui/")).status_code == 303, "the pre-sign-in id is worthless"


async def test_the_sixth_wrong_key_in_a_minute_is_429_even_with_the_right_one(ui):
    codes = [(await _sign_in(ui, "wrong-" + "k" * 30)).status_code for _ in range(5)]
    sixth = await _sign_in(ui, KEY)
    assert codes == [401] * 5 and sixth.status_code == 429


async def test_a_session_expires_when_idle_and_after_eight_hours(ui, monkeypatch):
    from app.ui import security

    clock = [1000.0]
    monkeypatch.setattr(security, "_now", lambda: clock[0])
    await _signed_in(ui)
    clock[0] += security.IDLE_SECONDS - 1
    assert (await ui.get("/ui/")).status_code == 200
    clock[0] += security.IDLE_SECONDS + 1
    assert (await ui.get("/ui/")).status_code == 303
    await _signed_in(ui)
    for _ in range(17):  # a request every 29 minutes: never idle, but 8 h pass
        clock[0] += security.IDLE_SECONDS - 60
        last = (await ui.get("/ui/")).status_code
    assert last == 303


async def test_sign_out_ends_the_session(ui):
    csrf = await _signed_in(ui)
    response = await ui.post("/ui/logout", data={"csrf": csrf})
    assert response.status_code == 303 and "Max-Age=0" in response.headers["set-cookie"]
    assert (await ui.get("/ui/")).status_code == 303


async def test_a_session_id_is_worthless_after_sign_out(ui):
    csrf = await _signed_in(ui)
    old = ui.cookies.get("__Host-ca_session")
    await ui.post("/ui/logout", data={"csrf": csrf})
    ui.cookies.clear()
    ui.cookies.set("__Host-ca_session", old, domain="t.local")  # the cookie jar's name for host "t"
    assert (await ui.get("/ui/")).status_code == 303


async def test_sign_in_needs_the_csrf_token_of_its_page(ui):
    await ui.get("/ui/login")
    response = await ui.post("/ui/login", data={"key": KEY})
    assert response.status_code == 403 and (await ui.get("/ui/")).status_code == 303


async def test_a_session_that_has_not_signed_in_cannot_submit(ui):
    csrf = _csrf((await ui.get("/ui/login")).text)
    response = await ui.post("/ui/op/create-user", data={"display_name": "x", "csrf": csrf})
    assert response.status_code == 403


async def test_signed_out_requests_are_sent_to_sign_in_or_refused(ui):
    assert (await ui.get("/ui/")).headers["location"] == "/ui/login"
    assert (await ui.get("/ui/op/list-users")).status_code == 303
    assert (await ui.post("/ui/op/create-user", data={"csrf": "x"})).status_code == 403
    assert (await ui.get("/ui/jobs/0123456789ab/status")).status_code == 401


# --- CSRF ---


async def test_a_forged_cross_site_post_is_403(ui):
    from app.admin import service
    from app.db.session import session_scope

    csrf = await _signed_in(ui)
    no_token = await ui.post("/ui/op/create-user", data={"display_name": "forged"})
    wrong = await ui.post("/ui/op/create-user", data={"display_name": "f", "csrf": "x" * 43})
    other_site = await ui.post(
        "/ui/op/create-user",
        data={"display_name": "forged", "csrf": csrf},
        headers={"Origin": "https://evil.example"},
    )
    good = await ui.post("/ui/op/create-user", data={"display_name": "Real", "csrf": csrf})
    assert [no_token.status_code, wrong.status_code, other_site.status_code] == [403, 403, 403]
    assert good.status_code == 200
    async with session_scope() as session:
        names = [u.display_name for u in await service.list_users(session)]
    assert names == ["Real"]


async def test_the_ui_acts_through_the_api_as_ui_web(ui):
    from app.admin import service
    from app.db.session import session_scope

    csrf = await _signed_in(ui)
    await ui.post("/ui/op/create-user", data={"display_name": "Bob", "csrf": csrf})
    async with session_scope() as session:
        events = await service.search_admin_events(session, action="user.create")
        logins = await service.search_admin_events(session, action="ui.login")
    assert [e.actor for e in events] == ["ui:web"] and len(logins) == 1


async def test_a_next_page_is_followed_only_inside_the_ui(ui):
    csrf = await _signed_in(ui)
    inside = await ui.post(
        "/ui/op/create-user", data={"display_name": "a", "csrf": csrf, "next": "/ui/requests"}
    )
    outside = await ui.post(
        "/ui/op/create-user", data={"display_name": "b", "csrf": csrf, "next": "//evil.example/"}
    )
    assert inside.status_code == 303 and inside.headers["location"] == "/ui/requests"
    assert outside.status_code == 200


# --- headers, escaping, reach ---


def _headers_ok(response: httpx.Response) -> bool:
    csp = response.headers.get("content-security-policy", "")
    return (
        re.search(r"script-src 'nonce-[A-Za-z0-9_-]{16,}'", csp) is not None
        and "unsafe-inline" not in csp
        and "frame-ancestors 'none'" in csp
        and response.headers.get("x-content-type-options") == "nosniff"
        and response.headers.get("cache-control") == "no-store"
        and response.headers.get("referrer-policy") == "same-origin"
    )


async def _every_screen(ui) -> list[httpx.Response]:
    from app.ui.app import operations

    pages = [
        await ui.get(u)
        for u in (
            "/ui/",
            "/ui/requests",
            "/ui/requests?status=all",
            "/ui/status",
            "/ui/backups",
            "/ui/config",
            "/ui/models",
            "/ui/tasks",
            "/ui/parity",
            "/ui/operations",
        )
    ]
    pages.append(await ui.get("/ui/logs", params={"run": "1"}))
    for command in operations():
        pages.append(await ui.get(f"/ui/op/{command}", params={"user_id": "1", "agent_id": "1"}))
    return pages


async def test_every_authenticated_response_carries_the_security_headers(ui):
    await _signed_in(ui)
    responses = await _every_screen(ui)
    responses.append(await ui.get("/ui/static/app.js"))
    responses.append(await ui.get("/ui/jobs/0123456789ab"))
    missing = [str(r.url) for r in responses if not _headers_ok(r)]
    print(f"security headers checked on {len(responses)} responses, missing on {len(missing)}")
    assert len(responses) >= 70 and missing == []


async def test_the_nonce_changes_with_every_response_and_marks_the_script(ui):
    first, second = (await ui.get("/ui/login")), (await ui.get("/ui/login"))
    nonces = [re.search(r'nonce="([^"]+)"', r.text).group(1) for r in (first, second)]
    assert nonces[0] != nonces[1]
    assert f"'nonce-{nonces[1]}'" in second.headers["content-security-policy"]


async def test_channel_text_is_escaped_on_every_screen(ui):
    await _seed_channel_text()
    await _signed_in(ui)
    responses = await _every_screen(ui)
    responses.append(await ui.get("/ui/op/list-requests", params={"status": "all"}))
    raw = [str(r.url) for r in responses if "<script>alert" in r.text or "<img src=x" in r.text]
    shown = [str(r.url) for r in responses if "&lt;script&gt;alert(" in r.text]
    print(f"{len(responses)} screens, payload raw on {len(raw)}, escaped on {len(shown)}")
    assert raw == []
    for page in ("/ui/requests", "/ui/logs?run=1", "/ui/op/list-users", "/ui/op/list-requests"):
        assert any(page.split("?")[0] in u for u in shown), page


async def test_every_api_operation_has_a_screen_linked_from_the_operations_page(ui):
    from app.ui.app import operations

    await _signed_in(ui)
    assert 'href="/ui/operations"' in (await ui.get("/ui/")).text.split("<main>")[0]
    overview = (await ui.get("/ui/operations")).text
    ops = operations()
    linked = [c for c in ops if f'href="/ui/op/{c}"' in overview]
    reached = [c for c in ops if (await ui.get(f"/ui/op/{c}")).status_code == 200]
    print(f"UI reach: {len(reached)} of {len(ops)}, linked {len(linked)}")
    assert len(ops) >= 66 and linked == list(ops) and reached == list(ops)


async def test_a_job_started_from_the_ui_is_followed_to_its_end(ui):
    import asyncio

    csrf = await _signed_in(ui)
    response = await ui.post("/ui/op/create-database-backup", data={"csrf": csrf})
    assert response.status_code == 303
    location = response.headers["location"]
    assert re.fullmatch(r"/ui/jobs/[0-9a-f]{12}", location)
    for _ in range(100):
        status = (await ui.get(location + "/status")).json()
        if status["status"] == "done":
            break
        await asyncio.sleep(0.05)
    assert status["status"] == "done", status.get("error")
    page = await ui.get(location)
    assert 'data-final="yes"' in page.text and status["result"]["name"] in page.text


async def test_a_running_job_is_cancelled_from_its_page(ui, monkeypatch):
    """"a pull form with progress and cancel": the job page offers Cancel while the job
    runs, and the job ends `cancelled`."""
    import time

    from app.api import operations

    def slow(_path, _label):
        time.sleep(2)

    monkeypatch.setattr(operations, "make_backup", slow)
    csrf = await _signed_in(ui)
    location = (await ui.post("/ui/op/create-database-backup", data={"csrf": csrf})).headers[
        "location"
    ]
    job_id = location.rsplit("/", 1)[1]
    page = await ui.get(location)
    assert 'action="/ui/op/cancel-job"' in page.text and f'value="{job_id}"' in page.text
    response = await ui.post(
        "/ui/op/cancel-job", data={"csrf": csrf, "job_id": job_id, "next": location}
    )
    assert response.status_code == 303 and response.headers["location"] == location
    assert (await ui.get(location + "/status")).json()["status"] == "cancelled"
    assert 'action="/ui/op/cancel-job"' not in (await ui.get(location)).text


async def test_access_requests_are_approved_from_their_screen(ui):
    await _seed_channel_text()
    csrf = await _signed_in(ui)
    page = await ui.get("/ui/requests")
    request_id = re.search(r'name="request_id" value="(\d+)"', page.text).group(1)
    response = await ui.post(
        "/ui/op/approve-request",
        data={"csrf": csrf, "request_id": request_id, "next": "/ui/requests"},
    )
    assert response.status_code == 303
    assert "approved" in (await ui.get("/ui/requests?status=approved")).text


async def test_a_post_from_the_uis_own_page_carries_its_origin(ui):
    """Measured in Chrome (2026-09-27): with `Referrer-Policy: no-referrer` the browser sent
    `Origin: null` on the sign-in form and the Origin check refused it. Same-origin keeps the
    real Origin on the UI's own posts; `null` stays refused."""
    page = await ui.get("/ui/login")
    assert page.headers["referrer-policy"] == "same-origin"
    own = await ui.post(
        "/ui/login", data={"csrf": _csrf(page.text), "key": KEY}, headers={"Origin": "https://t"}
    )
    null = await ui.post("/ui/login", data={"csrf": "x", "key": KEY}, headers={"Origin": "null"})
    assert own.status_code == 303 and null.status_code == 403


async def test_the_api_keeps_its_key_beside_the_ui(ui):
    assert (await ui.get("/users")).status_code == 401
    assert (await ui.get("/ui/login")).status_code == 200
    assert (await ui.get("/ui/login", headers={"Host": "evil.example"})).status_code == 421


# --- dedicated screens ---


async def test_the_status_screen_shows_the_engine_the_database_and_the_host(ui, monkeypatch):
    from app.ui import app as ui_app

    await _signed_in(ui)
    off = await ui.get("/ui/status")
    assert off.status_code == 200 and "Database" in off.text and "sqlite" in off.text
    assert "Host helper: The host helper is off" in off.text
    real = ui_app.call_api

    async def with_helper(request, method, path, query=None, body=None):
        if path == "/host/status":
            return 200, {"mode": "native", "running": True, "reasons": ["native process 1"],
                         "helper_started_at": "2026-09-27T12:00:00Z"}  # fmt: skip
        return await real(request, method, path, query, body)

    monkeypatch.setattr(ui_app, "call_api", with_helper)
    on = (await ui.get("/ui/status")).text
    for command in ("host-restart", "host-stop", "host-start", "host-rekey"):
        assert f'action="/ui/op/{command}"' in on
    assert 'name="dry_run" value="true"' in on and "native process 1" in on


async def test_the_backups_screen_creates_lists_restores_and_schedules(ui):
    import asyncio

    csrf = await _signed_in(ui)
    job = (await ui.post("/ui/op/create-database-backup", data={"csrf": csrf})).headers["location"]
    for _ in range(100):
        if (await ui.get(job + "/status")).json()["status"] == "done":
            break
        await asyncio.sleep(0.05)
    page = (await ui.get("/ui/backups")).text
    name = re.search(r'name="name" value="([^"]+-manual-[^"]+)"', page).group(1)
    assert 'action="/ui/op/restore-backup"' in page and "KB (" in page
    saved = await ui.post(
        "/ui/op/set-backup-schedule",
        data={"csrf": csrf, "enabled": "true", "interval_minutes": "30", "keep": "4",
              "next": "/ui/backups"},
    )  # fmt: skip
    assert saved.status_code == 303 and name
    page = (await ui.get("/ui/backups")).text
    assert 'name="interval_minutes" value="30"' in page and 'name="keep" value="4"' in page


async def test_the_config_screen_hides_secrets_and_saves_through_the_rules(
    ui, monkeypatch, tmp_path
):
    import shutil

    env = tmp_path / ".env"
    env.write_text(f"API_SERVER_KEY={KEY}\nEMAIL_USERNAME=old@example.org\n")
    shutil.copy(REPO / ".env.example", tmp_path / ".env.example")
    monkeypatch.setenv("ENV_FILE", str(env))
    monkeypatch.setenv("ENV_EXAMPLE_FILE", str(tmp_path / ".env.example"))
    csrf = await _signed_in(ui)
    page = (await ui.get("/ui/config")).text
    assert KEY not in page and "set (hidden)" in page and "old@example.org" in page
    assert 'name="key" value="ENCRYPTION_KEY"' not in page
    ok = await ui.post(
        "/ui/op/set-config",
        data={"csrf": csrf, "key": "EMAIL_USERNAME", "value": "new@example.org",
              "next": "/ui/config"},
    )  # fmt: skip
    bad = await ui.post(
        "/ui/op/set-config", data={"csrf": csrf, "key": "API_SERVER_PORT", "value": "99999999"}
    )
    assert ok.status_code == 303 and "EMAIL_USERNAME=new@example.org" in env.read_text()
    assert bad.status_code == 422 and "API_SERVER_PORT=99999999" not in env.read_text()


async def test_the_models_screen_lists_and_deletes(ui, monkeypatch, tmp_path):
    from app.config import get_settings

    models = tmp_path / "models"
    models.mkdir()
    (models / "tiny.gguf").write_bytes(b"GGUF" + b"\0" * 2044)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()
    csrf = await _signed_in(ui)
    page = (await ui.get("/ui/models")).text
    assert "tiny.gguf" in page and "2.0 KB (2,048 bytes)" in page
    assert 'action="/ui/op/pull-model"' in page and 'action="/ui/op/import-model"' in page
    assert 'action="/ui/op/delete-model"' in page and 'name="name" value="tiny.gguf"' in page
    gone = await ui.post(
        "/ui/op/delete-model", data={"csrf": csrf, "name": "tiny.gguf", "next": "/ui/models"}
    )
    assert gone.status_code == 303 and not (models / "tiny.gguf").exists()


def test_sizes_are_shown_in_units_and_bytes():
    from app.ui.app import _size

    assert _size(4_920_739_232) == "4.9 GB (4,920,739,232 bytes)"
    assert _size(512) == "512 bytes" and _size(None) == "-"


async def test_a_host_job_offers_no_cancel(ui, monkeypatch):
    """Measured in Chrome (2026-09-27): a restore's page showed Cancel, which the helper
    refuses (409). A host job cannot be cut half-way: no button."""
    from app.ui import app as ui_app

    await _signed_in(ui)
    real = ui_app.call_api

    async def running(request, method, path, query=None, body=None):
        if path == "/jobs/host-0123456789ab":
            return 200, {
                "id": "host-0123456789ab",
                "kind": "host-restore",
                "status": "running",
                "progress": 0.4,
                "message": "Restoring",
                "result": None,
                "error": None,
            }
        return await real(request, method, path, query, body)  # fmt: skip

    monkeypatch.setattr(ui_app, "call_api", running)
    page = (await ui.get("/ui/jobs/host-0123456789ab")).text
    assert "Restoring" in page and 'action="/ui/op/cancel-job"' not in page


async def test_the_dedicated_screens_are_in_the_navigation(ui):
    await _signed_in(ui)
    nav = (await ui.get("/ui/")).text.split("<main>")[0]
    for href in ("/ui/status", "/ui/backups", "/ui/config", "/ui/models"):
        assert f'href="{href}"' in nav, href


async def test_scheduled_tasks_are_listed_paused_stopped_run_and_deleted_from_their_screen(
    ui, monkeypatch
):
    """List, pause, run now, stop and delete any user's task from one screen."""
    from app import tasks
    from app.admin import service
    from app.db.models import Channel, PermissionKind
    from app.db.session import session_scope

    async with session_scope() as session:
        user = await service.create_user(session, "Sam")
        identity = await service.add_channel_identity(session, user.id, Channel.TELEGRAM, "111")
        await service.grant_identity_permission(
            session, user.id, identity.id, PermissionKind.CHAT
        )
        task = await tasks.create_task(
            session, user_id=user.id, prompt=PAYLOAD, kind="daily", expr="08:30", actor="t"
        )
        await session.commit()
        task_id = task.id
    ran: list[int] = []

    async def fake_execute(task_id, *, trigger="schedule"):
        ran.append(task_id)
        return {"task_id": task_id, "status": "ok", "delivered": True, "reply_chars": 3}

    monkeypatch.setattr(tasks, "execute", fake_execute)
    await _signed_in(ui)
    assert 'href="/ui/tasks"' in (await ui.get("/ui/")).text
    page = await ui.get("/ui/tasks")
    assert page.status_code == 200 and "daily 08:30 (UTC)" in page.text
    assert PAYLOAD not in page.text, "the prompt is escaped"
    assert "Running." in page.text

    async def click(label):
        """Submit the form of the button `label` as the page has it, hidden fields included."""
        html = (await ui.get("/ui/tasks")).text
        for form in re.findall(r"<form .*?</form>", html, re.S):
            if f">{label}</button>" in form:
                action = re.search(r'action="([^"]+)"', form).group(1)
                fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', form))
                return await ui.post(action, data=fields)
        raise AssertionError(f"no button {label!r}")

    assert (await click("Pause all")).status_code == 303
    assert "<strong>Paused</strong>" in (await ui.get("/ui/tasks")).text
    assert (await click("Resume all")).status_code == 303
    assert "Running." in (await ui.get("/ui/tasks")).text
    assert (await click("Stop")).status_code == 303
    assert '<span class="badge badge-warn">stopped</span>' in (await ui.get("/ui/tasks")).text
    assert (await click("Start")).status_code == 303
    assert '<span class="badge badge-warn">stopped</span>' not in (await ui.get("/ui/tasks")).text
    started = await click("Run now")
    assert started.status_code == 303 and started.headers["location"].startswith("/ui/jobs/")
    from app.admin.jobs import registry

    await registry.get(started.headers["location"].rsplit("/", 1)[1]).task
    assert ran == [task_id]
    assert (await click("Delete")).status_code == 303
    assert "No scheduled task." in (await ui.get("/ui/tasks")).text


async def test_the_bare_address_opens_the_ui_and_the_api_keeps_no_public_route(ui):
    """GET / sends a browser to the UI (then its sign-in page); every other request to /
    still meets the API's key check (the owner opened http://127.0.0.1:8700/ and got a 401)."""
    response = await ui.get("/")
    assert response.status_code == 303
    assert response.headers["location"].endswith("/ui/")
    assert (await ui.get("/", follow_redirects=True)).url.path == "/ui/login"
    assert (await ui.head("/")).status_code == 303
    assert (await ui.post("/")).status_code == 401
    assert (await ui.get("/users")).status_code == 401


async def test_the_browser_icon_is_served_by_the_ui_and_never_counts_as_a_failed_sign_in(ui):
    """A browser asks /favicon.ico by itself. Sent to the API it was a 401 recorded as a
    failed sign-in, and 10 in a minute made the UI's own API calls 429."""
    await _signed_in(ui)
    icons = [await ui.get("/favicon.ico") for _ in range(20)]
    codes = {r.status_code for r in icons}
    page = await ui.get("/ui/config")
    print(f"20 icon requests: {codes}; then /ui/config: {page.status_code}")
    assert codes == {200} and icons[0].headers["content-type"].startswith("image/svg+xml")
    assert (await ui.head("/favicon.ico")).status_code == 200
    assert page.status_code == 200 and "HTTP 429" not in page.text
    assert '<link rel="icon" href="/ui/static/favicon.svg"' in page.text
    assert (await ui.post("/favicon.ico")).status_code == 401, "no other method is public"


# --- forms a person can fill (2026-10-04, owner: "que des champs de saisie à taper en json") ---


async def test_an_id_is_chosen_from_a_list_and_a_list_of_texts_is_typed_one_per_line(ui):
    from app.admin import service
    from app.db.session import session_scope

    async with session_scope() as s:
        user = await service.create_user(s, "Sam")
        agent = await service.create_agent(s, user.id, "helper")
        await s.commit()
        agent_id = agent.id
    csrf = await _signed_in(ui)
    page = (await ui.get("/ui/op/update-agent")).text
    assert '<select name="agent_id" data-flag="--agent-id">' in page
    assert f'<option value="{agent_id}">#{agent_id} helper (Sam)</option>' in page
    assert 'placeholder="one per line"' in page and 'placeholder="JSON"' not in page
    done = await ui.post("/ui/op/update-agent", data={
        "csrf": csrf, "agent_id": str(agent_id), "tools": "mcp__a__one\n\n mcp__b__two \n",
    })  # fmt: skip
    assert done.status_code == 200, done.text[:300]
    async with session_scope() as s:
        from app.db.models import Agent

        assert (await s.get(Agent, agent_id)).tools == ["mcp__a__one", "mcp__b__two"]
    kept = await ui.post("/ui/op/update-agent", data={
        "csrf": csrf, "agent_id": str(agent_id), "tools": '["mcp__c__three"]',
    })  # fmt: skip
    assert kept.status_code == 200
    async with session_scope() as s:
        assert (await s.get(Agent, agent_id)).tools == ["mcp__c__three"], "JSON still accepted"
