"""Every command sent to the Admin API needs an identification token and an authorization
scope, checked before anything is read (mandatory). Runs
scripts/api_auth_matrix.py against the real application: for all routes, 401
without or with a wrong token, 401 for an invalid body without a token, 403 one scope
below the route's, accepted with its own scope, and no route outside the API routes.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_every_route_checks_the_token_then_the_scope():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_", "CHECKPOINT_"))}
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "api_auth_matrix.py")],
        cwd=REPO,
        env={**env, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = result.stdout
    assert result.returncode == 0, out[-3000:] + result.stderr[-2000:]
    total = int(re.search(r"^(\d+) operations on", out, re.M).group(1))
    assert total >= 56
    assert f"refused without a token: {total} of {total}" in out
    assert f"refused with a wrong token: {total} of {total}" in out
    assert f"accepted with their own scope: {total} of {total}" in out
    assert "routes outside the API routes: []" in out
    assert "problems: none" in out


async def test_the_command_line_client_never_calls_without_the_key(monkeypatch):
    from app.admin import client

    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    code, _out, err = await _run(client, "list-users")
    assert code == 2 and "API_SERVER_KEY is not set" in err


async def test_the_command_line_client_sends_the_key_on_every_request(monkeypatch):
    from app.admin import client

    monkeypatch.setenv("API_SERVER_KEY", "client-" + "k" * 30)
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    async with client.open_client("http") as c:
        assert c.headers["authorization"] == "Bearer client-" + "k" * 30
        assert c.headers["x-client"].startswith("cli:")


async def _run(client, *argv):
    import io

    out, err = io.StringIO(), io.StringIO()
    code = await client.amain(list(argv), out, err)
    return code, out.getvalue(), err.getvalue()
