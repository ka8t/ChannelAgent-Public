"""Tests: a chat turn makes no connection outside the machine, the pull token never
reaches a log, and the download code cannot be reached from the inference path.
"""

import ast
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


class _Llama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "hello from the model"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def network(monkeypatch):
    """Records every connection and name lookup, and refuses any that is not this machine."""
    seen = {"local": [], "outside": []}
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def is_local(host) -> bool:
        # asyncio hands the name over as bytes; `localhost` also gets fe80::1%lo0 (macOS)
        host = host.decode() if isinstance(host, bytes) else str(host)
        zone = host.partition("%")[2]
        return (
            host in ("127.0.0.1", "::1", "localhost")
            or host.startswith("127.")
            or zone.startswith("lo")
        )

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if self.family == socket.AF_UNIX or is_local(host):
            seen["local"].append(address)
            return real_connect(self, address)
        seen["outside"].append(("connect", address))
        raise OSError("network blocked by the test")

    def getaddrinfo(host, *a, **k):
        if host is not None and not is_local(host) and host != "":
            seen["outside"].append(("lookup", host))
            raise socket.gaierror("network blocked by the test")
        return real_getaddrinfo(host, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return seen


def test_the_guard_of_this_test_really_records_and_refuses(network):
    with pytest.raises(OSError):
        socket.getaddrinfo("example.org", 80)
    with socket.socket() as raw, pytest.raises(OSError):
        raw.connect(("93.184.216.34", 80))
    assert network["outside"] == [("lookup", "example.org"), ("connect", ("93.184.216.34", 80))]


@pytest.mark.asyncio
async def test_a_chat_turn_reaches_only_the_engine_on_this_machine(fresh_db, monkeypatch, network):
    from app import config
    from app.admin.service import create_agent
    from app.db.models import Channel, User
    from app.db.session import init_db, session_scope
    from app.graph import run_turn

    server = HTTPServer(("127.0.0.1", 0), _Llama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    config.get_settings.cache_clear()
    try:
        await init_db()
        async with session_scope() as session:
            user = User(display_name="Offline")
            session.add(user)
            await session.flush()
            agent = await create_agent(session, user.id, "default")
            await session.commit()
            agent_id = agent.id
        reply = await run_turn(Channel.TELEGRAM, "4242", agent_id, "hello")
    finally:
        server.shutdown()
        config.get_settings.cache_clear()
    print(
        f"turn: reply={reply!r}, connections to this machine={len(network['local'])}, "
        f"connections or lookups outside={len(network['outside'])}"
    )
    assert reply == "hello from the model"
    assert network["outside"] == []
    assert len(network["local"]) >= 1  # the guard was live: the engine call went through it


def _imports(path: Path) -> set:
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def test_the_inference_path_cannot_reach_the_download_code():
    files = [REPO / "app" / "graph.py", *sorted((REPO / "app" / "channels").glob("*.py"))]
    forbidden = ("app.admin.models", "app.security.outbound", "huggingface")
    for path in files:
        for module in _imports(path):
            assert not any(module.startswith(f) for f in forbidden), (path.name, module)
    hub = [p for p in (REPO / "app").rglob("*.py") if "huggingface" in p.read_text()]
    assert {p.name for p in hub} <= {"config.py", "outbound.py", "models.py"}


def test_the_pull_token_is_redacted_from_any_log(monkeypatch, caplog):
    from app import logging_setup
    from app.config import get_settings

    token = "hf_ThisTokenMustNeverAppearInALog_0123456789"
    monkeypatch.setenv("HF_TOKEN", token)
    get_settings.cache_clear()
    assert token in logging_setup.configured_secrets()
    logging_setup.install_redaction()
    with caplog.at_level(logging.INFO):
        logging.getLogger("httpx").info("HTTP Request: GET https://x/?token=%s", token)
        logging.getLogger("channelagent").warning("Authorization: Bearer %s", token)
    get_settings.cache_clear()
    assert token not in "\n".join(r.getMessage() for r in caplog.records)
    assert token not in logging_setup.scrub(f"failed with {token}")


def test_start_sh_points_from_the_removed_models_shortcut_to_the_api_commands():
    #`--models` was removed; the generated commands remain, through --admin.
    script = (REPO / "start.sh").read_text()
    assert '--models) MODE="models"' not in script
    hint = next(line for line in script.splitlines() if line.strip().startswith("--models)"))
    for command in ("list-models", "pull-model", "import-model", "delete-model"):
        assert command in hint


def test_a_name_given_as_bytes_is_judged_like_the_same_name_as_text(network):
    import socket as s

    assert s.getaddrinfo(b"localhost", 80)
    with pytest.raises(OSError):
        s.getaddrinfo(b"example.org", 80)
    assert network["outside"] == [("lookup", b"example.org")]
