"""Tests: ./start.sh --status and --stop.

Every run is in a temporary directory holding a copy of start.sh. Components
are real processes: a stub `llama-server` (a script of that name serving
/health), a stub Admin API, a fake `app.main`, and a fake `docker` that only
records what it was asked. Nothing touches the real application, the real
llama-server or Docker.
"""

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

REPO = Path(__file__).resolve().parent.parent

LLAMA_STUB = textwrap.dedent(
    """
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            code = 200 if self.path == "/health" else 404
            self.send_response(code); self.send_header("Content-Length", "2"); self.end_headers()
            self.wfile.write(b"ok")

    HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), H).serve_forever()
    """
)

FAKE_DOCKER = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_DOCKER_LOG"
echo "LLAMA_SERVER_URL=${LLAMA_SERVER_URL:-}" >> "$FAKE_DOCKER_LOG.env"
if [ "$1 $2 $3" = "compose ps --status" ]; then
  [ -f "$FAKE_DOCKER_STATE" ] && echo "0123456789abcdef"
  exit 0
fi
if [ "$1" = "inspect" ]; then
  [ -f "$FAKE_DOCKER_STATE" ] && cat "$FAKE_DOCKER_STATE"
  exit 0
fi
if [ "$1 $2 $3" = "compose stop channelagent" ]; then
  rm -f "$FAKE_DOCKER_STATE"
  exit 0
fi
exit 0
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait(condition, timeout=15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.1)
    return False


class Box:
    """A sandbox directory plus the helpers to start stub components in it."""

    def __init__(self, path: Path):
        self.path = path
        self.procs: list[subprocess.Popen] = []
        self.llama_port = _free_port()
        self.api_port = _free_port()
        shutil.copy(REPO / "start.sh", path / "start.sh")
        shutil.copy(REPO / ".env.example", path / ".env.example")
        (path / "app").mkdir()
        shutil.copy(REPO / "app" / "settings_rules.py", path / "app" / "settings_rules.py")
        (path / "app" / "admin").mkdir()
        shutil.copy(
            REPO / "app" / "admin" / "credentials.py", path / "app" / "admin" / "credentials.py"
        )
        (path / "bin").mkdir()
        (path / "bin" / "docker").write_text(FAKE_DOCKER)
        (path / "bin" / "docker").chmod(0o755)
        (path / "bin" / "llama-server").write_text(LLAMA_STUB)
        self.docker_log = path / "docker.log"
        self.docker_state = path / "docker.running"
        self.write_env(api_key="k" * 32)

    def write_env(self, api_key: str = ""):
        (self.path / ".env").write_text(
            f"LLAMA_PORT={self.llama_port}\nAPI_SERVER_PORT={self.api_port}\n"
            f"API_SERVER_KEY={api_key}\nENCRYPTION_KEY=x\n"
        )
        (self.path / ".env").chmod(0o600)

    def start_llama(self, pidfile=True) -> subprocess.Popen:
        script = str(self.path / "bin" / "llama-server")
        proc = subprocess.Popen([sys.executable, script, "--port", str(self.llama_port)])
        self.procs.append(proc)
        assert _wait(lambda: httpx_ok(f"http://127.0.0.1:{self.llama_port}/health"))
        if pidfile:
            (self.path / ".llama-server.pid").write_text(str(proc.pid))
        return proc

    def start_foreign_llama_lookalike(self) -> subprocess.Popen:
        """Serves /health on the llama port, but its command line does not say
        llama-server: a process this script did not start.
        """
        code = LLAMA_STUB
        proc = subprocess.Popen(
            [sys.executable, "-c", code, "--port", str(self.llama_port)],
        )
        self.procs.append(proc)
        assert _wait(lambda: httpx_ok(f"http://127.0.0.1:{self.llama_port}/health"))
        return proc

    def start_native_app(self) -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "app.main"])
        self.procs.append(proc)
        (self.path / ".app.pid").write_text(str(proc.pid))
        return proc

    def start_unrelated(self) -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        self.procs.append(proc)
        return proc

    def _without_docker(self) -> Path:
        """A PATH of every tool of /usr/bin and /bin but docker: a Linux CI machine has a real
        docker there, so leaving the stub out of PATH is not enough."""
        folder = self.path / "bin-without-docker"
        if not folder.exists():
            folder.mkdir()
            for directory in (Path("/usr/bin"), Path("/bin")):
                for tool in directory.iterdir():
                    link = folder / tool.name
                    if tool.name != "docker" and not link.exists():
                        link.symlink_to(tool)
        return folder

    def run(self, *args: str, docker=True, extra_env=None):
        path = f"{self.path / 'bin'}:/usr/bin:/bin" if docker else str(self._without_docker())
        env = {
            "PATH": path,
            "HOME": str(self.path),
            "FAKE_DOCKER_LOG": str(self.docker_log),
            "FAKE_DOCKER_STATE": str(self.docker_state),
            **(extra_env or {}),
        }
        return subprocess.run(
            ["bash", "start.sh", *args],
            cwd=self.path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )

    def docker_calls(self) -> list[str]:
        return self.docker_log.read_text().splitlines() if self.docker_log.exists() else []

    def cleanup(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=10)


def httpx_ok(url: str) -> bool:
    try:
        return httpx.get(url, timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture
def box(tmp_path):
    b = Box(tmp_path)
    yield b
    b.cleanup()


@pytest.fixture
def api_stub(box):
    """An Admin API stand-in that answers 401 without a key, like the real one."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", box.api_port), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def _line(output: str, label: str) -> str:
    return next(line for line in output.splitlines() if line.strip().startswith(label))


# --- --status ---


def test_status_reports_everything_down(box):
    result = box.run("--status")
    assert result.returncode == 0, result.stderr
    assert "app (container)  : not running" in result.stdout
    assert "app (native)     : not running" in result.stdout
    assert f"Admin API        : down at http://127.0.0.1:{box.api_port}" in result.stdout
    assert f"llama-server     : down on port {box.llama_port}" in result.stdout


def test_status_reports_each_component_up(box, api_stub):
    llama = box.start_llama()
    native = box.start_native_app()
    box.docker_state.write_text("running")

    out = box.run("--status").stdout

    assert "app (container)  : running (0123456789ab)" in out
    assert f"app (native)     : running (pid {native.pid})" in out
    assert f"up at http://127.0.0.1:{box.api_port} (HTTP 401)" in _line(out, "Admin API")
    assert f"up on port {box.llama_port} (pid {llama.pid}, started by start.sh)" in out


def test_status_looks_for_the_api_at_api_url_wherever_it_runs(box, api_stub):
    """The API may run on another machine than this script: --status probes API_URL,
    not the port this machine would listen on."""
    env = (
        (box.path / ".env")
        .read_text()
        .replace(f"API_SERVER_PORT={box.api_port}", f"API_SERVER_PORT={_free_port()}")
    )
    (box.path / ".env").write_text(env + f"API_URL=http://localhost:{box.api_port}\n")
    out = box.run("--status").stdout
    assert f"up at http://localhost:{box.api_port} (HTTP 401)" in _line(out, "Admin API")


def test_status_says_when_the_llama_server_was_not_started_by_the_script(box):
    box.start_foreign_llama_lookalike()
    out = box.run("--status").stdout
    # "(not started by start.sh)", or "(pid N, not started by start.sh)" where the script can
    # find the process (Linux)
    assert "not started by start.sh)" in _line(out, "llama-server")


def test_a_pid_file_naming_an_unrelated_process_does_not_count_as_the_app(box):
    other = box.start_unrelated()
    (box.path / ".app.pid").write_text(str(other.pid))
    assert "app (native)     : not running" in box.run("--status").stdout


def test_status_without_docker_says_so(box):
    out = box.run("--status", docker=False).stdout
    assert "docker not available" in _line(out, "app (container)")


def test_status_says_the_api_is_disabled_without_a_key(box):
    box.write_env(api_key="")
    assert "disabled (API_SERVER_KEY is not set)" in box.run("--status").stdout


def test_status_reads_and_changes_nothing(box):
    before = (box.path / ".env").read_bytes()
    box.run("--status")
    assert (box.path / ".env").read_bytes() == before
    assert not list(box.path.glob(".env.bak*"))
    assert box.docker_calls() == ["compose ps --status running --status restarting -q channelagent"]


def test_status_names_a_container_restarting_in_a_loop(box):
    """A container crashing at start is "restarting"; --status said "not running"."""
    box.docker_state.write_text("restarting")
    line = _line(box.run("--status").stdout, "app (container)")
    assert "restarting in a loop (0123456789ab)" in line
    assert "docker compose logs channelagent" in line


def test_status_says_running_for_a_running_container(box):
    box.docker_state.write_text("running")
    assert "running (0123456789ab)" in _line(box.run("--status").stdout, "app (container)")


# --- --stop ---


def test_stop_stops_a_container_restarting_in_a_loop(box):
    """--stop used to say "not running" and leave the crash loop going."""
    box.docker_state.write_text("restarting")
    result = box.run("--stop")
    assert result.returncode == 0, result.stderr
    assert "compose stop channelagent" in box.docker_calls()
    assert not box.docker_state.exists()


def test_stop_stops_the_native_app_and_the_container_but_not_the_llama_server(box):
    llama = box.start_llama()
    native = box.start_native_app()
    box.docker_state.write_text("running")

    result = box.run("--stop")

    assert result.returncode == 0, result.stderr
    assert _wait(lambda: native.poll() is not None), "the native app is stopped"
    assert not (box.path / ".app.pid").exists()
    assert "compose stop channelagent" in box.docker_calls()
    assert not box.docker_state.exists()
    assert llama.poll() is None, "llama-server keeps running without --all"
    assert "left as it is" in result.stdout


def test_stop_all_also_stops_the_llama_server_it_started(box):
    llama = box.start_llama()
    result = box.run("--stop", "--all")
    assert result.returncode == 0, result.stderr
    assert _wait(lambda: llama.poll() is not None)
    assert not (box.path / ".llama-server.pid").exists()
    assert not httpx_ok(f"http://127.0.0.1:{box.llama_port}/health")


def test_stop_never_signals_a_process_that_is_not_ours(box):
    """The pid file names a foreign server that answers on the llama port: it
    must be left running, and the file kept.
    """
    foreign = box.start_foreign_llama_lookalike()
    (box.path / ".llama-server.pid").write_text(str(foreign.pid))

    result = box.run("--stop", "--all")

    assert foreign.poll() is None, "the foreign process is untouched"
    assert httpx_ok(f"http://127.0.0.1:{box.llama_port}/health")
    assert (box.path / ".llama-server.pid").read_text() == str(foreign.pid)
    assert "left alone" in result.stderr


def test_stop_never_signals_an_unrelated_process_named_by_the_app_pid_file(box):
    other = box.start_unrelated()
    (box.path / ".app.pid").write_text(str(other.pid))
    box.run("--stop")
    assert other.poll() is None
    assert (box.path / ".app.pid").exists()


def test_a_stale_pid_file_is_removed_and_nothing_is_killed(box):
    gone = box.start_unrelated()
    gone.kill()
    gone.wait()
    (box.path / ".app.pid").write_text(str(gone.pid))
    result = box.run("--stop")
    assert result.returncode == 0
    assert not (box.path / ".app.pid").exists()
    assert "stale" in result.stdout


def test_stop_only_asks_docker_to_stop_the_channelagent_service(box):
    box.docker_state.write_text("running")
    box.run("--stop", "--all")
    calls = box.docker_calls()
    assert "compose stop channelagent" in calls
    assert not [c for c in calls if c.split()[0] in ("kill", "rm", "stop", "system")]
    assert not [c for c in calls if " down" in c or " rm" in c or " kill" in c]


def test_stop_with_nothing_running_is_fine(box):
    result = box.run("--stop", "--all")
    assert result.returncode == 0
    assert "compose stop" not in " ".join(box.docker_calls())


def test_stop_rejects_an_unknown_argument(box):
    llama = box.start_llama()
    result = box.run("--stop", "--everything")
    assert result.returncode == 1 and "Usage" in result.stderr
    assert llama.poll() is None


# --- the notice after --set ---


def test_set_says_the_change_applies_at_the_next_start_and_restarts_nothing(box):
    box.docker_state.write_text("running")
    result = box.run("--config", "LLAMA_CTX_SIZE=32768")
    assert result.returncode == 0, result.stderr
    assert "Applies at the next start" in result.stdout
    assert box.docker_calls() == [], "--set never touches Docker"
    assert box.docker_state.exists(), "and the container keeps running"


# --- the native start: its pid, and what Ctrl+C cleans up ---

FAKE_APP = textwrap.dedent(
    """
    import os, signal, sys, time, urllib.request
    from pathlib import Path

    def llama_up():
        port = os.environ["LLAMA_PORT"]
        try:
            url = f"http://127.0.0.1:{port}/health"
            return urllib.request.urlopen(url, timeout=1).status == 200
        except OSError:
            return False

    def on_int(*_):
        time.sleep(0.5)  # the real app stops Telegram, the API and the email adapter
        Path("app.cleanup").write_text("llama up during cleanup" if llama_up() else "llama down")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)  # how CPython ends after a KeyboardInterrupt

    Path("app.env").write_text(os.environ.get("LLAMA_SERVER_URL", ""))
    if "app.admin.client" in sys.argv:  # --admin / --models: only the engine URL matters here
        sys.exit(0)
    signal.signal(signal.SIGINT, on_int)
    Path("app.started").write_text(str(os.getpid()))
    while True:
        time.sleep(0.1)
    """
)


@pytest.fixture
def native(box):
    """Make `./start.sh --native` runnable in the sandbox: a stub llama-server started by
    the script itself, and a `.venv` whose python3 runs FAKE_APP as `-m app.main`."""
    if sys.platform != "darwin":
        pytest.skip("start.sh starts llama-server natively on macOS only")
    stubs = box.path / "stubs"
    stubs.mkdir()
    llama = box.path / "bin" / "llama-server"
    llama.write_text(f"#!{sys.executable}\n" + LLAMA_STUB)
    llama.chmod(0o755)
    (box.path / "models").mkdir()
    (box.path / "models" / "m.gguf").write_text("")
    (box.path / "fake_app.py").write_text(FAKE_APP)
    (stubs / "python3").write_text(
        f'#!/bin/bash\nexec {sys.executable} "{box.path}/fake_app.py" "$@"\n'
    )
    (stubs / "pip").write_text("#!/bin/bash\nexit 0\n")
    for stub in ("python3", "pip"):
        (stubs / stub).chmod(0o755)
    (box.path / ".venv" / "bin").mkdir(parents=True)
    (box.path / ".venv" / "bin" / "activate").write_text(f'PATH="{stubs}:$PATH"\n')
    with (box.path / ".env").open("a") as env:
        env.write(f"LLAMA_SERVER_BIN={llama}\nMODELS_DIR={box.path}/models\nMODEL_FILE=m.gguf\n")
    started = []

    def start(extra_env=None, wait=True) -> subprocess.Popen:
        proc = subprocess.Popen(
            ["bash", "start.sh", "--native"],
            cwd=box.path,
            env={
                "PATH": f"{box.path / 'bin'}:/usr/bin:/bin:/usr/sbin",
                "HOME": str(box.path),
                "FAKE_DOCKER_LOG": str(box.docker_log),
                "FAKE_DOCKER_STATE": str(box.docker_state),
                **(extra_env or {}),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # its own process group, like a terminal's foreground job
        )
        started.append(proc)
        if not wait:
            return proc
        if not _wait(lambda: (box.path / "app.started").exists(), timeout=60):
            os.killpg(proc.pid, signal.SIGKILL)
            pytest.fail("the app did not start:\n" + proc.communicate(timeout=10)[0][-2000:])
        return proc

    yield start
    for proc in started:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    llama_pid = box.path / ".llama-server.pid"
    if llama_pid.exists():
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(llama_pid.read_text()), signal.SIGKILL)


def _command(pid: int) -> str:
    return subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
    ).stdout


def test_the_native_start_records_the_app_s_own_pid(box, native):
    """--status and --stop find the application through .app.pid."""
    proc = native()
    pid = int((box.path / ".app.pid").read_text())
    assert pid == int((box.path / "app.started").read_text()), "the pid is the app's own"
    assert "app.main" in _command(pid)
    os.killpg(proc.pid, signal.SIGINT)
    proc.communicate(timeout=30)


def test_ctrl_c_stops_the_app_then_the_llama_server_like_stop_all(box, native):
    proc = native()
    llama_pid = int((box.path / ".llama-server.pid").read_text())
    app_pid = int((box.path / ".app.pid").read_text())
    assert _alive(llama_pid)
    os.killpg(proc.pid, signal.SIGINT)  # what the terminal sends on Ctrl+C
    output, _ = proc.communicate(timeout=30)
    assert proc.returncode == 130, output
    assert (box.path / "app.cleanup").read_text() == "llama up during cleanup", (
        "the app finished its own shutdown before llama-server was stopped"
    )
    assert _wait(lambda: not _alive(app_pid) and not _alive(llama_pid))
    assert not (box.path / ".app.pid").exists()
    assert not (box.path / ".llama-server.pid").exists()
    assert "llama-server stopped (pid" in output, output
    # Only the read-only "is the container running?" query of the start (one instance at a time).
    started_or_stopped = [c for c in box.docker_calls() if not c.startswith("compose ps ")]
    assert started_or_stopped == [], "the container is not part of a native run"


def test_stop_from_another_terminal_leaves_the_llama_server_running(box, native):
    """./start.sh --stop (without --all) keeps its meaning: only Ctrl+C cleans up llama."""
    proc = native()
    llama_pid = int((box.path / ".llama-server.pid").read_text())
    result = box.run("--stop")
    assert result.returncode == 0, result.stderr
    output, _ = proc.communicate(timeout=30)
    assert "Ctrl+C: cleaning up" not in output
    assert _alive(llama_pid), "llama-server keeps running"
    assert (box.path / ".llama-server.pid").exists()


# --- help and unknown commands ---


@pytest.mark.parametrize("arg", ["--help", "-h", "help"])
def test_help_prints_the_commands_and_starts_nothing(box, arg):
    box.start_llama()
    result = box.run(arg)
    assert result.returncode == 0, result.stderr
    assert "Usage: ./start.sh" in result.stdout
    assert box.docker_calls() == []


@pytest.mark.parametrize("arg", ["--stauts", "--nativ", "status", "--docker-compose"])
def test_an_unknown_command_is_refused_and_starts_nothing(box, arg):
    """It used to be taken as Docker mode: docker compose up --build, exit 0."""
    box.start_llama()
    result = box.run(arg)
    assert result.returncode == 2
    assert f"Unknown command: {arg}" in result.stderr
    assert "Usage: ./start.sh" in result.stderr
    assert box.docker_calls() == []


@pytest.mark.parametrize("args", [(), ("--docker",)])
def test_docker_mode_without_a_command_or_with_docker(box, args):
    box.start_llama()
    result = box.run(*args)
    assert result.returncode == 0, result.stderr
    assert box.docker_calls() == ["compose up --build"]


@pytest.mark.parametrize(
    "args, expected",
    [
        (("--docker", "--no-build"), "compose up"),
        (("--docker", "--detach"), "compose up -d --build"),
        (("--docker", "--detach", "--build"), "compose up -d --build"),
        (("--docker", "--no-build", "--detach"), "compose up -d"),
    ],
)
def test_docker_mode_rebuilds_or_not_as_asked(box, args, expected):
    """--build / --no-build; detached or not, the image is rebuilt unless --no-build (a
    stale image crashed on the real database's newer migrations, 2026-10-05)."""
    box.start_llama()
    result = box.run(*args)
    assert result.returncode == 0, result.stderr
    assert box.docker_calls() == [expected]


def test_an_unknown_docker_option_is_refused_and_starts_nothing(box):
    box.start_llama()
    result = box.run("--docker", "--rebuild")
    assert result.returncode == 2
    assert "Unknown option for --docker: --rebuild" in result.stderr
    assert box.docker_calls() == []


def test_docker_mode_gives_the_container_the_port_the_engine_runs_on(box):
    """A local engine runs on LLAMA_PORT: the container gets host.docker.internal:LLAMA_PORT
    even when .env's URL names another port (the live application found no engine with
    LLAMA_PORT=8090 and :8080 in the URL, 2026-09-28)."""
    env = box.path / ".env"
    env.write_text(env.read_text() + "LLAMA_SERVER_URL=http://host.docker.internal:8080\n")
    box.start_llama()
    result = box.run("--docker", "--detach")
    assert result.returncode == 0, result.stderr
    sent = (box.path / "docker.log.env").read_text().splitlines()
    assert sent == [f"LLAMA_SERVER_URL=http://host.docker.internal:{box.llama_port}"]
    compose = (REPO / "docker-compose.yml").read_text()
    assert 'LLAMA_SERVER_URL: "${LLAMA_SERVER_URL:-}"' in compose


# --- one instance at a time ---
# A native start while the container ran (2026-09-28) lost port 8700 to it
# ("[Errno 48] address already in use"), polled the mailbox next to it and wrote
# "native" into .run-mode. Each refusal happens before anything is started or written.


def _refused_before_anything(box, result):
    assert result.returncode == 1
    assert not (box.path / ".run-mode").exists()
    assert not (box.path / ".venv").exists()
    assert "compose up" not in " ".join(box.docker_calls())


def test_native_is_refused_while_the_container_runs(box):
    box.docker_state.write_text("running")
    result = box.run("--native")
    _refused_before_anything(box, result)
    assert "channelagent container is already running (0123456789ab)" in result.stderr


def test_native_is_refused_when_the_api_port_is_taken(box, api_stub):
    result = box.run("--native")
    _refused_before_anything(box, result)
    assert f"Port {box.api_port} (API_SERVER_PORT) is already in use on 127.0.0.1" in result.stderr


@pytest.mark.parametrize("args", [("--native",), ("--docker",)])
def test_a_start_is_refused_while_the_native_app_runs(box, args):
    native = box.start_native_app()
    result = box.run(*args)
    _refused_before_anything(box, result)
    assert f"native app is already running (pid {native.pid})" in result.stderr


def test_the_commands_that_talk_to_the_running_app_are_not_refused(box, api_stub, native):
    """Only the two starts are refused: the first version of the guard also refused
    --describe, --api, --admin and --models while the app ran (2026-09-28). --api is
    --admin COMMAND now."""
    box.start_native_app()
    box.docker_state.write_text("running")
    result = box.run("--admin", "whoami")
    assert result.returncode == 0, result.stderr
    assert "already running" not in result.stderr and "already in use" not in result.stderr


def test_the_help_names_every_command_of_the_dispatch():
    import re

    script = (REPO / "start.sh").read_text()
    dispatched = set(re.findall(r'^  (?:""\|)?(--[a-z-]+)(?:\|[^)]*)?\)', script, re.M))
    usage = script[script.index("usage() {") : script.index("USAGE\n}")]
    # 10 dispatched names (no command = --docker; -h and help are spellings of --help).
    assert dispatched == {
        "--admin",
        "--chat",
        "--config",
        "--docker",
        "--help",
        "--native",
        "--rekey",
        "--restore",
        "--status",
        "--stop",
    }, dispatched
    missing = sorted(c for c in dispatched if c not in usage)
    assert missing == [], missing


# --- a remote inference engine in native mode ---


@pytest.fixture
def remote_engine(tmp_path_factory):
    """An engine the rule calls remote that still answers here: https://0.0.0.0:PORT (not
    loopback for the rule; macOS connects 0.0.0.0 to this machine), with a test CA that
    curl (CURL_CA_BUNDLE) and Python (SSL_CERT_FILE) trust."""
    import datetime
    import ipaddress
    import ssl
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    if sys.platform != "darwin":
        pytest.skip("relies on 0.0.0.0 reaching this machine, measured on macOS")
    folder = tmp_path_factory.mktemp("engine")
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test engine")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("0.0.0.0"))]),
            False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
        .sign(key, hashes.SHA256())
    )
    ca, private = folder / "ca.pem", folder / "key.pem"
    ca.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    class Health(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    server = HTTPServer(("127.0.0.1", 0), Health)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ca, private)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    engine = SimpleNamespace(
        url=f"https://0.0.0.0:{server.server_port}",
        env={"CURL_CA_BUNDLE": str(ca), "SSL_CERT_FILE": str(ca)},
    )
    yield engine
    server.shutdown()


def _set_engine(box, url):
    with (box.path / ".env").open("a") as env:
        env.write(f"LLAMA_SERVER_URL={url}\n")


def test_native_uses_a_remote_engine_and_starts_none(box, native, remote_engine):
    _set_engine(box, remote_engine.url)
    proc = native(remote_engine.env)
    assert (box.path / "app.env").read_text() == remote_engine.url
    assert not (box.path / ".llama-server.pid").exists(), "no local engine started"
    os.killpg(proc.pid, signal.SIGINT)
    output, _ = proc.communicate(timeout=30)
    assert proc.returncode == 130, output
    assert f"remote, up at {remote_engine.url}" in output
    assert "left alone" in output
    assert "starting it" not in output
    assert _curl_ok(remote_engine), "Ctrl+C left the remote engine alone"


def _curl_ok(engine) -> bool:
    result = subprocess.run(
        ["curl", "-sf", "--max-time", "3", f"{engine.url}/health"],
        env={**os.environ, **engine.env},
        capture_output=True,
    )
    return result.returncode == 0


def test_native_stops_when_the_remote_engine_does_not_answer(box, native, remote_engine):
    _set_engine(box, f"https://0.0.0.0:{_free_port()}")
    proc = native(remote_engine.env, wait=False)
    output, _ = proc.communicate(timeout=60)
    assert proc.returncode == 1, output
    assert "remote inference engine does not answer" in output
    assert not (box.path / "app.started").exists()
    assert not (box.path / ".llama-server.pid").exists()


def test_native_refuses_a_remote_engine_over_plain_http(box, native):
    _set_engine(box, "http://192.0.2.10:8080")
    proc = native(wait=False)
    output, _ = proc.communicate(timeout=60)
    assert proc.returncode == 1, output
    assert "plain http" in output
    assert not (box.path / "app.started").exists()
    assert not (box.path / ".llama-server.pid").exists()


def test_status_reports_a_remote_engine(box, remote_engine):
    _set_engine(box, remote_engine.url)
    up = box.run("--status", extra_env=remote_engine.env).stdout
    assert f"remote, up at {remote_engine.url}" in _line(up, "llama-server")
    (box.path / ".env").write_text(
        (box.path / ".env").read_text().replace(remote_engine.url, "https://0.0.0.0:1")
    )
    down = box.run("--status", extra_env=remote_engine.env).stdout
    assert "remote, down at https://0.0.0.0:1" in _line(down, "llama-server")


@pytest.mark.parametrize("args", [("--admin", "whoami"), ("--admin", "list-models")])
def test_api_and_models_in_process_use_the_remote_engine(box, native, remote_engine, args):
    _set_engine(box, remote_engine.url)
    result = box.run(*args, extra_env={"PATH": f"{box.path / 'bin'}:/usr/bin:/bin:/usr/sbin"})
    assert result.returncode == 0, result.stderr
    assert (box.path / "app.env").read_text() == remote_engine.url


def test_api_in_process_still_uses_the_local_engine_by_default(box, native):
    result = box.run("--admin", "whoami")
    assert result.returncode == 0, result.stderr
    assert (box.path / "app.env").read_text() == f"http://localhost:{box.llama_port}"


# --- the scheduled backup in --status ---


def test_status_shows_the_scheduled_backup_read_with_the_key(box):
    """The stub answers the schedule only with the right key, and 401 otherwise, like the API."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    key = "k" * 32
    seen = []

    class Api(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            if (
                self.path == "/backups/schedule"
                and self.headers.get("Authorization") == f"Bearer {key}"
            ):
                body = json.dumps(
                    {
                        "enabled": True,
                        "failing": False,
                        "interval_minutes": 1440,
                        "keep": 7,
                        "last_success_at": "2026-09-27T10:00:00Z",
                        "last_run_at": "2026-09-27T10:00:00Z",
                        "last_error": None,
                        "next_run_at": "2026-09-28T10:00:00Z",
                    }
                ).encode()
                self.send_response(200)
            else:
                body = b"{}"
                self.send_response(401)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", box.api_port), Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = box.run("--status").stdout
    finally:
        server.shutdown()
    line = _line(out, "Backups")
    assert "every 1440 min, keep 7; last ok 2026-09-27T10:00:00Z" in line
    assert ("/backups/schedule", f"Bearer {key}") in seen
    assert "llama-server" in out, "the report goes on after the backup line"


# --- the host helper ---


def _start_fake_helper(box) -> subprocess.Popen:
    """A process whose command line names app.host.helper, as start.sh starts it."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "-m", "app.host.helper"]
    )
    box.procs.append(proc)
    (box.path / ".host-helper.pid").write_text(str(proc.pid))
    return proc


def test_stop_leaves_the_host_helper_and_stop_all_stops_it(box):
    helper = _start_fake_helper(box)
    assert box.run("--stop").returncode == 0
    assert helper.poll() is None, "the helper runs start.sh --stop itself: it must survive it"
    result = box.run("--stop", "--all")
    assert result.returncode == 0, result.stderr
    assert _wait(lambda: helper.poll() is not None)
    assert "host helper stopped" in result.stdout
    assert not (box.path / ".host-helper.pid").exists()


def test_status_shows_the_host_helper(box):
    assert "Host helper      : off" in box.run("--status").stdout
    env = box.path / ".env"
    env.write_text(env.read_text() + "HOST_HELPER_ENABLED=true\n")
    assert "enabled, not running" in box.run("--status").stdout
    helper = _start_fake_helper(box)
    assert f"running (pid {helper.pid}, 127.0.0.1:8701)" in box.run("--status").stdout


def test_stop_with_api_url_acts_remotely_but_never_from_the_helper(box):
    """With API_URL set, --stop is the remote host's stop; the helper's own run of
    start.sh --stop carries START_SH_LOCAL=1 and stops this machine's application,
    otherwise it would call the API that called it."""
    env = box.path / ".env"
    env.write_text(env.read_text() + "API_URL=https://agent.example.org\n")
    native = box.start_native_app()
    remote = box.run("--stop")
    assert remote.returncode == 1 and "No .venv here" in remote.stderr
    assert "through its host helper" in remote.stdout
    assert native.poll() is None, "the local application is not touched"
    local = box.run("--stop", extra_env={"START_SH_LOCAL": "1"})
    assert local.returncode == 0, local.stderr
    assert _wait(lambda: native.poll() is not None)


def test_stop_all_is_refused_with_api_url(box):
    env = box.path / ".env"
    env.write_text(env.read_text() + "API_URL=https://agent.example.org\n")
    result = box.run("--stop", "--all")
    assert result.returncode == 1 and "Run --stop --all on that machine" in result.stderr


def test_the_host_helper_has_no_command_of_its_own_any_more(box):
    # It starts with the application, stops with --stop --all, shows in --status.
    for sub in ("start", "stop", "status"):
        args = ("--host-helper", sub)
        gone = box.run(*args)
        assert gone.returncode == 2, args
        assert "starts with the application" in gone.stderr and "--stop --all" in gone.stderr
    assert "Host helper      : off" in box.run("--status").stdout


# --- the engine's bearer key ---


def _recording_llama(box) -> Path:
    """The stub llama-server, recording what it was given before it serves."""
    record = box.path / "llama.given"
    llama = box.path / "bin" / "llama-server"
    llama.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"json.dump({{'key': os.environ.get('LLAMA_API_KEY'), 'argv': sys.argv, "
        f"'env': sorted(os.environ), 'umask': os.umask(0o077)}}, open({str(record)!r}, 'w'))\n"
        + LLAMA_STUB
    )
    llama.chmod(0o755)
    return record


def test_the_engine_gets_its_key_in_its_environment_never_on_its_command_line(box, native):
    import json
    import secrets

    key = secrets.token_hex(24)
    with (box.path / ".env").open("a") as env:
        env.write(f"LLAMA_SERVER_API_KEY={key}\n")
    record = _recording_llama(box)
    proc = native()
    given = json.loads(record.read_text())
    assert given["key"] == key
    assert key not in " ".join(given["argv"]) and "--api-key" not in given["argv"]
    assert "ENCRYPTION_KEY" not in given["env"], "no other secret of .env"
    llama_pid = int((box.path / ".llama-server.pid").read_text())
    assert key not in _command(llama_pid), "invisible to ps"
    os.killpg(proc.pid, signal.SIGINT)
    proc.communicate(timeout=30)


@pytest.mark.parametrize("cap", ["", "0"])
def test_the_engine_saves_slots_in_data_slots_readable_by_this_user_alone(box, native, cap):
    """--slot-save-path is the folder app/slot_cache.py reads (data/slots, 700) and the
    engine runs with umask 077, so its saved slots are 600; SLOT_CACHE_MAX_MB=0 leaves it out."""
    import json

    if cap:
        with (box.path / ".env").open("a") as env:
            env.write(f"SLOT_CACHE_MAX_MB={cap}\n")
    record = _recording_llama(box)
    proc = native()
    given = json.loads(record.read_text())
    os.killpg(proc.pid, signal.SIGINT)
    proc.communicate(timeout=30)
    if cap == "0":
        assert "--slot-save-path" not in given["argv"]
        return
    folder = Path(given["argv"][given["argv"].index("--slot-save-path") + 1])
    assert folder == (box.path / "data" / "slots").resolve()
    assert oct(folder.stat().st_mode & 0o777) == "0o700"
    assert given["umask"] == 0o077


def test_without_a_key_the_engine_gets_none(box, native):
    import json

    record = _recording_llama(box)
    proc = native()
    assert json.loads(record.read_text())["key"] is None
    os.killpg(proc.pid, signal.SIGINT)
    proc.communicate(timeout=30)


# --- the headless browser ---


def _recording_python(box) -> Path:
    """The stub python3 of the sandbox, recording a `-m playwright` call instead of running it."""
    record = box.path / "playwright.called"
    stub = box.path / "stubs" / "python3"
    stub.write_text(
        "#!/bin/bash\n"
        f'if [ "$1 $2" = "-m playwright" ]; then\n'
        f'  echo "$* | $PLAYWRIGHT_BROWSERS_PATH" > {record}; exit 0\nfi\n'
        f'exec {sys.executable} "{box.path}/fake_app.py" "$@"\n'
    )
    stub.chmod(0o755)
    (box.path / "stubs" / "python").write_text(stub.read_text())
    (box.path / "stubs" / "python").chmod(0o755)
    return record


def test_the_browser_is_installed_in_the_repository_only_when_switched_on(box, native):
    record = _recording_python(box)
    with (box.path / ".env").open("a") as env:
        env.write("WEB_FETCH_BROWSER=true\n")
    proc = native()
    os.killpg(proc.pid, signal.SIGINT)
    proc.communicate(timeout=30)
    call, path = record.read_text().strip().split(" | ")
    assert call == "-m playwright install chromium-headless-shell"
    assert path == str(box.path / "vendor" / "playwright")


def test_the_browser_is_not_installed_when_off_or_already_there(box, native):
    record = _recording_python(box)
    proc = native()
    os.killpg(proc.pid, signal.SIGINT)
    proc.communicate(timeout=30)
    assert not record.exists()
    (box.path / "vendor" / "playwright" / "chromium_headless_shell-1").mkdir(parents=True)
    with (box.path / ".env").open("a") as env:
        env.write("WEB_FETCH_BROWSER=true\n")
    proc = native()
    os.killpg(proc.pid, signal.SIGINT)
    proc.communicate(timeout=30)
    assert not record.exists()
