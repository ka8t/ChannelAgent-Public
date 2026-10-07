"""Tests: no secret reaches a log line, whatever the level.

The Telegram bot token is part of every Bot API URL, and httpx logs the URL of
each request at INFO. Records are scrubbed when they are created, so it holds
for every logger and every handler, including handlers installed later (uvicorn
replaces its own).
"""

import io
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from app import logging_setup as ls

REPO = Path(__file__).resolve().parent.parent
TOKEN = "123456789:AAHfakeFAKEfake-fake_fakeFAKEfakeFAKEfa"
API_KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"


_touched: list = []


@pytest.fixture
def scrubbing():
    """Install the scrubbing with known secrets; restore every global it touches."""
    import threading

    old_factory = logging.getLogRecordFactory()
    old_hooks = (sys.excepthook, threading.excepthook)
    levels = {n: logging.getLogger(n).level for n in ("httpx", "httpcore")}
    _touched.clear()
    ls.install_redaction(secrets=[API_KEY, "Test-Only-Passw0rd-Value", "abc"])
    yield
    logging.setLogRecordFactory(old_factory)
    sys.excepthook, threading.excepthook = old_hooks
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    for logger, handlers, level, propagate in _touched:
        logger.handlers, logger.level, logger.propagate = handlers, level, propagate
    _touched.clear()


def _capture(logger_name: str = "test.redaction", level: int = logging.DEBUG):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    logger = logging.getLogger(logger_name)
    _touched.append((logger, logger.handlers, logger.level, logger.propagate))
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
    return logger, stream


# --- what is redacted ---


def test_a_telegram_token_in_a_url_is_redacted(scrubbing):
    logger, out = _capture()
    logger.info("HTTP Request: POST https://api.telegram.org/bot%s/getMe", TOKEN)
    text = out.getvalue()
    assert TOKEN not in text and "AAHfake" not in text
    assert "https://api.telegram.org/bot<redacted>/getMe" in text


def test_a_token_is_redacted_in_the_message_its_arguments_and_the_traceback(scrubbing):
    logger, out = _capture()
    logger.warning(f"literal {TOKEN} in the message")
    logger.warning("as an argument: %s and %r", TOKEN, {"url": f"/bot{TOKEN}/x"})
    try:
        raise RuntimeError(f"request to https://api.telegram.org/bot{TOKEN}/sendMessage failed")
    except RuntimeError:
        logger.exception("delivery failed")
    text = out.getvalue()
    assert TOKEN not in text and "AAHfake" not in text
    assert text.count("<redacted>") >= 4
    assert "Traceback" in text and "RuntimeError" in text, "the traceback itself is kept"


def test_a_stack_info_block_is_scrubbed_too(scrubbing):
    """stack_info prints the source lines of the call stack: a token written in one is redacted.
    The line is built at run time so no formatter can move the token out of it.
    """
    import linecache

    logger, out = _capture()
    source = f'log.warning("with a stack", stack_info=True)  # {TOKEN}\n'
    linecache.cache["<leak>"] = (len(source), None, [source], "<leak>")
    exec(compile(source, "<leak>", "exec"), {"log": logger})  # noqa: S102
    text = out.getvalue()
    assert "Stack (most recent call last)" in text and "with a stack" in text
    assert '"<leak>"' in text and "<redacted>" in text, "the source line was really in the stack"
    assert TOKEN not in text and "AAHfake" not in text


def test_configured_secrets_are_redacted_by_their_literal_value(scrubbing):
    logger, out = _capture()
    logger.error("key=%s pw=Test-Only-Passw0rd-Value end", API_KEY)
    text = out.getvalue()
    assert API_KEY not in text and "Test-Only-Passw0rd-Value" not in text
    assert text.count("<redacted>") == 2


def test_a_very_short_secret_is_ignored_and_ordinary_text_is_untouched(scrubbing):
    logger, out = _capture()
    logger.info("abc stays, so do 12:30:45, 2026-09-21T10:00:00 and http://localhost:8080/x")
    text = out.getvalue()
    assert "abc stays" in text and "12:30:45" in text and "http://localhost:8080/x" in text
    assert "<redacted>" not in text


def test_a_message_that_cannot_be_formatted_is_left_alone_and_never_raises(scrubbing):
    logger, out = _capture()
    logger.info("%d items", "not a number")  # logging reports its own formatting error
    logging.raiseExceptions = False
    try:
        logger.info("%d items", "not a number")
    finally:
        logging.raiseExceptions = True


# --- it holds for every handler and logger ---


def test_a_handler_added_after_installation_still_gets_scrubbed_records(scrubbing):
    """uvicorn rebuilds its handlers when it starts: the scrubbing must not depend on them."""
    logger, _ = _capture("uvicorn.error")
    late = io.StringIO()
    logger.addHandler(logging.StreamHandler(late))
    logger.info("token %s", TOKEN)
    assert TOKEN not in late.getvalue() and "<redacted>" in late.getvalue()


def test_every_logger_is_covered_not_only_the_application_one(scrubbing):
    for name in ("channelagent", "httpx", "telegram.ext.Application", "uvicorn.access", "alembic"):
        logger, out = _capture(name)
        logger.warning("t=%s", TOKEN)
        assert TOKEN not in out.getvalue(), name


def test_installing_twice_does_not_wrap_the_factory_again(scrubbing):
    factory = logging.getLogRecordFactory()
    ls.install_redaction(secrets=[API_KEY])
    assert logging.getLogRecordFactory() is factory or getattr(
        logging.getLogRecordFactory(), "_redacting", False
    )
    logger, out = _capture()
    logger.info("%s", TOKEN)
    assert out.getvalue().count("<redacted>") == 1


# --- records must keep working with structured formatters ---


def _uvicorn_access_logger():
    """uvicorn's own access logger with uvicorn's own formatter, as it runs in production."""
    from uvicorn.logging import AccessFormatter

    stream = io.StringIO()
    problems: list = []
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False)
    )
    handler.handleError = lambda record: problems.append(record)
    logger = logging.getLogger("uvicorn.access")
    _touched.append((logger, logger.handlers, logger.level, logger.propagate))
    logger.handlers, logger.propagate = [handler], False
    logger.setLevel(logging.INFO)
    return logger, stream, problems


def test_a_uvicorn_access_line_is_formatted_and_not_an_error(scrubbing):
    logger, stream, problems = _uvicorn_access_logger()
    logger.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", "/users", "1.1", 401)
    assert problems == [], "the formatter could not read the record"
    assert stream.getvalue().strip().startswith('127.0.0.1:5000 - "GET /users HTTP/1.1" 401')


def test_a_uvicorn_access_line_holding_a_token_is_redacted_and_still_formatted(scrubbing):
    logger, stream, problems = _uvicorn_access_logger()
    logger.info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "POST", f"/bot{TOKEN}/getMe", "1.1", 200
    )
    assert problems == []
    text = stream.getvalue()
    assert TOKEN not in text and "AAHfake" not in text
    assert 'POST /bot<redacted>/getMe HTTP/1.1" 200' in text


def test_a_record_with_nothing_to_scrub_keeps_its_message_and_arguments(scrubbing):
    seen = []

    class Grab(logging.Handler):
        def emit(self, record):
            seen.append((record.msg, record.args))

    logger = logging.getLogger("test.untouched")
    _touched.append((logger, logger.handlers, logger.level, logger.propagate))
    logger.handlers, logger.propagate = [Grab()], False
    logger.setLevel(logging.INFO)
    logger.info("%s items in %s", 3, "a room")
    assert seen == [("%s items in %s", (3, "a room"))]


def test_a_record_with_a_secret_in_a_non_string_argument_is_still_scrubbed(scrubbing):
    logger, out = _capture()

    class Holder:
        def __str__(self):
            return f"holder({TOKEN})"

    logger.info("value %s", Holder())
    assert TOKEN not in out.getvalue() and "<redacted>" in out.getvalue()


def test_a_mapping_style_record_is_scrubbed(scrubbing):
    logger, out = _capture()
    logger.info("who %(who)s", {"who": TOKEN})
    assert TOKEN not in out.getvalue() and "<redacted>" in out.getvalue()


# --- uncaught exceptions are printed by the interpreter, not by logging ---


def test_an_uncaught_exception_is_printed_scrubbed(scrubbing, capsys):
    try:
        try:
            raise ValueError(f"Unauthorized for /bot{TOKEN}/getMe")
        except ValueError as cause:
            raise RuntimeError(f"The token `{TOKEN}` was rejected by the server.") from cause
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    err = capsys.readouterr().err
    assert TOKEN not in err and "AAHfake" not in err
    assert "RuntimeError" in err and "ValueError" in err and err.count("<redacted>") >= 2


def test_an_exception_in_a_thread_is_printed_scrubbed(scrubbing, capsys):
    import threading

    def boom():
        raise RuntimeError(f"token {TOKEN} in a thread")

    thread = threading.Thread(target=boom, name="worker-1")
    thread.start()
    thread.join()
    err = capsys.readouterr().err
    assert TOKEN not in err and "worker-1" in err and "RuntimeError" in err


# --- httpx, the source of the leak ---


def test_httpx_request_lines_are_off_at_the_default_level(scrubbing):
    ls.configure_logging()
    for name in ("httpx", "httpcore"):
        logger = logging.getLogger(name)
        assert logger.level == logging.WARNING, "set on the logger itself, not inherited"
        assert not logger.isEnabledFor(logging.INFO)


async def test_a_real_httpx_request_to_the_bot_api_leaves_no_token_even_at_info(scrubbing):
    """Force the level back to INFO to prove the filter, not the level, is the safety net."""
    ls.configure_logging()
    logging.getLogger("httpx").setLevel(logging.INFO)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(f"https://api.telegram.org/bot{TOKEN}/getMe")
    finally:
        root.removeHandler(handler)
    text = stream.getvalue()
    assert "api.telegram.org/bot" in text, "the request line was really logged"
    assert TOKEN not in text and "AAHfake" not in text
    assert "bot<redacted>/getMe" in text


# --- the settings ---


def test_the_configured_secrets_come_from_the_settings(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("API_SERVER_KEY", API_KEY)
    monkeypatch.setenv("EMAIL_PASSWORD", "Test-Only-Passw0rd-Value")
    monkeypatch.setenv("MATRIX_BOT_ACCESS_TOKEN", "syt_matrix_token_value")
    monkeypatch.setenv("OLD_ENCRYPTION_KEY", "old-key-given-to-a-rotation-000")
    get_settings.cache_clear()
    secrets = ls.configured_secrets()
    assert {TOKEN, API_KEY, "Test-Only-Passw0rd-Value", "syt_matrix_token_value"} <= set(secrets)
    assert "old-key-given-to-a-rotation-000" in secrets
    assert os.environ["ENCRYPTION_KEY"] in secrets
    assert all(len(s) >= ls.MIN_SECRET_LENGTH for s in secrets)
    get_settings.cache_clear()


# --- the real application ---


def test_the_real_application_never_prints_a_token_shaped_secret(tmp_path):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_", "CHECKPOINT_"))}
    env.update(
        PYTHONPATH=str(REPO),
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path}/data/channelagent.db",
        TELEGRAM_BOT_TOKEN=TOKEN,
        EMAIL_IMAP_HOST="",
        API_SERVER_KEY=API_KEY,
        API_SERVER_PORT=str(port),
        MIGRATION_BACKUPS_KEEP="0",
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.main"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    answered = None
    deadline = time.time() + 30
    while time.time() < deadline and answered is None:  # start-up and the first Bot API calls
        try:
            answered = httpx.get(f"http://127.0.0.1:{port}/users", timeout=2).status_code
        except httpx.HTTPError:
            time.sleep(0.5)
    time.sleep(6)
    proc.send_signal(signal.SIGINT)
    try:
        output, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
    assert "ChannelAgent started" in output, output[-500:]
    assert answered == 401
    assert TOKEN not in output and "AAHfake" not in output
    assert API_KEY not in output
    assert '"GET /users HTTP/1.1" 401' in output, "the access line is printed"
    assert "Logging error" not in output, output[-800:]


def test_every_entry_point_installs_the_scrubbing():
    for path in ("app/main.py", "app/admin/cli.py", "app/admin/rekey.py", "app/admin/restore.py"):
        text = (REPO / path).read_text()
        assert "install_redaction()" in text or "configure_logging()" in text, path
