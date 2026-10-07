"""No secret reaches a log.

The Telegram bot token is part of every Bot API URL (`/bot<token>/getMe`) and
httpx logs the URL of each request at INFO: with the adapter running, the
complete token was written to the terminal and to `docker logs`, one line per
poll. Two layers:

1. `httpx` and `httpcore` do not log at INFO (their line adds nothing the
   adapter's own lines do not say);
2. every log record is scrubbed **when it is created**, whatever logger emits it
   and whatever handler prints it (uvicorn rebuilds its own handlers when it
   starts, so a filter on handlers would not hold): anything shaped like a
   Telegram token, and the literal value of every configured secret, becomes
   `<redacted>` in the message, its arguments, the traceback and the stack.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import traceback

REDACTED = "<redacted>"
MIN_SECRET_LENGTH = 8  # a shorter value would redact ordinary words
_TELEGRAM_TOKEN = re.compile(r"[0-9]{5,}:[A-Za-z0-9_-]{20,}")

_secrets: tuple[str, ...] = ()


def configured_secrets() -> list[str]:
    """The literal values that must never be printed: the configured secrets and
    the old key a rotation is given through the environment.
    """
    from app.config import get_settings

    settings = get_settings()
    values = [
        settings.telegram_bot_token,
        settings.api_server_key,
        settings.encryption_key,
        settings.email_password,
        settings.matrix_bot_access_token,
        settings.hf_token,
        settings.llama_server_api_key,
        os.environ.get("OLD_ENCRYPTION_KEY"),
    ]
    return [v for v in values if v and len(v) >= MIN_SECRET_LENGTH]


def scrub(text: str) -> str:
    for secret in _secrets:
        text = text.replace(secret, REDACTED)
    return _TELEGRAM_TOKEN.sub(REDACTED, text)


def _replace_message(record: logging.LogRecord, scrubbed: str) -> None:
    """Make `record` print `scrubbed` instead of its message.

    The record keeps its structure (a format string and a tuple of arguments)
    when it can: uvicorn's access formatter reads its five values from
    `record.args`, and flattening them to None broke every access line.
    It is flattened to plain text only when a secret would still be visible
    after scrubbing the string parts (a secret inside a non-string argument, a
    mapping-style record).
    """
    if isinstance(record.args, tuple) and record.args and isinstance(record.msg, str):
        old_msg, old_args = record.msg, record.args
        record.msg = scrub(old_msg)
        record.args = tuple(scrub(a) if isinstance(a, str) else a for a in old_args)
        try:
            formatted = record.getMessage()
        except Exception:
            formatted = None
        if formatted is not None and scrub(formatted) == formatted:
            return
        record.msg, record.args = old_msg, old_args
    record.msg = scrubbed
    record.args = None


def _scrub_record(record: logging.LogRecord) -> None:
    message = record.getMessage()  # may raise on a bad format: the caller leaves the record alone
    scrubbed = scrub(message)
    if scrubbed != message:  # nothing to hide: the record is left exactly as it was
        _replace_message(record, scrubbed)
    if record.exc_info and not record.exc_text:
        record.exc_text = scrub(logging.Formatter().formatException(record.exc_info))
    if record.stack_info:
        record.stack_info = scrub(record.stack_info)


def install_redaction(secrets: list[str] | None = None) -> None:
    """Scrub every record from now on. `secrets` defaults to the configured ones.
    Calling it again only updates the list of secrets.
    """
    global _secrets
    if secrets is None:
        try:
            secrets = configured_secrets()
        except Exception:  # settings that cannot load are reported by the app itself
            secrets = []
    _secrets = tuple(
        sorted({s for s in secrets if s and len(s) >= MIN_SECRET_LENGTH}, key=len, reverse=True)
    )
    current = logging.getLogRecordFactory()
    if getattr(current, "_redacting", False):
        return

    def factory(*args, **kwargs) -> logging.LogRecord:
        record = current(*args, **kwargs)
        try:
            _scrub_record(record)
        except Exception:  # logging must never break the caller
            pass
        return record

    factory._redacting = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)
    _install_excepthooks()


def _install_excepthooks() -> None:
    """An uncaught exception is printed by the interpreter, not by logging: an
    invalid Telegram token made the app end with `The token <token> was rejected
    by the server` in a raw traceback. Print those scrubbed too.
    """
    if not getattr(sys.excepthook, "_redacting", False):
        previous = sys.excepthook

        def hook(exc_type, exc, tb) -> None:
            try:
                sys.stderr.write(scrub("".join(traceback.format_exception(exc_type, exc, tb))))
            except Exception:
                previous(exc_type, exc, tb)

        hook._redacting = True  # type: ignore[attr-defined]
        sys.excepthook = hook
    if not getattr(threading.excepthook, "_redacting", False):
        previous_thread = threading.excepthook

        def thread_hook(args) -> None:
            try:
                text = "".join(
                    traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
                )
                sys.stderr.write(f"Exception in thread {getattr(args.thread, 'name', '?')}:\n")
                sys.stderr.write(scrub(text))
            except Exception:
                previous_thread(args)

        thread_hook._redacting = True  # type: ignore[attr-defined]
        threading.excepthook = thread_hook


def configure_logging(level: int = logging.INFO) -> None:
    """The logging setup of the long-running application (`app/main.py`): the terminal, and
    the file LOG_FILE as well ("je veux un retour terminal et logs
    fichiers")."""
    logging.basicConfig(level=level)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    install_redaction()
    add_file_logging([""])


# The log file (LOG_FILE): rotated at LOG_FILE_MAX_BYTES, LOG_FILE_BACKUPS old files kept,
# readable by its owner only (a log names users and channels). A record is scrubbed when
# it is created (install_redaction), so the file never holds more than the terminal.
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUPS = 5
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_file_handler: logging.Handler | None = None


def _open_file_handler() -> logging.Handler | None:
    global _file_handler
    if _file_handler is not None:
        return _file_handler
    from logging.handlers import RotatingFileHandler

    from app.config import get_settings

    try:
        path = (get_settings().log_file or "").strip()
    except Exception:
        return None
    if not path:
        return None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        old_umask = os.umask(0o077)
        try:
            handler = RotatingFileHandler(
                path, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUPS, encoding="utf-8"
            )
        finally:
            os.umask(old_umask)
        os.chmod(path, 0o600)
    except OSError as exc:
        # The terminal still has every line: a log file that cannot be written never stops the
        # application (a read-only directory in a container, a full disk).
        logging.getLogger("channelagent").warning(
            "The log file %s cannot be written: %s", path, exc
        )
        return None
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    _file_handler = handler
    return handler


def add_file_logging(logger_names: list[str]) -> None:
    """Also write these loggers to LOG_FILE ("" is the root logger). uvicorn sets its own
    handlers on "uvicorn" and "uvicorn.access", which do not reach the root logger: it is
    called for them once uvicorn is configured."""
    handler = _open_file_handler()
    if handler is None:
        return
    for name in logger_names:
        target = logging.getLogger(name)
        if handler not in target.handlers:
            target.addHandler(handler)


# A client following a job asks for it every POLL_SECONDS (0.3 s, `start.sh`) or every second
# (the admin UI): uvicorn printed one access line per question, about 3 lines a second in the
# owner's terminal, saying nothing the job itself does not record ("je ne
# vois pas a quoi ca sert"). A successful poll is dropped; an error and every other request
# keep their line.
_JOB_POLL = re.compile(r"^(/jobs/[^/?]+|/ui/jobs/[^/?]+/status)(\?.*)?$")


class _JobPollFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        # uvicorn's access record: (client, method, path with query, http version, status).
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        _client, method, path, _version, status = args
        if method != "GET" or not isinstance(path, str) or not _JOB_POLL.match(path):
            return True
        return not (isinstance(status, int) and status < 400)


def quiet_job_polls() -> None:
    """Drop the access line of every successful job poll. Called once uvicorn has configured
    its loggers; a filter on the logger holds for the terminal and LOG_FILE alike."""
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _JobPollFilter) for f in access.filters):
        access.addFilter(_JobPollFilter())
