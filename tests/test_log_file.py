"""The application's log goes to the terminal and to LOG_FILE as well (
"je veux un retour terminal et logs fichiers"): rotated, mode 600, secrets masked as in the
terminal, uvicorn's lines included, and a file that cannot be written never stops the app.
"""

import logging
import logging.handlers
import stat

import pytest

from app import logging_setup as ls

SECRET = "api-key-for-the-log-file-test-" + "x" * 16


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    from app.config import get_settings

    path = tmp_path / "logs" / "channelagent.log"
    monkeypatch.setenv("LOG_FILE", str(path))
    monkeypatch.setenv("API_SERVER_KEY", SECRET)
    get_settings.cache_clear()
    ls._file_handler = None
    yield path
    handler = ls._file_handler
    for name in ("", "uvicorn", "uvicorn.access"):
        if handler is not None and handler in logging.getLogger(name).handlers:
            logging.getLogger(name).removeHandler(handler)
    if handler is not None:
        handler.close()
    ls._file_handler = None
    ls.install_redaction([])
    get_settings.cache_clear()


def _flush():
    if ls._file_handler is not None:
        ls._file_handler.flush()


def test_a_line_goes_to_the_terminal_and_to_the_file(log_file, capsys):
    ls.configure_logging()
    logging.getLogger("channelagent").warning("a turn failed for telegram/55")
    _flush()
    assert "a turn failed for telegram/55" in log_file.read_text()
    assert logging.getLogger().handlers, "the terminal handler is still there"
    assert any(not isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers)


def test_the_file_is_readable_by_its_owner_only(log_file):
    ls.configure_logging()
    assert stat.S_IMODE(log_file.stat().st_mode) == 0o600


def test_a_secret_is_masked_in_the_file(log_file):
    ls.configure_logging()
    logging.getLogger("channelagent").warning("key was %s", SECRET)
    _flush()
    text = log_file.read_text()
    assert SECRET not in text and ls.REDACTED in text


def test_uvicorn_lines_reach_the_file(log_file):
    ls.configure_logging()
    access = logging.getLogger("uvicorn.access")
    # An earlier test may have run alembic's fileConfig, which disables existing loggers.
    # An earlier test may have disabled it (alembic's fileConfig) or left it at ERROR (measured
    # in the full suite: level 40): set it as uvicorn does, INFO.
    was_disabled, access.disabled = access.disabled, False
    was_level = access.level
    access.setLevel(logging.INFO)
    access.propagate = False  # as uvicorn sets it
    try:
        ls.add_file_logging(["uvicorn", "uvicorn.access"])
        access.warning("GET /status 200")
        _flush()
        assert "GET /status 200" in log_file.read_text()
    finally:
        access.propagate = True
        access.disabled = was_disabled
        access.setLevel(was_level)


def test_the_file_is_rotated_and_old_files_kept(log_file):
    ls.configure_logging()
    handler = ls._file_handler
    assert handler.maxBytes == ls.LOG_FILE_MAX_BYTES == 10 * 1024 * 1024
    assert handler.backupCount == ls.LOG_FILE_BACKUPS == 5


def test_an_empty_log_file_means_the_terminal_only(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("LOG_FILE", "")
    get_settings.cache_clear()
    ls._file_handler = None
    try:
        ls.configure_logging()
        assert ls._file_handler is None
        # pytest adds its own file handler (to /dev/null): only a rotated file is ours.
        rotated = logging.handlers.RotatingFileHandler
        assert not any(isinstance(h, rotated) for h in logging.getLogger().handlers)
    finally:
        get_settings.cache_clear()


def test_a_file_that_cannot_be_written_never_stops_the_app(tmp_path, monkeypatch, caplog):
    from app.config import get_settings

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    monkeypatch.setenv("LOG_FILE", str(blocker / "channelagent.log"))
    get_settings.cache_clear()
    ls._file_handler = None
    try:
        with caplog.at_level(logging.WARNING, logger="channelagent"):
            ls.configure_logging()
        assert ls._file_handler is None
        assert "cannot be written" in caplog.text
    finally:
        get_settings.cache_clear()


def test_an_existing_log_file_readable_by_others_is_closed_again(log_file):
    log_file.parent.mkdir(parents=True)
    log_file.write_text("older lines\n")
    log_file.chmod(0o644)
    ls.configure_logging()
    assert stat.S_IMODE(log_file.stat().st_mode) == 0o600
    assert log_file.read_text().startswith("older lines"), "appended, not truncated"
